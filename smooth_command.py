import torch
import genesis as gs
from genesis_forge.managers import PositionActionManager
from genesis_forge.managers.command.velocity_command import VelocityCommandManager


def ramp_toward(current: torch.Tensor, target: torch.Tensor, max_delta: torch.Tensor) -> torch.Tensor:
    """Mueve current hacia target sin dar un paso mayor que max_delta (por componente)."""
    return current + torch.clamp(target - current, -max_delta, max_delta)


class SmoothVelocityCommandManager(VelocityCommandManager):
    """
    Igual que VelocityCommandManager, pero el muestreo solo define el objetivo.
    El comando efectivo (el que ve la politica y la recompensa) se acerca al objetivo
    con una aceleracion maxima, asi un cambio de orden (p. ej. correr -> caminar)
    se nota como desaceleracion y no como un salto.
    Limitacion: solo suaviza el muestreo interno; un controlador externo (gamepad)
    escribe directamente el comando y no pasa por la rampa.
    """

    def __init__(self, env, range, max_accel: dict, resample_time_sec: float = 5.0, **kwargs):
        super().__init__(env, range=range, resample_time_sec=resample_time_sec, **kwargs)
        # Orden de componentes = orden de las claves de range
        self._max_accel = torch.tensor(
            [float(max_accel[k]) for k in range.keys()], device=gs.device
        )
        self._target = self._command.clone()

    def resample_command(self, env_ids):
        # El muestreo base escribe en _command: lo guardamos como objetivo y
        # conservamos el comando efectivo actual
        effective = self._command[env_ids].clone()
        super().resample_command(env_ids)
        self._target[env_ids] = self._command[env_ids]
        self._command[env_ids] = effective

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=gs.device)
        # Tras un reset el comando empieza ya en el objetivo, sin rampa
        self._command[env_ids] = self._target[env_ids]

    def step(self):
        super().step()
        if not self.enabled:
            return
        max_delta = self._max_accel * self.env.dt
        self._command[:] = ramp_toward(self._command, self._target, max_delta)


class FilteredPositionActionManager(PositionActionManager):
    """
    PositionActionManager con filtro paso bajo (media movil exponencial) sobre la posicion
    objetivo que se envia a los actuadores. alpha=1 desactiva el filtro.
    Filtrar la posicion ya escalada equivale a filtrar la accion cruda (la EMA conmuta con
    escala + desplazamiento fijos) si el estado arranca en la postura por defecto.
    Con dt=1/50 s, alpha=0.5 corta en ~6 Hz: deja pasar la marcha (~2 Hz) y quita tirones.
    """

    def __init__(self, env, alpha: float = 0.5, **kwargs):
        super().__init__(env, **kwargs)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha debe estar en (0, 1], recibido {alpha}")
        self._alpha = alpha
        self._filtered = None

    def _default_like(self, x: torch.Tensor) -> torch.Tensor:
        return self.default_dofs_pos.expand_as(x).clone()

    def send_actions_to_simulation(self, actions: torch.Tensor) -> torch.Tensor:
        if self._filtered is None:
            self._filtered = self._default_like(actions)
        self._filtered = self._alpha * actions + (1.0 - self._alpha) * self._filtered
        return super().send_actions_to_simulation(self._filtered)

    def reset(self, envs_idx=None):
        super().reset(envs_idx)
        if self._filtered is not None:
            default = self._default_like(self._filtered)
            if envs_idx is None:
                self._filtered[:] = default
            else:
                self._filtered[envs_idx] = default[envs_idx]
