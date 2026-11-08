import genesis as gs
import torch
import os
from PIL import Image
import numpy as np

from genesis_forge import ManagedEnvironment, EnvMode
from genesis_forge.managers import (
    RewardManager,
    TerminationManager,
    EntityManager,
    ObservationManager,
    ActuatorManager,
    PositionActionManager,
    VelocityCommandManager,
    TerrainManager,
    ContactManager
)
from genesis_forge.mdp import reset, terminations, observations, rewards
from smooth_command import SmoothVelocityCommandManager, FilteredPositionActionManager

# Alturas de la base (m). En la postura por defecto la base queda a 0.117 m sobre las suelas
# y en reposo a ~0.115 m. Con 0.09 el robot aparecia con los pies 27 mm bajo el suelo, el
# terreno no los detectaba y el robot se apoyaba en las piernas.
SPAWN_HEIGHT       = 0.12    # aparicion: suelas unos mm sobre el suelo
BASE_HEIGHT_TARGET = 0.115   # objetivo de la recompensa base_height (medido de pie)
INITIAL_BODY_POSITION = [0.0, 0.0, SPAWN_HEIGHT]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]

# Perturbaciones de entrenamiento (solo mode="train")
PUSH_INTERVAL_S = 3.0   # cada cuanto se empuja la base
PUSH_MAX_VEL    = 0.3   # cambio maximo de velocidad lineal xy por empujon (m/s)


def randomize_friction(env, entity, envs_idx, ratio_range):
    """Multiplica la friccion de todos los links por un factor aleatorio por entorno."""
    ratios = torch.empty((len(envs_idx), 1), device=gs.device).uniform_(*ratio_range)
    entity.set_friction_ratio(
        ratios.expand(-1, entity.n_links),
        links_idx_local=list(range(entity.n_links)),
        envs_idx=envs_idx,
    )

def pose_to_T(pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
    """Convierte posicion (x,y,z) y cuaternion (w,x,y,z) a matriz homogenea 4x4."""
    w, x, y, z = quat
    R = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = pos
    return T


class DuckDuckEnv(ManagedEnvironment):
    def __init__(
        self,
        num_envs:             int        = 1,
        dt:                   float      = 1 / 50,
        max_episode_length_s: int | None = 20,
        headless:             bool       = True,
        mode: EnvMode = "train",
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            max_episode_random_scaling=0.1,
        )

        self._mode = mode
        self._curriculum_level = 0

        self.scene = gs.Scene(
            show_viewer=not headless,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            # Solo batching por entorno (necesario para aleatorizar la masa por entorno en
            # genesis-world >= 1.4); el resto de RigidOptions queda en sus valores por defecto
            rigid_options=gs.options.RigidOptions(batch_links_info=True),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, 1.5, 1.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=45,
                refresh_rate= int(0.5/self.dt)
            ),
            vis_options=gs.options.VisOptions(
                shadow=False,
                plane_reflection=False,
                rendered_envs_idx=list(range(1)),
            ),
        )

        self.terrain = self.create_terrain(self.scene) #type:ignore

        self.robot = self.scene.add_entity( #type:ignore
            gs.morphs.MJCF(
                file="model/robot.xml",
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ),
        )

        self.camera = self.scene.add_camera(
            pos=(1.5, 1.5, 1.0),
            lookat=(0.0, 0.0, 0.1),
            fov=45,
            res=(640, 480),
            env_idx=0,
            debug=True,
        )

        self.robot_camera = self.scene.add_camera(
            pos=(0.0, 0.0, 0.0),
            lookat=(0.0, 1.0, 0.0),
            fov=45,
            res=(640, 480),
            env_idx=0,
            debug=True,
        )

    def config(self):

        self.terrain_manager = TerrainManager(self)

        on_reset = {
            "position": {
                "fn": reset.randomize_terrain_position(
                    height_offset=SPAWN_HEIGHT,
                    terrain_manager=self.terrain_manager,
                ),
            },
        }
        if self._mode == "train":
            # Aleatorizacion de dominio en cada reset: masa del tronco y friccion
            on_reset["mass"] = {
                "fn": reset.randomize_link_mass_shift(link_name="trunk_base", mass_range=(-0.02, 0.04)),
            }
            on_reset["friction"] = {
                "fn": randomize_friction,
                "params": {"ratio_range": (0.7, 1.3)},
            }

        self.robot_manager = EntityManager(
            self,
            entity=self.robot,
            on_reset=on_reset, #type:ignore
        )

        self.actuator_manager = ActuatorManager(
            self,
            joint_names=[
                "left_hip.*", "left_knee", "left_ankle",
                "right_hip.*", "right_knee", "right_ankle",
                "neck_pitch", "head_pitch", "head_yaw", "head_roll"
            ],
            default_pos={
                "left_hip_yaw": 0.0,
                "left_hip_roll": -0.08726646259971647,
                "left_hip_pitch": -0.457924,
                "left_knee": -0.004940,
                "left_ankle": 0.452984,
                "neck_pitch": 0.3490658503988659,
                "head_pitch": 0.3490658503988659,
                "head_yaw": 0.0,
                "head_roll": 0.0,
                "right_hip_yaw": 0.0,
                "right_hip_roll": 0.08726646259971647,
                "right_hip_pitch": 0.457924,
                "right_knee": 0.004940,
                "right_ankle": -0.452984,
            },
            kp= 0.55,
            kv= 0.0
        )

        self.action_manager = FilteredPositionActionManager(
            self,
            alpha=0.5,
            scale= 0.5,
            use_default_offset=True,
            actuator_manager=self.actuator_manager,
        )

        self.velocity_command = SmoothVelocityCommandManager(
            self,
            range={ #type:ignore
                "lin_vel_x": [0.0, 0.6],   # pierna 0.0975 m: marcha posible hasta ~0.69 m/s (Froude 0.5)
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_z": [-0.5, 0.5],
            },
            # Aceleracion maxima del comando efectivo (unidades/s^2) por componente
            max_accel={"lin_vel_x": 1.0, "lin_vel_y": 1.0, "ang_vel_z": 2.0},
            resample_time_sec=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg = { #type:ignore
                "envs_idx":[0],        
            },
        )

        self.feet_contact_manager = ContactManager(
            self,
            link_names=["ankle_left", "ankle_right"],
            track_air_time=True,
        )

        RewardManager(
            self,
            logging_enabled=True,
            cfg={ #type:ignore
                "is_alive": {
                    "weight": 0.5,
                    "fn": rewards.is_alive(),
                },
                "tracking_lin_vel": {
                    "weight": 1.5,
                    # sensitivity 0.05: con 0.25, quedarse quieto ante 0.3 m/s ya daba el 70%
                    "fn": rewards.command_tracking_lin_vel(
                        entity_manager=self.robot_manager,
                        vel_cmd_manager=self.velocity_command,
                        sensitivity=0.05,
                    ),
                },
                "tracking_ang_vel": {
                    "weight": 0.5,
                    # explicito: en genesis-forge 1.0 None la deriva del rango de comandos
                    "fn": rewards.command_tracking_ang_vel(
                        entity_manager=self.robot_manager,
                        vel_cmd_manager=self.velocity_command,
                        sensitivity=0.25,
                    ),
                },
                "feet_air_time": {
                    "weight": 1.0,
                    "fn": rewards.feet_air_time(
                        contact_manager=self.feet_contact_manager,
                        time_threshold=0.2,
                        time_threshold_max=0.5,
                        vel_cmd_manager=self.velocity_command,
                    ),
                },
                "feet_slide": {
                    "weight": -0.2,
                    "fn": rewards.feet_slide(contact_manager=self.feet_contact_manager),
                },
                "ang_vel_xy": {
                    "weight": -0.3,
                    "fn": rewards.ang_vel_xy_l2(entity_manager=self.robot_manager),
                },
                "lin_vel_z": {
                    "weight": -0.2,
                    "fn": rewards.lin_vel_z_l2(entity_manager=self.robot_manager),
                },
                "flat_orientation": {
                    "weight": -0.5,
                    "fn": rewards.flat_orientation_l2(entity_manager=self.robot_manager),
                },
                "action_rate": {
                    "weight": -0.2,
                    "fn": rewards.action_rate_l2(),
                },
                "base_height": {
                    # error en m^2: con -0.5, 1 cm de error pesaba 5e-5 y el termino no hacia nada
                    "weight": -50.0,
                    "fn": rewards.base_height(
                        target_height=BASE_HEIGHT_TARGET,
                        terrain_manager=self.terrain_manager,
                        entity_manager=self.robot_manager,
                    ),
                },
                "stand_still": {
                    "weight": -0.5,
                    "fn": rewards.stand_still_joint_deviation_l1(
                        vel_cmd_manager=self.velocity_command,
                        actuator_manager=self.actuator_manager,
                    ),
                },
                "contact_force": {
                    "weight": -0.01,
                    # umbral ~1.5x el peso (7.2 N): penaliza impactos, no el apoyo normal
                    "fn": rewards.contact_force(contact_manager=self.feet_contact_manager, threshold=11.0),
                },
            },
        )

        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={ #type:ignore
                "timeout": {
                    "fn": terminations.timeout(),
                    "time_out": True,
                },

                "out_of_bounds": {
                    "fn": terminations.out_of_bounds(terrain_manager=self.terrain_manager),
                },

                "bad_orientation": {
                    "fn": terminations.bad_orientation(
                        limit_angle=30.0,
                        entity_manager=self.robot_manager,
                        grace_steps=20,
                    ),
                },
            },
        )

        ObservationManager(
            self,
            name="policy",
            cfg={ #type:ignore
                "velocity_cmd": {"fn": self.velocity_command.observation},
                "angle_velocity": {
                    "fn": lambda env: self.robot_manager.get_angular_velocity(),
                },
                "linear_velocity": {
                    "fn": lambda env: self.robot_manager.get_linear_velocity(),
                },
                "projected_gravity": {
                    "fn": lambda env: self.robot_manager.get_projected_gravity(),
                },
                "actions": {
                    "fn": lambda env: self.action_manager.get_actions(),
                },
            },
        )

        ObservationManager(
            self,
            name="critic",
            cfg={ #type:ignore
                "velocity_cmd": {"fn": self.velocity_command.observation},
                "angle_velocity": {
                    "fn": lambda env: self.robot_manager.get_angular_velocity(),
                },
                "linear_velocity": {
                    "fn": lambda env: self.robot_manager.get_linear_velocity(),
                },
                "projected_gravity": {
                    "fn": lambda env: self.robot_manager.get_projected_gravity(),
                },
                "dof_position": {
                    "fn": lambda env: self.action_manager.get_dofs_position(),
                },
                "dof_velocity": {
                    "fn": lambda env: self.action_manager.get_dofs_velocity(),
                    "scale": 0.05,
                },
                "actions": {
                    "fn": lambda env: self.action_manager.get_actions(),
                },
                "dof_force": {
                    "fn": observations.entity_dofs_force(actuator_manager=self.actuator_manager),
                    "scale": 0.01,
                },
                "contact_force": {
                    "fn": observations.contact_force(contact_manager=self.feet_contact_manager),
                    "scale": 0.01,
                },
            },
        )
    
    def create_terrain(self, scene: gs.Scene):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        tile_path = os.path.join(this_dir, "checker.png")
        img = Image.open(tile_path)
        img = img.resize((128, 128), Image.Resampling.NEAREST) 
        checker_image = np.array(img)
        tiled_image = np.tile(checker_image, (24, 24, 1))

        return scene.add_entity(
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(
                    image_array=tiled_image,
                )
            ),
            morph=gs.morphs.Terrain(
                pos=(-12, -12, 0),
                n_subterrains=(1, 1),
                subterrain_size=(24, 24),
                vertical_scale=0.001,
                subterrain_types=[["flat_terrain"]],
                subterrain_parameters={
                    "random_uniform_terrain": {
                        "min_height": 0.0,
                        "max_height": 0.1,
                        "step": 0.05,
                        "downsampled_scale": 0.25,
                    },
                },
            ),
        )

    def step(self, actions):
        result = super().step(actions)
        if self._mode == "train":
            self._apply_pushes()
        return result

    def _apply_pushes(self):
        """Empujon: suma una velocidad lineal xy aleatoria a la base cada PUSH_INTERVAL_S."""
        push_steps = max(1, int(PUSH_INTERVAL_S / self.dt))
        envs_idx = (
            (self.episode_length > 0) & (self.episode_length % push_steps == 0)
        ).nonzero(as_tuple=False).flatten()
        if len(envs_idx) == 0:
            return
        # DOFs locales 0 y 1 = traslacion x, y del freejoint de la base
        vel = self.robot.get_dofs_velocity(dofs_idx_local=[0, 1], envs_idx=envs_idx)
        vel = vel + torch.empty_like(vel).uniform_(-PUSH_MAX_VEL, PUSH_MAX_VEL)
        self.robot.set_dofs_velocity(vel, dofs_idx_local=[0, 1], envs_idx=envs_idx)

    def build(self):
        super().build()
        self.camera.follow_entity(self.robot)

        cam_pos  = (0.0155, -9.13778e-05, -0.0733)
        cam_quat = (-0.7071068, 0, 0, 0.7071068) 
        offset_T = pose_to_T(pos=cam_pos, quat=cam_quat)
        cam_link = self.robot.get_link("jaw_soft")
        self.robot_camera.attach(cam_link, offset_T)
