import os
import glob
import argparse
import genesis as gs
from genesis_forge.wrappers import RslRlWrapper, VideoWrapper
from rsl_rl.runners import OnPolicyRunner
from environment import DuckDuckEnv

EXPERIMENT_NAME = "duckduck_locomotion"

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-n", "--num_envs",        type=int,   default=1024)
parser.add_argument("--max_iterations",        type=int,   default=6000)
parser.add_argument("-d", "--device",          type=str,   default="cpu", choices=["cpu", "gpu"])
parser.add_argument("-e", "--exp_name",        type=str,   default=EXPERIMENT_NAME)
parser.add_argument("--resume",                action="store_true",
                    help="Continuar desde el ultimo checkpoint guardado")
parser.add_argument("--resume_path",           type=str,   default=None,
                    help="Ruta exacta al .pt a cargar (si no se da, se busca automaticamente en ./logs)")
parser.add_argument("--no_video",              action="store_true", default=False,
                    help="Desactiva grabacion de video")
parser.add_argument("--verbose",               action="store_true", default=False,
                    help="Muestra logs de FPS de Genesis en consola")
args = parser.parse_args()

# ── Intervalos proporcionales a max_iterations ─────────────────────────────────
# Guarda modelos al 1%, 5%, 10%... del total (min 1 iter)
save_interval   = max(1, args.max_iterations // 100)   # cada ~1% del total
# Graba video al 5% del total (no demasiado frecuente)
record_interval = max(1,args.max_iterations // 60)

if record_interval <= 1:
    videos_deseados = min(20, args.max_iterations // 2)
    record_interval = max(1, args.max_iterations // max(1, videos_deseados))

if save_interval <= 1:
    int_deseados = min(20, args.max_iterations // 2)
    save_interval = max(1, args.max_iterations // max(1, int_deseados))

# ── Auto-descubrimiento del checkpoint más reciente ────────────────────────────
resume_path = args.resume_path
log_dir = os.path.join("logs", args.exp_name)

if args.resume:
    if resume_path is None:
        pattern = os.path.join(log_dir, "**", "model_*.pt")
        candidates = glob.glob(pattern, recursive=True)
        if candidates:
            # Ordenar por número de iteración extraído del nombre
            candidates.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]))
            resume_path = candidates[-1]
            print(f"[Resume] Cargando checkpoint: {resume_path}")
        else:
            print(f"[Resume] No se encontro ningun checkpoint en {log_dir}/. Iniciando desde cero.")
            args.resume = False
else:
    # Si no estamos resumiendo, limpiamos la carpeta de logs para evitar datos sucios
    import shutil
    if os.path.exists(log_dir):
        print(f"[Limpieza] Borrando logs del experimento anterior en {log_dir}...")
        shutil.rmtree(log_dir)

# ── Genesis ────────────────────────────────────────────────────────────────────
gs.init(
    backend=gs.gpu if args.device == "gpu" else gs.cpu, #type: ignore
    logging_level="info" if args.verbose else "warning",
)

# ── Entorno ────────────────────────────────────────────────────────────────────
env = DuckDuckEnv(num_envs=args.num_envs, headless=True)

if not args.no_video:
    env = VideoWrapper(
        env,
        video_length_sec=12,
        out_dir=os.path.join(log_dir, "videos"),
        episode_trigger=lambda episode_id: episode_id % 4 == 0,
    )
else:
    print("Video desactivado. Solo se guardaran checkpoints.")

# RslRlWrapper must remain the outermost wrapper because it changes the step
# return signature to the four values expected by rsl_rl.
env = RslRlWrapper(env)  # type: ignore
env.build()
env.reset()
env.cfg = {} # type: ignore

# ── Configuración PPO ──────────────────────────────────────────────────────────
train_cfg = {
    "seed": 42,
    "num_steps_per_env": round(98304 / env.num_envs),  # Batch size constante sin importar num_envs
    "save_interval": save_interval,
    "empirical_normalization": None,
    "torch_compile_mode": None,
    "multi_gpu": None,

    "algorithm": {
        "class_name": "PPO",
        "learning_rate":             1e-3,
        "num_learning_epochs":       5,
        "num_mini_batches":          4,
        "gamma":                     0.99,
        "lam":                       0.95,
        "clip_param":                0.2,
        "entropy_coef":              0.01,
        "value_loss_coef":           1.0,
        "max_grad_norm":             1.0,
        "desired_kl":                0.01,
        "schedule":                  "adaptive",
        "use_clipped_value_loss":    True,
        "normalize_advantage_per_mini_batch": True,
        "rnd_cfg":                   None,
        "symmetry_cfg":              None,
    },

    "actor": {
        "class_name":       "MLPModel",
        "hidden_dims":      [512, 256, 128, 64],
        "activation":       "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std":   1.0,
        },
    },

    "critic": {
        "class_name":       "MLPModel",
        "hidden_dims":      [2030, 1024, 512, 256, 128],
        "activation":       "elu",
        "obs_normalization": True,
    },

    "obs_groups": {
        "actor":  ["policy"],
        "critic": ["policy", "critic"],
    },

    "runner": {
        "experiment_name": "",
        "max_iterations":  args.max_iterations,
        "log_interval":    1,
        "checkpoint":      -1,
        "resume":          args.resume,
        "resume_path":     resume_path,
    },
    "runner_class_name": "OnPolicyRunner",
}

print(
    f"\n[Config] envs={args.num_envs} | iters={args.max_iterations} | "
    f"save_every={save_interval} | video_every={record_interval} | "
    f"resume={args.resume} | device={args.device}\n"
)

# ── Entrenamiento ──────────────────────────────────────────────────────────────
runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device) #type: ignore
runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"])

print(f"Entrenamiento completo. Revisa ./logs y ./logs/{args.exp_name}/videos.")