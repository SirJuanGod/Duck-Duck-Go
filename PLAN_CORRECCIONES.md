# Plan de correcciones: DuckDuckEnv

Revisión del proyecto (train.py, environment.py, model/*.xml, colab_test.ipynb, .gitignore). No se ha modificado ningún archivo del repo: todo lo de abajo es propuesta.

## Método y límites

- 3 revisores por dominio (entrada, MDP, modelo/repo), 24 candidatos.
- Cada hallazgo de severidad alta o crítica lo verificaron 2 revisores escépticos; el resto, 1. Votación: confirmado si ningún voto lo refuta.
- Cada corrección propuesta fue analizada por un agente aparte que preguntó: ¿puede romper checkpoints, cambiar la dinámica de entrenamiento o borrar datos?
- **Límite:** las librerías están en el entorno conda `robot_bpd` (ver verificación posterior al final). No se ejecutó el entrenamiento completo.

Resultado: 16 confirmados (train-1 y train-2 aparecían duplicados, y los fusionamos), 8 descartados, 0 disputados, 0 sin verificar.

## Resumen

| # | Sev. | Dónde | Problema | Decisión |
|---|------|-------|----------|----------|
| A | crítica | train.py:57-62 | Sin `--resume` se hace `shutil.rmtree(logs/<exp>)`: checkpoints, videos y logs se borran sin aviso. `logs/` y `*.pt` están en .gitignore, así que no hay copia. | Aplicar ya |
| B | alta | train.py:19-20, 42-62 | `--resume_path` sin `--resume` no evita el borrado y el checkpoint pedido puede quedar borrado | Aplicar ya |
| C | media | train.py:46-52 | `int()` sobre el nombre del checkpoint revienta con nombres no numéricos (`model_best.pt`) | Aplicar ya |
| D | media | train.py:27-35, 154 | `record_interval` es código muerto y el texto dice "video al 5%" cuando el video real es cada 4 episodios | Aplicar ya |
| E | baja | train.py:93 | `round(98304/num_envs)`: el comentario "batch constante" es falso; con num_envs ≥ 196608 da 0 pasos | Aplicar ya (solo comentario + validación) |
| F | baja | train.py:131 | `critic.hidden_dims` = 2030 (¿typo?) | **No cambiar el valor**; preguntar al autor |
| G | baja | environment.py:78 | MJCF con ruta relativa (`model/robot.xml`); falla si no se ejecuta desde la raíz | Opcional, probar en Colab |
| H | media (latente) | environment.py:287-292 | El actor recibe velocidad lineal sin ruido ni estimador | **No aplicar ahora** |
| I | baja | environment.py:46,55-56,355-364; robot.xml:16-52 | Configuración muerta (`mode`, `_curriculum_level`, `subterrain_parameters`) y clases de actuador sin uso en el XML | Aplicar solo la parte del XML |
| J | crítica | environment.py:60,120-127; commit 34d6ff8 | El commit "Fix Bug Rigid Options" quita `rigid_options` y el error `'Scene' has no attribute 'rigid_options'` sigue en el notebook | Fijar versiones primero; luego decidir |
| K | media | colab_test.ipynb (celdas 0, 1, 2) | Dependencias sin fijar; el notebook termina en error | Aplicar pines tras probar en Colab limpio |
| L | baja | model/robot.xml:279-281, 514-520 | Comentario de exclusión de contacto que nombra bodies inexistentes y un "closed-loop" que no existe | Aplicar (solo comentario) |
| M | baja | model/robot.xml:53-55, 93-97 | La geom `self_collision_only` (bits 2/2) no colisiona con ninguna otra geom | Aplicar (solo comentario) |
| N | baja | model/robot.xml:16-52 | 6 clases `<default>` sin ninguna referencia; el resto del XML no cambia | Aplicar ya |

---

## 1. Aplicar ya (cambios seguros, sin efecto sobre el entrenamiento)

### A. No borrar el experimento anterior (train.py:57-62)
- **Cambio:** sustituir `shutil.rmtree(log_dir)` por un renombrado a una carpeta hermana con marca de tiempo, `os.rename(log_dir, f"{log_dir}_bak_{timestamp}")`. Quitar `import shutil`.
- **Por qué no hace daño:** no toca redes, observaciones, acciones, recompensas ni hiperparámetros, así que los checkpoints existentes siguen siendo compatibles. El respaldo queda fuera de `logs/<exp>/**`, así que el glob de `--resume` no lo recoge.
- **Daño potencial y mitigación:** cada arranque sin `--resume` deja copias (unos 100 checkpoints por run). Hay que limpiar a mano. En Windows, `os.rename` falla si hay un archivo abierto (por ejemplo, un video que escribe el proceso anterior); conviene cerrar procesos antes.
- **Alternativa más estricta (no recomendada como primera opción):** exigir `--fresh` para borrar y abortar si `log_dir` tiene datos. Protege más, pero cambia el comportamiento por defecto.

### B. `--resume_path` sin `--resume` (train.py tras la línea 25)
- **Cambio:**
  ```python
  if args.resume_path is not None and not args.resume:
      parser.error("--resume_path requiere --resume")
  ```
- **Por qué no hace daño:** se ejecuta antes de cualquier borrado y antes de `gs.init`. Solo afecta a la combinación que hoy borra el checkpoint pedido.
- **Nota:** esto no arregla el borrado por defecto de A; son independientes.

### C. Selección del checkpoint con regex (train.py:46-52)
- **Cambio:** filtrar con `re.fullmatch(r"model_(\d+)\.pt", basename)` y tomar `max()` sobre `(iteración, ruta)`. Añadir `import re`.
- **Por qué no hace daño:** solo cambia cómo se elige el checkpoint con `--resume`. No escribe ni borra nada.
- **Pendiente:** confirmar el patrón de nombres que escribe rsl_rl (no instalado aquí).

### D. Eliminar `record_interval` (train.py:27-35, 154)
- **Cambio:** borrar las líneas de `record_interval` (son código muerto: solo se usa en el print) y quitar `video_every` del print.
- **Por qué no hace daño:** la frecuencia real de video la fija `episode_trigger` (train.py:78, `episode_id % 4 == 0`), y no depende de esta variable.
- **Nota:** el cálculo de `save_interval` (train.py:27-39) tiene un salto no monótono: con `max_iterations=199` guarda ~22 veces, y con 200 guarda 100. Es menor; anotarlo, no es urgente.

### E. Comentario y validación de `num_envs` (train.py:93)
- **Cambio:** corregir el comentario ("batch ~98304, exacto solo si num_envs divide 98304") y añadir `if args.num_envs < 1: parser.error(...)`.
- **Por qué no hace daño:** solo texto y validación de entrada que hoy ya falla.

### L. Comentarios de `robot.xml` (279-281, 514-520)
- **Cambio:** reescribir los comentarios con nombres reales: `jaw` y `bottom_head_shell` son geoms del body `jaw_soft`, no bodies distintos, y no hay closed-loop (`<equality/>` vacío).
- **Por qué no hace daño:** los comentarios XML no se interpretan.

### M. Comentario de `self_collision_only` (robot.xml:93)
- **Cambio:** solo comentario que diga que la geom no colisiona con otras geoms del robot.
- **No borrar** la geom ni la clase: no hay evidencia de que sea inocua en Genesis (ver pendiente P2).

### N. Clases `<default>` sin uso (robot.xml:16-52)
- **Cambio:** borrar `perfect_actuator`, `chosen_actuator_old`, `chosen_actuator_new`, `chosen_actuator_antoine`, `passive_wheel` y `passive_joint`. Verificado: no hay ningún `class=` ni `childclass=` que las use (el único `childclass` es `microduck`).
- **Verificación:** contar bodies, joints, actuadores y sensores antes y después; deben ser iguales (15 bodies, 14 joints). Los assets (pares .stl/.part) no se tocan.

### I (parte XML)
- Igual que N para las clases de actuador sin uso. Lo de `environment.py` va en la sección 3.

---

## 2. Aplicar después de verificar en un entorno real (Colab con GPU)

### J. `rigid_options` y versiones (crítica)
- **Estado actual:** el commit `34d6ff8` ("Fix Bug Rigid Options") quita el bloque `rigid_options` que añadió `cd993c7`. El notebook sigue fallando con `AttributeError: 'Scene' object has no attribute 'rigid_options'`. O sea, el commit no arregla el fallo.
- **Causa probable (no verificada):** versiones distintas de genesis-world / genesis-forge. El notebook no fija versiones, y los dos traces de error muestran líneas distintas de `managed_env.py`.
- **Orden:**
  1. Fijar versiones (ver K) y reproducir en un runtime de Colab limpio.
  2. Con esas versiones, comprobar con grep si `rigid_options` existe en `genesis/`. Si no existe, restaurarlo es un error.
  3. Si existe, restaurar el bloque de `cd993c7` y comparar recompensas.
- **Daño de restaurarlo a ciegas:** cambia la física sin aviso (solver Newton, joint limits, colisiones, `dt/2`, `max_collision_pairs=30`). Los checkpoints de antes no son comparables. Si la versión no tiene ese atributo, hacerlo rompe el arranque.

### K. Pines de versiones y ejecución del notebook

**Aplicado (versiones de robot_bpd):** celda de genesis-forge → `genesis-forge==0.3.0 genesis-world==0.3.14`; celda de rsl-rl → `rsl-rl-lib==5.2.0`. Pendiente: el clonado idempotente de la celda 2, y la celda 6 sigue en `%%python` (no tocar). Las salidas del notebook se conservan. Sin probar en Colab todavía.
- **Cambio propuesto:**
  - Celda 0: `genesis-forge==0.5.3 genesis-world==1.4.0` (genesis-forge solo exige `genesis-world>=1.0.0`).
  - Celda 1: `rsl-rl-lib==5.5.1`.
  - Celda 2: clonado idempotente, `test -d ... || git clone ...`.
- **NO cambiar la celda 6 de `%%python` a `!python`:** en Colab, `%%python` lanza `CalledProcessError` al salir con código distinto de cero. Con `!` el fallo quedaría oculto y la celda aparecería como correcta. Es la corrección que más daño haría.
- **Riesgo:** las versiones pinadas podrían no ser compatibles entre sí. Probar en runtime limpio antes de confirmar.

### G. Ruta absoluta del MJCF (environment.py:78) — opcional
- **Cambio:** `file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "robot.xml")`.
- **Por qué no hace daño:** con la raíz como directorio de trabajo apunta al mismo archivo. Falta verificar que `gs.morphs.MJCF` acepta rutas absolutas en Windows.
- **Alternativa sin tocar código:** documentar "ejecutar siempre desde la raíz", que es lo que hace el notebook.

---

## 3. No aplicar ahora (la corrección causa más daño que el error)

### F. `critic.hidden_dims` = 2030 (train.py:131)
- Cambiarlo a 2048 o 512 cambia la forma de la primera capa del crítico. Los checkpoints guardados con 2030 dejan de cargar con `--resume`. Además, el valor puede ser intencional: no se puede saber desde el código.
- **Hacer:** dejar el valor y preguntar al autor. Si confirma 2048, hacerlo en un commit aparte, avisando de que los checkpoints antiguos dejan de cargarse.

### H. Actor sin ruido ni estimador de velocidad (environment.py:287-292)
- Cambiar las observaciones del actor (quitar `linear_velocity` o añadir ruido) cambia las dimensiones de entrada: los checkpoints existentes no cargan. Además, sin velocidad lineal el actor pierde la señal para `lin_vel_x` en [0, 2].
- **Hacer:** no tocarlo. Dejar un comentario de una línea con la forma `shortcut: actor usa velocidad lineal sin ruido, añadir ruido o estimador antes de desplegar`. Cuando exista un despliegue real, hacerlo en rama nueva con un `-e` nuevo.

### I (parte Python): configuración muerta de `environment.py`
- **No borrar** `mode`, `_mode` ni `_curriculum_level`: `colab_test.ipynb` (línea ~509) tiene la misma firma y `ManagedEnvironment` podría leerlos internamente. No hay forma de comprobarlo sin genesis_forge instalado.
- **Hacer:** dejar el código y añadir un comentario que diga que son reservados.

---

## 4. Pendiente de verificar (riesgo no confirmado)

### P1. ¿Funciona realmente `--resume`? (prioridad alta)
- `train.py` nunca llama a `runner.load()`. La reanudación depende de que rsl_rl lea `resume` y `resume_path` de la configuración. Si no lo hace, `--resume` arranca desde cero **sin avisar**.
- **Prueba:** entrenar 2 iteraciones, reanudar, y comprobar que el contador de iteración continúa en vez de empezar en 0.

### P2. ¿Colisiona la geom `self_collision_only`?
- Depende de si Genesis respeta `contype/conaffinity` y de su valor por defecto de `enable_self_collision` tras `34d6ff8`. Medir pares de contacto en Genesis.

### P3. Versiones de Genesis (ver J)
- Confirmar con grep en el paquete instalado qué APIs existen.

---

## 5. Descartados (la revisión los refutó)

| Hallazgo | Motivo |
|---|---|
| train-3: resume y seed dependen de claves que rsl_rl puede ignorar | No se pudo confirmar sin rsl_rl. Ver P1 para la prueba real. |
| train-8: `env.cfg = {}` enmascara un atributo | El escenario depende de código externo no verificable. |
| env-1: el actor no recibe posiciones ni velocidades articulares | Diseño, no error. |
| env-3: orientación de la cámara del robot | Nadie consume `robot_camera`. |
| env-5: unidades de `limit_angle` | Depende de la firma de genesis_forge, no verificable aquí. |
| env-6: comando de velocidad solo hacia delante | Decisión de diseño. Anotar si se quiere marcha atrás. |
| env-7: `HEIGHT_OFFSET` y origen del MJCF | Depende de la semántica de origen de Genesis. |
| repo-1: `Reference/isaaclab-microduck` es un gitlink sin `.gitmodules` | Efecto no reproducible en el uso del proyecto. Reference/ está vacío en el worktree. |

---

## Verificación posterior (entorno conda `robot_bpd`)

Versiones instaladas: genesis-world 0.3.14, genesis-forge 0.3.0, rsl-rl-lib 5.2.0, mujoco 3.9.0, torch 2.11.0.
El notebook de Colab usa genesis-world 1.4.0, genesis-forge 0.5.3 y rsl-rl-lib 5.5.1 (según sus trazas). **Las versiones locales y las de Colab no coinciden**; el pin de K tiene que elegir una sola.

- **P1 confirmada (prioridad alta):** en rsl-rl-lib 5.2.0 `OnPolicyRunner` no lee `resume` ni `resume_path` de la configuración y no llama a `load()` por su cuenta. `--resume` no restaura el modelo: el entrenamiento arranca desde la iteración 0 sin avisar. **Aplicada y verificada:** `train.py` llama a `runner.load(resume_path)` y continúa desde `iter + 1`, con `remaining = max_iterations - start_it`. Prueba: entrenamiento de 1 iteración, reanudación con `--max_iterations 2` → arranca en `Learning iteration 1/2` y genera `model_1.pt`.
- **J aclarado:** `rigid_options` existe en genesis 0.3.14 (`genesis/engine/scene.py:149`) y genesis-forge lo usa (`genesis_forge/managers/actuator/actuator_manager.py:119`). El `AttributeError` es de versión: no aparece con la versión local. Restaurar `cd993c7` localmente cambiaría la física respecto a lo que corre hoy.
- **MuJoCo:** el XML original y el modificado dan los mismos conteos (16 bodies, 15 joints, 14 actuadores, 6 sensores, 81 geoms, nq 21, nv 20). Los cambios XML son neutros para la simulación.
- **Pruebas de `train.py` (aplicadas):** la selección de checkpoint elige `model_200.pt` entre `model_best.pt`, `model_1000_bak.pt` y `model_9.pt`; el respaldo renombra la carpeta sin perder archivos; `--resume_path` sin `--resume` y `-n 0` salen con código 2 antes de inicializar Genesis.

## Orden de ejecución recomendado

1. **K** (pines) y reproducir el fallo de **J** en Colab limpio.
2. **A, B, C** (seguridad de datos y resume).
3. **D, E, L, M, N, I-XML, G** (limpieza sin efecto en el entrenamiento).
4. **J** (restaurar `rigid_options` solo con versiones fijadas y comparando métricas).
5. **P1** (prueba de resume) antes de confiar en `--resume` de nuevo.
6. **F** y **H**: decisiones del autor, no cambios de código.
