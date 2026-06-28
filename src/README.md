## Obtencion de Problemas (Dataset --> [ProblemSolutionOut])

get_problems.py --> lee el dataset y genera data/problems.json
dataset_wrapping.py --> genera global_constant_probs: list[LoadedProblem]
problem_setup.py --> procesa el dataset cargado y genera las soluciones normales (ProblemSolutionOut) que consume el pipeline.
client.py --> (modulo propio) cliente OpenAI y variables globales 

---

### `get_normal_solutions(problems, models) -> tuple[list[Optional[ProblemSolutionOut]], list[CommentNormalRet]]`
Genera una solución normal para cada problema usando el modelo configurado y la convierte a `ProblemSolutionOut`. También retorna los outputs intermedios del modelo.

### `get_problems_initial(desired_problems, get_safety_factor, timeout_per) -> list[ProblemIdx]`
Selecciona un subconjunto de problemas del dataset, descartando problemas no válidos y aquellos cuyas soluciones de referencia no pasan la evaluación de correctitud.

### `get_problems_normal_solutions_filt(desired_problems) -> tuple[list[ProblemIdx], list[ProblemSolutionOut], list[CommentNormalRet]]`
Obtiene un conjunto de problemas filtrados, genera sus soluciones normales y descarta las generaciones fallidas.

---

## Pipeline de Generación de Backdoors (ProblemSolutionOut --> )

ORIGINAL: backdoor_funcs.py
PROPIO: backdoor_generation_pipeline.py