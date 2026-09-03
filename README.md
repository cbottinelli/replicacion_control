# Replicación de AI Control: Improving Safety Despite Intentional Subversion

Replicación parcial del paper [*AI Control: Improving Safety Despite Intentional Subversion*](https://arxiv.org/abs/2312.06942) (Greenblatt et al., 2024), sobre los protocolos **Trusted Monitoring** y **Defer to Trusted** . El objetivo del experimento fue replicar los resultados del paper utilizando modelos equivalentes de OpenAI disponibles publicamente en 2026.

El informe completo de la replicación está en [`Carlos Bottinelli - Informe.pdf`](./Carlos%20Bottinelli%20-%20Informe.pdf).

## Observaciones

- **Escala:** la replicación usa una muestra de 120 problemas (10% del tamaño del dataset original).
- **Modelos:** el stack final es U = `gpt-4.1-mini`, T = `gpt-3.5-turbo`, con los suspicion scores (logprobs) obtenidos vía `gpt-3.5-turbo-instruct`.
- **Código:** el pipeline de generación de datos reutiliza y/o adapta código del repositorio original. El código que implementa los protocolos y analiza el output de los experimentos es propio, desarrollado con asistencia de herramientas de IA para tareas de escritura y debugging.

## Estructura del repositorio

```
├── 0 - Dev - Pipeline Generacion Datos.ipynb   # Desarrollo y prueba del pipeline de generación de datos y backdoors
├── 1- Generacion_datos.ipynb                   # Corrida final del pipeline sobre la muestra completa de problemas
├── 2 - Experimentos.ipynb                      # Experimentos de Trusted Monitoring y Defer to Trusted, y comparación con el paper
├── Carlos Bottinelli - Informe.pdf             # Informe completo de la replicación
├── requirements.txt
├── data/                                       # Datos intermedios cacheados (generados al correr los notebooks)
└── src/
    ├── monitoring/                             # Pipeline de generación de datos, scoring, y protocolos de control
    └── elk/                                    # Modulos reutilizados del repositorio original 
```

**Orden de lectura sugerido:** `0 - Dev...` (si se quiere una visión detallada del pipeline de generación de datos y algunas decisiones de desarrollo asociadas) → `1- Generacion_datos.ipynb` (generación de datos sobre la muestra completa) → `2 - Experimentos.ipynb` (protocolos y resultados) → informe en PDF (análisis y discusión).

## Instalación

Requiere Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Configurar la API key de OpenAI en un archivo `.env` en la raíz del proyecto:

```
OPENAI_API_KEY=tu-api-key
```

## Cómo correr el pipeline

1. **Obtener el dataset de problemas.** `data/problems.jsonl` no está versionado (ver más abajo). Generarlo corriendo `src/monitoring/get_problems.py`, que descarga y filtra el dataset [APPS](https://huggingface.co/datasets/codeparrot/apps).
2. **Generar los datos** (soluciones normales, backdoors, suspicion scores) corriendo `1- Generacion_datos.ipynb`. Este notebook cachea todos los resultados intermedios en `data/` vía `JsonDiskCache`, por lo que se puede interrumpir y retomar sin recomputar desde cero.
3. **Correr los experimentos** de Trusted Monitoring y Defer to Trusted en `2 - Experimentos.ipynb`, que consume los datos generados en el paso anterior.

## Nota sobre `data/`

`problems.jsonl` no está incluido en el repositorio (pesa ~700 MB y proviene íntegramente de un dataset público de terceros). El resto de los archivos en `data/` son outputs intermedios del pipeline (soluciones generadas, scores, resultados de corrección) — se regeneran corriendo los notebooks en el orden indicado arriba.
