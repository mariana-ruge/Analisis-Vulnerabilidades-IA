# Detección de tráfico anómalo a partir de vulnerabilidades de IA

Pipeline que toma los snapshots mensuales de vulnerabilidades de IA
(`vulnerabilidades-ia-2026-*.csv|json`, fuente: laautopsia.com, CC-BY-SA-4.0),
simula el tráfico HTTP de una plataforma de IA que sufre esos ataques y entrena
modelos para detectar tráfico anormal.

```
CSV/JSON ──► 1. ingesta ──► 2. simulador ──► 3. características ──► 4. modelo ──► reporte + modelos .joblib
              limpieza        tráfico normal     payload, IP, UA,      Isolation Forest
              CVSS, CWE       + 1 campaña por    ventanas 5 min        Random Forest
              categoría       vulnerabilidad
```

## Uso

```bash
pip install -r requirements.txt
python -m pipeline                     # ejecuta todo (≈25 s)
python -m pipeline --dias 14 --semilla 7 --peticiones-hora 1500
python -m pipeline.detectar logs.csv --modelo isolation_forest   # puntuar logs nuevos
python -m pytest tests                 # pruebas de humo
```

Todo lo generado va a `salidas/` (ignorado por git, se regenera con la semilla):

| Fichero | Contenido |
|---|---|
| `vulnerabilidades_limpias.csv` | 74 vulnerabilidades únicas con CVSS completo y `categoria_ataque` |
| `trafico_simulado.csv` | Logs etiquetados (`es_anomalo`, `avidId`, `fase_ataque`) |
| `modelos/*.joblib` | Pipeline completo (extractor + preprocesado + modelo) y umbral |
| `predicciones_*.csv` | Puntuación y predicción por evento del periodo de prueba |
| `reporte_metricas.json` | Métricas globales, por categoría y por campaña |
| `graficas/*.png` | Curvas precisión-recall y recall por categoría |

## Etapas

1. **Ingesta** ([pipeline/ingesta.py](pipeline/ingesta.py)): une CSV y JSON (114 filas → 74
   vulnerabilidades, hay solapes entre formatos y meses), normaliza `N/A`, completa las
   métricas CVSS desde `cvssVector`, imputa el score por severidad y asigna una categoría de
   ataque a partir del CWE (con respaldo por palabras clave; la inyección de prompt prima).
2. **Simulación** ([pipeline/simulador.py](pipeline/simulador.py)): 7 días de tráfico de una
   plataforma ficticia (chat, embeddings, RAG, MCP, carga de modelos…).
   - *Normal*: patrón diurno, menos actividad en fin de semana, usuarios con actividad Pareto,
     integraciones automáticas con ráfagas (algunas mal configuradas, con muchos errores) y un
     1 % de peticiones benignas que contienen tokens sospechosos.
   - *Ataque*: una campaña por vulnerabilidad. La categoría define payloads y endpoints; el CVSS
     define el comportamiento: `AC:High` → lento y con UA de navegador, `UI:Required` → payload
     oculto dentro de contenido legítimo, `PR:None` → sin token, `AV:Local` → IP interna, score →
     volumen. El 30 % de campañas ofusca payloads (base64, URL-encoding, mayúsculas), el 30 % rota
     IP y el 25 % usa un token robado.
3. **Características** ([pipeline/caracteristicas.py](pipeline/caracteristicas.py)): 28 features
   numéricas (tamaño/entropía del payload, 8 familias de indicadores léxicos, estado, bytes,
   latencia, hora cíclica, comportamiento por IP en ventanas de 5 min) + método, endpoint
   normalizado y familia de user-agent. Las etiquetas nunca entran como features.
4. **Modelo** ([pipeline/modelo.py](pipeline/modelo.py)): partición **temporal** 70/30, así las
   campañas de prueba son vulnerabilidades no vistas en entrenamiento.
   - `isolation_forest`: no supervisado, entrenado sin etiquetas; umbral = percentil
     `1 - contaminacion_esperada` (5 %) de las puntuaciones de entrenamiento. Es el escenario real.
   - `random_forest`: supervisado con las etiquetas simuladas; cota superior e importancia de features.

## Resultados (semilla 42)

66 405 eventos, 5,3 % anómalos; prueba = 19 922 eventos con 15 campañas.

| Modelo | Precisión | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---|---|---|---|---|---|
| Isolation Forest | 0,81 | 0,71 | 0,76 | 0,99 | 0,86 | 0,69 % |
| Random Forest | 1,00 | 0,99 | 1,00 | 1,00 | 1,00 | 0,00 % |

Lectura:
- El Isolation Forest detecta bien inyección de código/prompt, path traversal y fuga de datos
  (71–83 % de recall), pero casi no ve la **evasión de autorización** (9 %): un IDOR es una
  petición válida a un endpoint normal; hace falta contexto de sesión/negocio.
- Sus falsos positivos son casi todos integraciones automáticas legítimas mal configuradas
  (scripts con ráfagas de 401/429/404). Conviene listarlas como conocidas o alertar sobre
  ellas con menor prioridad.
- El Random Forest casi perfecto indica que los datos simulados son separables con etiquetas;
  no debe leerse como rendimiento esperado en producción.

## Limitaciones

- El tráfico es sintético: los payloads salen de plantillas por categoría, no de exploits reales.
- Con el corte temporal, la prueba solo cubre las categorías cuyas campañas caen en los últimos
  días (6 de 10 con la semilla 42); cambia `--semilla` para otras combinaciones.
- Las features de comportamiento se calculan dentro de cada lote; para producción habría que
  mantener estado por IP entre lotes.
