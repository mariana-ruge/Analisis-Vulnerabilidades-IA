"""Gráficas del reporte de evaluación (matplotlib, estáticas)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
SERIE = {"isolation_forest": "#2a78d6", "random_forest": "#eb6834"}
NOMBRE = {"isolation_forest": "Isolation Forest (no supervisado)", "random_forest": "Random Forest (supervisado)"}


def _estilo(ax, titulo: str):
    ax.set_facecolor(SUPERFICIE)
    ax.set_title(titulo, loc="left", color=TINTA, fontsize=12, pad=12)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#d9d8d4")
    ax.tick_params(colors=TINTA_2, labelsize=9)
    ax.grid(axis="x", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)


def curvas_pr(resultados: dict[str, pd.DataFrame], ruta: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SUPERFICIE)
    for tipo, res in resultados.items():
        p, r, _ = precision_recall_curve(res["es_anomalo"], res["puntuacion"])
        ax.plot(r, p, color=SERIE[tipo], linewidth=2, label=NOMBRE[tipo])
    _estilo(ax, "Precisión vs. recall en el periodo de prueba")
    ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
    ax.set_xlabel("Recall", color=TINTA_2)
    ax.set_ylabel("Precisión", color=TINTA_2)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=TINTA)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def deteccion_por_categoria(metricas: dict[str, dict], ruta: Path):
    tipos = list(metricas)
    cats = sorted({c for m in metricas.values() for c in m["deteccion_por_categoria"]})
    y = np.arange(len(cats))
    alto = 0.38
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(cats) + 1.4), facecolor=SUPERFICIE)
    for i, tipo in enumerate(tipos):
        vals = [metricas[tipo]["deteccion_por_categoria"].get(c, {}).get("recall", 0) for c in cats]
        pos = y + (i - (len(tipos) - 1) / 2) * (alto + 0.03)
        ax.barh(pos, vals, height=alto, color=SERIE[tipo], label=NOMBRE[tipo])
    ax.set_yticks(y, [c.replace("_", " ") for c in cats])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _estilo(ax, "Recall por categoría de ataque (prueba)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.08), ncol=2,
              fontsize=9, labelcolor=TINTA)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
