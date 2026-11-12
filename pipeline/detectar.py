"""Inferencia sobre logs nuevos.

    python -m pipeline.detectar logs.csv [--modelo isolation_forest|random_forest] [--salida alertas.csv]

El CSV debe tener: timestamp, ip_origen, user_agent, metodo, endpoint,
tiene_token, payload, status, bytes_respuesta, latencia_ms.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import modelo
from .config import Config

COLUMNAS_REQUERIDAS = [
    "timestamp", "ip_origen", "user_agent", "metodo", "endpoint",
    "tiene_token", "payload", "status", "bytes_respuesta", "latencia_ms",
]


def detectar(logs: pd.DataFrame, artefacto: dict) -> pd.DataFrame:
    faltan = set(COLUMNAS_REQUERIDAS) - set(logs.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en los logs: {sorted(faltan)}")
    out = logs.copy()
    out["puntuacion_anomalia"] = modelo.puntuar(artefacto, logs)
    out["alerta"] = (out["puntuacion_anomalia"] >= artefacto["umbral"]).astype(int)
    return out


def main(argv=None):
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", type=Path)
    ap.add_argument("--modelo", default="isolation_forest", choices=["isolation_forest", "random_forest"])
    ap.add_argument("--salida", type=Path, default=cfg.dir_salidas / "alertas.csv")
    args = ap.parse_args(argv)

    artefacto = modelo.cargar(cfg.dir_modelos / f"{args.modelo}.joblib")
    res = detectar(pd.read_csv(args.logs), artefacto)
    alertas = res[res["alerta"] == 1].sort_values("puntuacion_anomalia", ascending=False)
    alertas.to_csv(args.salida, index=False)
    print(f"{len(alertas)} alertas de {len(res)} eventos -> {args.salida}")
    if len(alertas):
        print(alertas.groupby("ip_origen").size().sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
