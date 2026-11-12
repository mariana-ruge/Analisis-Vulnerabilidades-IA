"""Etapa 4 – Entrenamiento y evaluación de los modelos de detección.

* `isolation_forest` (no supervisado): aprende el tráfico de la ventana de
  entrenamiento sin usar etiquetas; el umbral se fija con la contaminación
  esperada. Es el escenario realista: no hay ataques etiquetados.
* `random_forest` (supervisado): usa las etiquetas de la simulación. Sirve
  como cota superior y para ver qué features discriminan más.

El corte entre entrenamiento y prueba es temporal, así que parte de las
campañas de prueba explotan vulnerabilidades que el modelo nunca vio.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .caracteristicas import CATEGORICAS, NUMERICAS, ExtractorCaracteristicas, preprocesador


def particion_temporal(trafico: pd.DataFrame, fraccion: float):
    corte = trafico["timestamp"].quantile(fraccion)
    return trafico[trafico["timestamp"] < corte], trafico[trafico["timestamp"] >= corte], corte


def construir(tipo: str, cfg) -> Pipeline:
    if tipo == "isolation_forest":
        modelo = IsolationForest(n_estimators=cfg.n_arboles, contamination="auto",
                                 random_state=cfg.semilla, n_jobs=-1)
    elif tipo == "random_forest":
        modelo = RandomForestClassifier(n_estimators=cfg.n_arboles, class_weight="balanced",
                                        min_samples_leaf=2, random_state=cfg.semilla, n_jobs=-1)
    else:
        raise ValueError(tipo)
    return Pipeline([("extractor", ExtractorCaracteristicas()), ("prep", preprocesador()), ("modelo", modelo)])


def puntuar(artefacto: dict, logs: pd.DataFrame) -> np.ndarray:
    """Devuelve una puntuación de anomalía (mayor = más anómalo)."""
    pipe = artefacto["pipeline"]
    if artefacto["tipo"] == "isolation_forest":
        return -pipe.decision_function(logs)
    return pipe.predict_proba(logs)[:, 1]


def entrenar(tipo: str, train: pd.DataFrame, cfg) -> dict:
    pipe = construir(tipo, cfg)
    if tipo == "isolation_forest":
        pipe.fit(train)  # sin etiquetas
        artefacto = {"tipo": tipo, "pipeline": pipe}
        puntuaciones = puntuar(artefacto, train)
        artefacto["umbral"] = float(np.quantile(puntuaciones, 1 - cfg.contaminacion_esperada))
    else:
        pipe.fit(train, train["es_anomalo"])
        artefacto = {"tipo": tipo, "pipeline": pipe, "umbral": 0.5}
    return artefacto


def evaluar(artefacto: dict, test: pd.DataFrame, vistas_en_train: set[str]) -> tuple[dict, pd.DataFrame]:
    y = test["es_anomalo"].to_numpy()
    s = puntuar(artefacto, test)
    pred = (s >= artefacto["umbral"]).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    res = test[["id_evento", "timestamp", "ip_origen", "endpoint", "categoria_ataque",
                "avidId", "fase_ataque", "es_anomalo"]].copy()
    res["puntuacion"] = s
    res["prediccion"] = pred

    ataques = res[res["es_anomalo"] == 1]
    por_categoria = ataques.groupby("categoria_ataque")["prediccion"].agg(["mean", "size"])
    por_vuln = ataques.groupby("avidId")["prediccion"].mean()
    nuevas = [v for v in por_vuln.index if v not in vistas_en_train]

    metricas = {
        "modelo": artefacto["tipo"],
        "umbral": artefacto["umbral"],
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
        "pr_auc": round(float(average_precision_score(y, s)), 4),
        "matriz_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "tasa_falsos_positivos": round(float(fp / max(fp + tn, 1)), 4),
        "deteccion_por_categoria": {
            k: {"recall": round(float(v["mean"]), 3), "eventos": int(v["size"])}
            for k, v in por_categoria.iterrows()
        },
        "campanas_en_test": int(len(por_vuln)),
        "campanas_detectadas_50pct": int((por_vuln >= 0.5).sum()),
        "campanas_no_vistas_en_train": len(nuevas),
        "recall_medio_campanas_no_vistas": round(float(por_vuln[nuevas].mean()), 3) if nuevas else None,
    }
    return metricas, res


def importancias(artefacto: dict, top: int = 15) -> dict:
    if artefacto["tipo"] != "random_forest":
        return {}
    pipe = artefacto["pipeline"]
    nombres = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["modelo"].feature_importances_
    orden = np.argsort(imp)[::-1][:top]
    return {str(nombres[i]).split("__", 1)[-1]: round(float(imp[i]), 4) for i in orden}


def guardar(artefacto: dict, ruta: Path) -> None:
    artefacto = {**artefacto, "features": {"numericas": NUMERICAS, "categoricas": CATEGORICAS}}
    joblib.dump(artefacto, ruta)


def cargar(ruta: Path) -> dict:
    return joblib.load(ruta)


def guardar_json(obj, ruta: Path) -> None:
    ruta.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
