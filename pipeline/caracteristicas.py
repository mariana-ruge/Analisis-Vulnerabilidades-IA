"""Etapa 3 – Ingeniería de características sobre los logs de tráfico.

Tres familias de features:

1. Petición: tamaño, entropía y composición del payload, indicadores léxicos
   agrupados por familia de ataque, autenticación, códigos de estado, bytes,
   latencia y hora del día.
2. Comportamiento por IP (ventanas de 5 minutos): volumen, endpoints
   distintos, tasa de errores, payload medio y tiempo entre peticiones.
3. Categóricas: método, endpoint normalizado y familia del user-agent.

El extractor es un transformer de scikit-learn, de modo que el modelo
guardado incluye el preprocesado completo y se puede aplicar a logs crudos.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

INDICADORES = {
    "ind_traversal": r"\.\./|\.\.\\|%2e%2e|%252e|/etc/(passwd|shadow)|/proc/self|win\.ini|\.ssh/|\.aws/",
    "ind_ssrf": r"169\.254\.169\.254|127\.0\.0\.1|localhost|\[::1\]|0x7f|metadata\.(google\.)?internal|gopher://|file://|@[a-z0-9.-]+\.[a-z]+/",
    "ind_codigo": r"__import__|eval\(|exec|compile\(|__globals__|\{\{.*\}\}|\$\{|child_process|new function|_target_|gasv",
    "ind_comandos": r";\s*(curl|wget|nc|sh|rm|ls|cat)\b|\$\(|`[^`]+`|&&|\|\s*sh|--server=|--upload-pack|insecure-skip",
    "ind_sql": r"'\s*or\s*'?1'?\s*=\s*'?1|union\s+select|drop\s+table|--\s*\"?$|\$where|\$ne|detach delete|select \* from",
    "ind_prompt": r"ignora (todas )?las instrucciones|ignore previous|system prompt|modo desarrollador|instrucci[oó]n oculta|system:",
    "ind_exfil": r"atacante\.example|https?://[^\"\s]*\?(d|to|data)=|credentials|api_keys|\.env\b|secrets",
    "ind_privilegios": r"\"role\":\s*\"admin\"|default_admin|x-forwarded-for|trust_remote_code\":\s*true|verify_hash\":\s*false",
}
_INDICADORES_RE = {k: re.compile(v, re.IGNORECASE) for k, v in INDICADORES.items()}

NUMERICAS = [
    "long_payload", "entropia_payload", "ratio_no_alfanum", "ratio_mayus", "num_caracteres_raros",
    *INDICADORES, "total_indicadores", "tiene_token", "es_error_cliente", "es_error_servidor",
    "log_bytes_respuesta", "log_latencia_ms", "hora_sin", "hora_cos", "profundidad_endpoint",
    "ip_peticiones_5min", "ip_endpoints_distintos_5min", "ip_ratio_errores_5min",
    "ip_payload_medio_5min", "ip_log_delta_t", "ip_ratio_sin_token_5min",
]
CATEGORICAS = ["metodo", "endpoint_norm", "familia_ua"]


def _entropia(texto: str) -> float:
    if not texto:
        return 0.0
    n = len(texto)
    return -sum(c / n * math.log2(c / n) for c in Counter(texto).values())


def _familia_ua(ua: str) -> str:
    ua = (ua or "").lower()
    if any(s in ua for s in ("sqlmap", "nuclei", "nikto", "masscan", "zgrab")):
        return "escaner"
    if any(s in ua for s in ("curl", "wget", "python-requests", "go-http", "aiohttp")):
        return "script"
    if any(s in ua for s in ("openai", "langchain", "llama-index", "anthropic")):
        return "sdk"
    if "mozilla" in ua:
        return "navegador"
    return "otro"


def _normalizar_endpoint(ep: str) -> str:
    ep = re.sub(r"/\d+", "/{id}", (ep or "").split("?")[0])
    conocidos = ep.startswith(("/api/", "/mcp/", "/health"))
    return ep if conocidos else "otro"


class ExtractorCaracteristicas(BaseEstimator, TransformerMixin):
    """Convierte logs crudos (DataFrame) en la tabla de features."""

    ventana: str = "5min"

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["payload"] = df["payload"].fillna("").astype(str)
        orden_original = df.index
        df = df.sort_values("timestamp", kind="stable")

        p = df["payload"]
        f = pd.DataFrame(index=df.index)
        f["long_payload"] = np.log1p(p.str.len())
        f["entropia_payload"] = p.map(_entropia)
        longitud = p.str.len().replace(0, 1)
        f["ratio_no_alfanum"] = p.str.count(r"[^\w\s]") / longitud
        f["ratio_mayus"] = p.str.count(r"[A-Z]") / longitud
        f["num_caracteres_raros"] = p.str.count(r"[;|`$<>\\{}%]")
        for nombre, patron in _INDICADORES_RE.items():
            f[nombre] = p.map(lambda s, r=patron: len(r.findall(s)))
        f["total_indicadores"] = f[list(INDICADORES)].sum(axis=1)

        status = pd.to_numeric(df["status"], errors="coerce").fillna(0)
        f["tiene_token"] = df["tiene_token"].astype(str).str.lower().isin(["true", "1"]).astype(int)
        f["es_error_cliente"] = status.between(400, 499).astype(int)
        f["es_error_servidor"] = (status >= 500).astype(int)
        f["log_bytes_respuesta"] = np.log1p(pd.to_numeric(df["bytes_respuesta"], errors="coerce").fillna(0))
        f["log_latencia_ms"] = np.log1p(pd.to_numeric(df["latencia_ms"], errors="coerce").fillna(0))
        hora = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
        f["hora_sin"] = np.sin(2 * np.pi * hora / 24)
        f["hora_cos"] = np.cos(2 * np.pi * hora / 24)
        f["profundidad_endpoint"] = df["endpoint"].fillna("").str.count("/")

        # Comportamiento por IP en ventanas fijas
        f["metodo"] = df["metodo"].fillna("NA").astype(str)
        f["endpoint_norm"] = df["endpoint"].map(_normalizar_endpoint)
        f["familia_ua"] = df["user_agent"].map(_familia_ua)
        grupo_claves = [df["ip_origen"], df["timestamp"].dt.floor(self.ventana)]
        aux = pd.DataFrame({
            "ep": df["endpoint"].fillna(""),
            "err": (status >= 400).astype(int),
            "lp": p.str.len(),
            "st": 1 - f["tiene_token"],
        })
        g = aux.groupby(grupo_claves)
        f["ip_peticiones_5min"] = np.log1p(g["ep"].transform("size"))
        f["ip_endpoints_distintos_5min"] = g["ep"].transform("nunique")
        f["ip_ratio_errores_5min"] = g["err"].transform("mean")
        f["ip_payload_medio_5min"] = np.log1p(g["lp"].transform("mean"))
        f["ip_ratio_sin_token_5min"] = g["st"].transform("mean")
        delta = df.groupby("ip_origen")["timestamp"].diff().dt.total_seconds()
        f["ip_log_delta_t"] = np.log1p(delta.fillna(3600).clip(upper=3600))

        return f.loc[orden_original, NUMERICAS + CATEGORICAS]


def preprocesador() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", RobustScaler(), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), CATEGORICAS),
    ])
