"""Etapa 1 – Ingesta y limpieza de los snapshots de vulnerabilidades.

Lee los ficheros CSV y JSON mensuales, unifica el esquema, completa los campos
CVSS a partir del vector, elimina duplicados entre snapshots y asigna a cada
vulnerabilidad una categoría de ataque que luego usa el simulador de tráfico.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNAS_CVSS = {
    "AV": ("attackVector", {"N": "Network", "A": "Adjacent", "L": "Local", "P": "Physical"}),
    "AC": ("attackComplexity", {"L": "Low", "H": "High"}),
    "PR": ("privilegesRequired", {"N": "None", "L": "Low", "H": "High"}),
    "UI": ("userInteraction", {"N": "None", "R": "Required"}),
    "S": ("scope", {"U": "Unchanged", "C": "Changed"}),
    "C": ("confidentialityImpact", {"N": "None", "L": "Low", "H": "High"}),
    "I": ("integrityImpact", {"N": "None", "L": "Low", "H": "High"}),
    "A": ("availabilityImpact", {"N": "None", "L": "Low", "H": "High"}),
}

# CWE -> familia de ataque observable en tráfico
CWE_A_CATEGORIA = {
    "path_traversal": ["CWE-22", "CWE-59", "CWE-73"],
    "ssrf": ["CWE-918"],
    "inyeccion_codigo": ["CWE-94", "CWE-95", "CWE-1336", "CWE-502"],
    "inyeccion_comandos": ["CWE-78", "CWE-77", "CWE-88"],
    "inyeccion_sql_nosql": ["CWE-89", "CWE-943"],
    "evasion_autorizacion": [
        "CWE-284", "CWE-285", "CWE-287", "CWE-290", "CWE-420",
        "CWE-639", "CWE-668", "CWE-862", "CWE-863",
    ],
    "fuga_informacion": ["CWE-200", "CWE-532"],
    "denegacion_servicio": ["CWE-1333", "CWE-400", "CWE-770"],
    "cadena_suministro": ["CWE-506", "CWE-354", "CWE-345"],
}
MAPA_CWE = {cwe: cat for cat, cwes in CWE_A_CATEGORIA.items() for cwe in cwes}

PALABRAS_CLAVE = [  # respaldo por texto cuando el CWE no es concluyente
    (r"prompt|instrucci[oó]n inyectada|llm", "inyeccion_prompt"),
    (r"path traversal|lectura arbitraria|escritura arbitraria", "path_traversal"),
    (r"ssrf", "ssrf"),
    (r"sql|nosql|cypher", "inyeccion_sql_nosql"),
    (r"rce|ejecuci[oó]n.*c[oó]digo|sandbox|plantilla|ssti", "inyeccion_codigo"),
    (r"comando|flags?|argumento", "inyeccion_comandos"),
    (r"autenticaci[oó]n|autorizaci[oó]n|idor|acceso", "evasion_autorizacion"),
    (r"redos|dos\b|denegaci[oó]n", "denegacion_servicio"),
    (r"pypi|malicioso|integridad|dependencia", "cadena_suministro"),
    (r"credencial|expuest|fuga|exfiltra", "fuga_informacion"),
]

SEVERIDAD_POR_SCORE = [(9.0, "CRITICAL"), (7.0, "HIGH"), (4.0, "MEDIUM"), (0.1, "LOW")]


def _snapshot_desde_nombre(ruta: Path) -> str:
    m = re.search(r"(\d{4}-\d{2})", ruta.name)
    return m.group(1) if m else "desconocido"


def cargar_fuentes(dir_datos: Path, patron: str = "vulnerabilidades-ia-*") -> pd.DataFrame:
    """Carga todos los CSV/JSON que cumplan el patrón en un único DataFrame."""
    marcos = []
    for ruta in sorted(Path(dir_datos).glob(patron)):
        if ruta.suffix == ".csv":
            df = pd.read_csv(ruta, encoding="utf-8-sig", dtype=str)
        elif ruta.suffix == ".json":
            with open(ruta, encoding="utf-8") as fh:
                contenido = json.load(fh)
            # Mantener los null de JSON como NaN (no como el texto "None",
            # que en CVSS es un valor legítimo: PR:N, UI:N...).
            df = pd.DataFrame(contenido["data"]["rows"]).map(
                lambda v: np.nan if v is None else str(v)
            )
        else:
            continue
        df["snapshot"] = _snapshot_desde_nombre(ruta)
        df["fichero_origen"] = ruta.name
        marcos.append(df)
    if not marcos:
        raise FileNotFoundError(f"No hay fuentes '{patron}' en {dir_datos}")
    return pd.concat(marcos, ignore_index=True)


def _parsear_vector(vector: str) -> dict[str, str]:
    if not isinstance(vector, str) or not vector.startswith("CVSS:"):
        return {}
    partes = dict(p.split(":", 1) for p in vector.split("/")[1:] if ":" in p)
    return {
        col: valores.get(partes[k])
        for k, (col, valores) in COLUMNAS_CVSS.items()
        if k in partes
    }


def _categoria(fila: pd.Series) -> str:
    texto = f"{fila.get('titleEs', '')} {fila.get('descriptionEs', '')}".lower()
    # La inyección de prompt es transversal: si el texto lo menciona, prima.
    if re.search(r"inyecci[oó]n de prompt|prompt injection|via prompt|instrucci[oó]n inyectada", texto):
        return "inyeccion_prompt"
    if fila["cweId"] in MAPA_CWE:
        return MAPA_CWE[fila["cweId"]]
    for patron, cat in PALABRAS_CLAVE:
        if re.search(patron, texto):
            return cat
    return "evasion_validacion"


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    nulos = {"N/A": np.nan, "nan": np.nan, "None": np.nan, "": np.nan, "null": np.nan}
    # "None" es un valor CVSS legítimo (PR:N, UI:N...), así que solo se limpian
    # los campos de texto libre con el mapa completo.
    for col in df.columns:
        if col in {c for c, _ in COLUMNAS_CVSS.values()}:
            df[col] = df[col].replace({"N/A": np.nan, "nan": np.nan, "": np.nan})
        else:
            df[col] = df[col].replace(nulos)

    # Completar métricas CVSS desde el vector
    parseado = df["cvssVector"].map(_parsear_vector)
    for _, (col, _) in COLUMNAS_CVSS.items():
        desde_vector = parseado.map(lambda d, c=col: d.get(c))
        df[col] = df[col].fillna(desde_vector)

    df["cvssScore"] = pd.to_numeric(df["cvssScore"], errors="coerce")
    df["cvssVersion"] = df["cvssVersion"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df["reportedDate"] = pd.to_datetime(df["reportedDate"], errors="coerce")
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)

    # Severidad coherente con el score cuando falta
    def _sev(fila):
        if isinstance(fila["severity"], str):
            return fila["severity"].upper()
        if pd.notna(fila["cvssScore"]):
            return next(s for umbral, s in SEVERIDAD_POR_SCORE if fila["cvssScore"] >= umbral)
        return "UNKNOWN"

    df["severity"] = df.apply(_sev, axis=1)
    # Imputación del score con la mediana de su severidad (o global)
    mediana_sev = df.groupby("severity")["cvssScore"].median()
    df["cvss_imputado"] = df["cvssScore"].isna()
    df["cvssScore"] = df["cvssScore"].fillna(df["severity"].map(mediana_sev)).fillna(df["cvssScore"].median())

    # Valores por defecto conservadores para el simulador
    defaults = {
        "attackVector": "Network", "attackComplexity": "Low", "privilegesRequired": "None",
        "userInteraction": "None", "scope": "Unchanged", "confidentialityImpact": "High",
        "integrityImpact": "High", "availabilityImpact": "None",
    }
    df = df.fillna(value=defaults)
    df["cweId"] = df["cweId"].fillna("CWE-UNKNOWN")

    # Deduplicar: la misma vulnerabilidad aparece en CSV y JSON del mismo mes
    # y puede repetirse entre meses. Nos quedamos con la fila más completa y reciente.
    df["_completitud"] = df.notna().sum(axis=1)
    df = (
        df.sort_values(["avidId", "snapshot", "_completitud"])
        .drop_duplicates("avidId", keep="last")
        .drop(columns="_completitud")
        .reset_index(drop=True)
    )
    df["categoria_ataque"] = df.apply(_categoria, axis=1)
    return df


def ejecutar(dir_datos: Path, patron: str, destino: Path | None = None) -> pd.DataFrame:
    crudo = cargar_fuentes(dir_datos, patron)
    limpio = limpiar(crudo)
    if destino is not None:
        columnas = [
            "avidId", "cveId", "titleEs", "severity", "cvssScore", "cvss_imputado",
            *[c for c, _ in COLUMNAS_CVSS.values()], "cweId", "cweDescription",
            "categoria_ataque", "developer", "products", "company", "country",
            "reportedDate", "snapshot",
        ]
        limpio[columnas].to_csv(destino, index=False, encoding="utf-8")
    print(f"[ingesta] {len(crudo)} filas leídas -> {len(limpio)} vulnerabilidades únicas")
    return limpio
