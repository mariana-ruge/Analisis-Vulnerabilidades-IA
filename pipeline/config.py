"""Configuración central del pipeline de detección de tráfico anómalo."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Datos de entrada (snapshots mensuales de vulnerabilidades de IA)
    dir_datos: Path = RAIZ
    patron_fuentes: str = "vulnerabilidades-ia-*"

    # Salidas
    dir_salidas: Path = RAIZ / "salidas"

    # Simulación de tráfico
    semilla: int = 42
    inicio_simulacion: str = "2026-08-01 00:00:00"
    dias_simulacion: int = 7
    peticiones_normales_por_hora: int = 900  # media en hora punta
    num_usuarios: int = 180
    tasa_falsos_positivos_dificiles: float = 0.01  # tráfico benigno "sospechoso"

    # Modelado
    fraccion_entrenamiento: float = 0.7  # corte temporal
    contaminacion_esperada: float = 0.05  # umbral del modelo no supervisado
    n_arboles: int = 300

    @property
    def dir_modelos(self) -> Path:
        return self.dir_salidas / "modelos"

    @property
    def dir_graficas(self) -> Path:
        return self.dir_salidas / "graficas"

    def crear_directorios(self) -> None:
        for d in (self.dir_salidas, self.dir_modelos, self.dir_graficas):
            d.mkdir(parents=True, exist_ok=True)
