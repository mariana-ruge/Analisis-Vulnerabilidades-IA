"""Pruebas de humo del pipeline (simulación corta para que sea rápida)."""
import pandas as pd
import pytest

from pipeline import ingesta, modelo, simulador
from pipeline.caracteristicas import CATEGORICAS, NUMERICAS, ExtractorCaracteristicas
from pipeline.config import Config
from pipeline.detectar import detectar


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    c = Config(dias_simulacion=2, peticiones_normales_por_hora=200, n_arboles=50)
    c.dir_salidas = tmp_path_factory.mktemp("salidas")
    c.crear_directorios()
    return c


@pytest.fixture(scope="module")
def vulns(cfg):
    return ingesta.ejecutar(cfg.dir_datos, cfg.patron_fuentes)


@pytest.fixture(scope="module")
def trafico(vulns, cfg):
    return simulador.ejecutar(vulns, cfg)


def test_ingesta_deduplica_y_categoriza(vulns):
    assert vulns["avidId"].is_unique
    assert vulns["categoria_ataque"].notna().all()
    assert set(vulns["categoria_ataque"]) <= set(simulador.ATAQUES)
    assert vulns["cvssScore"].between(0, 10).all()


def test_vector_cvss_completa_campos():
    d = ingesta._parsear_vector("CVSS:3.1/AV:N/AC:H/PR:L/UI:R/S:C/C:H/I:N/A:N")
    assert d["attackVector"] == "Network" and d["attackComplexity"] == "High"
    assert d["privilegesRequired"] == "Low" and d["scope"] == "Changed"


def test_simulador_reproducible_y_etiquetado(vulns, cfg, trafico):
    assert trafico["es_anomalo"].between(0, 1).all()
    # En una simulación corta las 74 campañas pesan más; solo exigimos ambas clases.
    assert set(trafico["es_anomalo"]) == {0, 1}
    assert trafico.loc[trafico.es_anomalo == 1, "avidId"].notna().all()
    otra = simulador.ejecutar(vulns, cfg)
    pd.testing.assert_frame_equal(trafico, otra)


def test_features_sin_nulos_y_sin_fuga_de_etiquetas(trafico):
    f = ExtractorCaracteristicas().transform(trafico)
    assert list(f.columns) == NUMERICAS + CATEGORICAS
    assert not f[NUMERICAS].isna().any().any()
    assert not {"es_anomalo", "avidId", "categoria_ataque"} & set(f.columns)


@pytest.mark.parametrize("tipo", ["isolation_forest", "random_forest"])
def test_entrenamiento_y_deteccion(trafico, cfg, tipo, tmp_path):
    train, test, _ = modelo.particion_temporal(trafico, 0.7)
    art = modelo.entrenar(tipo, train, cfg)
    metricas, _ = modelo.evaluar(art, test, set(train["avidId"].dropna()))
    assert metricas["roc_auc"] > 0.8

    ruta = tmp_path / f"{tipo}.joblib"
    modelo.guardar(art, ruta)
    logs = test.drop(columns=["es_anomalo", "avidId", "categoria_ataque", "fase_ataque", "id_evento"])
    res = detectar(logs, modelo.cargar(ruta))
    assert {"puntuacion_anomalia", "alerta"} <= set(res.columns)
    assert len(res) == len(test)
