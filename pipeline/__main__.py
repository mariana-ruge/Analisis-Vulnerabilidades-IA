"""Orquestador: `python -m pipeline [--dias N] [--semilla S]`.

ingesta -> simulación de tráfico -> features -> entrenamiento -> evaluación -> reporte
"""
from __future__ import annotations

import argparse
import time

from . import graficas, ingesta, modelo, simulador
from .config import Config


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Pipeline de detección de tráfico anómalo")
    ap.add_argument("--dias", type=int, help="días de tráfico a simular")
    ap.add_argument("--semilla", type=int)
    ap.add_argument("--peticiones-hora", type=int, help="peticiones normales por hora punta")
    args = ap.parse_args(argv)

    cfg = Config()
    if args.dias:
        cfg.dias_simulacion = args.dias
    if args.semilla is not None:
        cfg.semilla = args.semilla
    if args.peticiones_hora:
        cfg.peticiones_normales_por_hora = args.peticiones_hora
    cfg.crear_directorios()
    t0 = time.time()

    vulns = ingesta.ejecutar(cfg.dir_datos, cfg.patron_fuentes, cfg.dir_salidas / "vulnerabilidades_limpias.csv")
    trafico = simulador.ejecutar(vulns, cfg, cfg.dir_salidas / "trafico_simulado.csv")

    train, test, corte = modelo.particion_temporal(trafico, cfg.fraccion_entrenamiento)
    vistas = set(train["avidId"].dropna())
    print(f"[modelo] corte temporal {corte} | train={len(train)} test={len(test)}")

    metricas, resultados = {}, {}
    for tipo in ("isolation_forest", "random_forest"):
        art = modelo.entrenar(tipo, train, cfg)
        m, res = modelo.evaluar(art, test, vistas)
        m["importancia_features"] = modelo.importancias(art)
        modelo.guardar(art, cfg.dir_modelos / f"{tipo}.joblib")
        metricas[tipo], resultados[tipo] = m, res
        res.to_csv(cfg.dir_salidas / f"predicciones_{tipo}.csv", index=False)
        print(
            f"[modelo] {tipo:17s} precision={m['precision']:.3f} recall={m['recall']:.3f} "
            f"f1={m['f1']:.3f} roc_auc={m['roc_auc']:.3f} pr_auc={m['pr_auc']:.3f} "
            f"FPR={m['tasa_falsos_positivos']:.3%}"
        )

    graficas.curvas_pr(resultados, cfg.dir_graficas / "curvas_precision_recall.png")
    graficas.deteccion_por_categoria(metricas, cfg.dir_graficas / "recall_por_categoria.png")

    resumen = {
        "vulnerabilidades": len(vulns),
        "categorias": vulns["categoria_ataque"].value_counts().to_dict(),
        "eventos": len(trafico),
        "tasa_anomalia": round(float(trafico["es_anomalo"].mean()), 4),
        "corte_temporal": str(corte),
        "metricas": metricas,
        "segundos": round(time.time() - t0, 1),
    }
    modelo.guardar_json(resumen, cfg.dir_salidas / "reporte_metricas.json")
    print(f"[pipeline] listo en {resumen['segundos']}s -> {cfg.dir_salidas}")
    return resumen


if __name__ == "__main__":
    main()
