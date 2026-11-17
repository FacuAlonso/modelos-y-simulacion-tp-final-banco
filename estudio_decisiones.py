"""Experimentos de decisiones para el gemelo didáctico de mesa de ayuda.

Conceptos: diseño de escenarios, réplicas Monte Carlo, comparación pareada,
objetivos múltiples, restricciones explícitas, sensibilidad uno por vez,
verificación y validación de procedimiento sobre referencia sintética.

No estima costos monetarios reales ni valida una entidad bancaria real.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from simulacion_tickets_banco import (
    Configuracion, EQUIPOS, generar_tickets, simular, semiamplitud_ic,
    pruebas_verificacion, calibracion_y_validacion, validar_config,
)

# Tarifas relativas e ILUSTRATIVAS por hora de dotación; no son pesos ni salarios.
TARIFAS_RELATIVAS = {"N1": 1.0, "N2_aplicaciones": 1.6,
                    "N2_infraestructura": 1.6, "N3": 2.2}


def costo_relativo(c: Configuracion) -> float:
    return sum((fin - ini) / 60.0 * c.dias * TARIFAS_RELATIVAS[e]
               for e in EQUIPOS for ini, fin in c.turnos[e])


def _con_turnos(c, equipo, quitar=0, agregar=0):
    s = copy.deepcopy(c.turnos)
    if quitar and len(s[equipo]) < quitar:
        raise ValueError(f"No se pueden quitar {quitar} técnicos de {equipo}")
    if quitar:
        # La baja se modela retirando el último turno listado, sin reoptimizar otros.
        s[equipo] = s[equipo][:-quitar]
    if agregar:
        s[equipo] += [[0, 480]] * agregar
    return replace(c, turnos=s)


def alternativas(c):
    """Cambios individuales sobre la MISMA base, para atribuir efectos."""
    s = copy.deepcopy(c.turnos)
    n = len(s["N1"])
    s["N1"] = [[0, 480]] * (n // 2) + [[120, 600]] * (n - n // 2)
    casos = {
        "Base": copy.deepcopy(c),
        "Capacitación N1 (−25 % escaladas)": replace(c, factor_escalamiento_n1=c.factor_escalamiento_n1 * 0.75),
        "Redistribución de turnos": replace(c, turnos=s),
        "Refuerzo N2 (+1 técnico)": _con_turnos(c, "N2_aplicaciones", agregar=1),
        "Refuerzo N1 (+1 técnico)": _con_turnos(c, "N1", agregar=1),
        "Reducción N1 (−1 técnico)": _con_turnos(c, "N1", quitar=1),
        "Reducción N2 (−1 técnico)": _con_turnos(c, "N2_aplicaciones", quitar=1),
        "Atención más rápida (−15 % tiempo)": replace(c, factor_servicio=c.factor_servicio * 0.85),
        "Prioridad con envejecimiento": replace(c, politica_prioridad="envejecimiento"),
    }
    for cfg in casos.values():
        validar_config(cfg)
    return casos


def _estadistica(diferencias):
    a = np.asarray(diferencias, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"diferencia": None, "ic95": [None, None]}
    m = float(a.mean())
    h = semiamplitud_ic(a)
    return {"diferencia": round(m, 4), "ic95": [round(m-h, 4), round(m+h, 4)]
            if math.isfinite(h) else [None, None]}


def _corridas(casos, c, replicas):
    """Diseño bloqueado: mismo conjunto de semillas para cada política.

    Al variar demanda/tasa de Poisson, la semilla común no garantiza tickets
    idénticos; las comparaciones a igualdad de DEMANDA sí son pareadas por semilla.
    """
    nombres = list(casos)
    datos = {nombre: [] for nombre in nombres}
    for r in range(replicas):
        semilla = c.semilla + 101 * r
        for nombre, cfg in casos.items():
            metrics, *_ = simular(cfg, generar_tickets(cfg, semilla))
            datos[nombre].append(metrics)
    return datos


def _media(filas, campo):
    a = np.asarray([f.get(campo, float('nan')) for f in filas], dtype=float)
    a = a[np.isfinite(a)]
    return round(float(np.mean(a)), 4) if len(a) else None


def _tabla(casos, corridas, base="Base"):
    campos = ("ingresados", "resueltos", "pendientes", "resolucion_p95_min",
              "resolucion_p95_con_abiertos_min", "respuesta_p95_min",
              "incumplimiento_resolucion_pct", "incumplimiento_respuesta_pct",
              "resolucion_p95_critica_min", "cola_maxima", "throughput_por_hora")
    basef = corridas[base]
    salida = []
    for nombre, cfg in casos.items():
        filas = corridas[nombre]
        indicadores = {k: _media(filas, k) for k in campos}
        dif = {k: _estadistica([a[k] - b[k] for a, b in zip(filas, basef)])
               for k in campos}
        costo = costo_relativo(cfg)
        costo_base = costo_relativo(casos[base])
        salida.append({"nombre": nombre, "indicadores": indicadores,
                       "costo_relativo": round(costo, 3),
                       "variacion_costo_pct": round(100 * (costo / costo_base - 1), 3) if costo_base else None,
                       "diferencias": dif,
                       "horas_dotacion": round(_media(filas, "horas_dotacion"), 3)})
    return salida


def _limite_sup(fila, clave):
    v = fila["diferencias"][clave]["ic95"][1]
    return v if v is not None else float('inf')


def _factible(fila, sla_pp=1.0, backlog_tickets=5.0):
    # Regla conservadora: el LÍMITE SUPERIOR del IC95% respeta el umbral.
    return (_limite_sup(fila, "incumplimiento_resolucion_pct") <= sla_pp
            and _limite_sup(fila, "pendientes") <= backlog_tickets)


def _decisiones(tabla, estres):
    base = next(x for x in tabla if x["nombre"] == "Base")
    # Todas las decisiones son condicionales a las restricciones ilustrativas.
    factibles = [x for x in tabla if x["nombre"] == "Base" or _factible(x)]

    def val(x, campo):
        a = x["indicadores"].get(campo)
        return a if a is not None else float('inf')

    sla = min(factibles, key=lambda x: (val(x, "incumplimiento_resolucion_pct"),
                                       val(x, "pendientes"), x["costo_relativo"]))
    costos = min(factibles, key=lambda x: (x["costo_relativo"],
                                          val(x, "incumplimiento_resolucion_pct")))

    # Puntuación explícita y solo para ilustrar preferencias, no verdad económica.
    # Normalizada: +1 por 2 puntos de SLA, +1 por 5 tickets o +1 por 10% del costo base.
    # Penaliza aumentos y recompensa reducciones; 50/30/20 es una preferencia arbitraria.
    def puntaje(x):
        return (0.5 * (val(x, "incumplimiento_resolucion_pct") - val(base, "incumplimiento_resolucion_pct")) / 2.0
                + 0.3 * (val(x, "pendientes") - val(base, "pendientes")) / 5.0
                + 0.2 * (x["costo_relativo"] - base["costo_relativo"]) / (0.1 * base["costo_relativo"]))
    balance = min(factibles, key=puntaje)

    # Para resiliencia comparamos todas las políticas bajo +30% de demanda,
    # incluyendo el escenario base estresado, con un tope de +20% del costo normal.
    baratos = [x for x in estres if x["costo_relativo"] <= 1.2 * base["costo_relativo"] + 1e-9]
    resiliencia = min(baratos, key=lambda x: (val(x, "pendientes"),
                                             val(x, "incumplimiento_resolucion_pct")))
    respuesta = min(factibles, key=lambda x: val(x, "respuesta_p95_min"))
    return {"sla": sla["nombre"], "costos": costos["nombre"],
            "balance": balance["nombre"], "resiliencia": resiliencia["nombre"],
            "respuesta": respuesta["nombre"],
            "factibles": [x["nombre"] for x in factibles],
            "criterio": "IC95 superior: aumento SLA ≤ 1 punto porcentual y backlog ≤ 5 tickets; "
                        "base siempre admitida como referencia; costo relativo por horas de dotación.",
            "ponderaciones_balance": {"SLA": 0.5, "backlog": 0.3, "costo": 0.2},
            "puntajes_balance": {x["nombre"]: round(puntaje(x), 3) for x in factibles}}


def _sensibilidad(c, replicas):
    """Un factor a la vez ±20 %, sin mezclarlo con mejoras de políticas."""
    casos = {"Base": copy.deepcopy(c)}
    factores = {"Demanda": "factor_demanda", "Duración del servicio": "factor_servicio",
                "Escalamiento N1": "factor_escalamiento_n1"}
    for etiqueta, atributo in factores.items():
        for m in (0.8, 1.2):
            casos[f"{etiqueta} {'−20%' if m == .8 else '+20%'}"] = replace(
                c, **{atributo: getattr(c, atributo) * m})
    tabla = _tabla(casos, _corridas(casos, c, replicas))
    return [{"factor": etiqueta, "variacion": m,
             "cambio_p95": next(f for f in tabla if f["nombre"] == f"{etiqueta} {m}")
             ["diferencias"]["resolucion_p95_con_abiertos_min"],
             "cambio_sla": next(f for f in tabla if f["nombre"] == f"{etiqueta} {m}")
             ["diferencias"]["incumplimiento_resolucion_pct"]}
            for etiqueta in factores for m in ("−20%", "+20%")]


def evaluar_gemelo(c, replicas=6):
    """Verifica reglas y ensaya el PROCESO de calibración con referencia sintética.

    Un histórico artificial distinto se reserva para evaluación; no se introduce
    en la calibración. No es validación empírica, ni garantía de predictive validity.
    """
    pruebas_verificacion()
    v = calibracion_y_validacion(c, None, replicas)
    tabla = v["validacion"]
    n_cal = int(tabla["calibrado_consistente"].sum())
    n_sup = int(tabla["supuesto_consistente"].sum())
    return {
        "verificacion": "superada", "proceso_validacion": "referencia sintética independiente reservada (25 %)",
        "origen_real": False, "dias_calibracion": v["dias_calibracion"],
        "dias_reserva": v["dias_validacion"], "indicadores_evaluados": len(tabla),
        "indicadores_dentro_banda_calibrado": n_cal,
        "indicadores_dentro_banda_supuesto": n_sup,
        "comparaciones": [{"indicador": str(f["metrica"]),
                           "referencia": None if not math.isfinite(float(f["observado"])) else round(float(f["observado"]), 2),
                           "calibrado": None if not math.isfinite(float(f["calibrado_media"])) else round(float(f["calibrado_media"]), 2),
                           "dentro_banda": bool(f["calibrado_consistente"])}
                          for _, f in tabla.iterrows()],
        "advertencia": "Comparación controlada con un sistema sintético de la misma familia. "
                       "No demuestra correspondencia con el banco ni exactitud fuera de la muestra. "
                       "Las bandas son de dispersión simulada, no intervalos de confianza para parámetros reales."
    }


def ejecutar_estudio(c=None, replicas=12, dias=3, incluir_gemelo=True):
    c = copy.deepcopy(c or Configuracion())
    c.dias, c.replicas = int(dias), int(replicas)
    validar_config(c)
    if not 2 <= replicas <= 100 or not 1 <= dias <= 30:
        raise ValueError("Se requieren 2-100 réplicas y 1-30 días")
    casos = alternativas(c)
    normal = _tabla(casos, _corridas(casos, c, replicas))
    estres_cfg = {nombre: replace(x, factor_demanda=x.factor_demanda * 1.3) for nombre, x in casos.items()}
    estres = _tabla(estres_cfg, _corridas(estres_cfg, c, replicas))
    decisiones = _decisiones(normal, estres)
    sens = _sensibilidad(c, replicas)
    gemelo = evaluar_gemelo(c, min(6, replicas)) if incluir_gemelo else None
    return {"configuracion": {"dias": dias, "dias_calentamiento": c.dias_calentamiento,
                               "replicas": replicas, "semilla": c.semilla},
            "costos": {"unidad": "unidades relativas por hora; valores ilustrativos", "tarifas": TARIFAS_RELATIVAS},
            "escenarios": normal, "estres_mas_30pct": estres,
            "decisiones": decisiones, "sensibilidad": sens, "evaluacion_gemelo": gemelo,
            "advertencias": [
                "Los datos y costos son sintéticos; no se infiere impacto económico en pesos.",
                "Decisiones condicionales a objetivos, tolerancias y parámetros supuestos.",
                "P95 de cerrados puede mejorar si quedan muchos abiertos: usar P95 incluyendo abiertos y backlog.",
                "La misma semilla no garantiza tickets idénticos al cambiar tasas de demanda.",
                "Los 95% son intervalos de la diferencia media entre réplicas, no intervalos de predicción de un banco real.",
                "Varias políticas/indicadores: la selección entre múltiples pruebas es exploratoria, sin corrección por multiplicidad."
            ]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replicas", type=int, default=12)
    p.add_argument("--dias", type=int, default=3)
    p.add_argument("--salida", type=Path, default=Path("resultados_decisiones.json"))
    p.add_argument("--sin-evaluacion", action="store_true", help="Omitir prueba de evaluación sintética")
    args = p.parse_args()
    resultado = ejecutar_estudio(replicas=args.replicas, dias=args.dias,
                                 incluir_gemelo=not args.sin_evaluacion)
    args.salida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Preguntas de decisión (resultado sintético, no validación de banco real):")
    for objetivo, nombre in resultado["decisiones"].items():
        if objetivo in ("sla", "costos", "balance", "resiliencia", "respuesta"):
            print(f"  {objetivo:12}: {nombre}")
    print("Evaluación del procedimiento:", resultado["evaluacion_gemelo"] and
          resultado["evaluacion_gemelo"]["verificacion"])
    print("JSON:", args.salida)


if __name__ == "__main__":
    main()
