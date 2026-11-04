"""Simulación de eventos discretos de una mesa de ayuda bancaria.

Unidades: minutos corridos; las atenciones solo avanzan durante el turno del técnico.
Datos y SLA de ejemplo: supuestos didácticos, no información real de un banco.
"""

import argparse
import heapq
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# PARÁMETROS EDITABLES

PRIORIDADES = ("critica", "alta", "media", "baja")
CATEGORIAS = ("accesos", "aplicaciones", "conectividad", "hardware")
EQUIPOS = ("N1", "N2_aplicaciones", "N2_infraestructura", "N3")
MINUTOS_DIA = 1440
MINUTOS_ATENCION = 600


@dataclass
class Configuracion:
    dias: int = 5
    semilla: int = 2026
    replicas: int = 30
    tasas_horarias: list = field(default_factory=lambda: [15, 19, 16, 12, 9, 10, 14, 12, 8, 5])
    factores_dia: list = field(default_factory=lambda: [1.0, 1.12, 0.95, 1.08, 0.92])
    proporcion_categorias: list = field(default_factory=lambda: [0.32, 0.38, 0.18, 0.12])
    proporcion_complejidad: list = field(default_factory=lambda: [0.48, 0.37, 0.15])
    turnos: dict = field(default_factory=lambda: {
        "N1": [[0, 480]] * 4 + [[120, 600]] * 2,
        "N2_aplicaciones": [[0, 480]] * 2 + [[120, 600]],
        "N2_infraestructura": [[0, 480], [120, 600]],
        "N3": [[60, 540]],
    })
    servicio_triangular: dict = field(default_factory=lambda: {
        "N1": [6, 12, 25],
        "N2_aplicaciones": [12, 28, 65],
        "N2_infraestructura": [14, 35, 70],
        "N3": [30, 60, 110],
    })
    factor_complejidad: dict = field(default_factory=lambda: {
        "simple": 0.70, "media": 1.0, "compleja": 1.6
    })
    prob_escalamiento_n1: dict = field(default_factory=lambda: {
        "accesos": 0.12, "aplicaciones": 0.44,
        "conectividad": 0.32, "hardware": 0.25,
    })
    prob_escalamiento_n2: float = 0.10
    prob_reapertura: float = 0.025
    sla_respuesta: dict = field(default_factory=lambda: {
        "critica": 20, "alta": 60, "media": 180, "baja": 360,
    })
    sla_resolucion: dict = field(default_factory=lambda: {
        "critica": 180, "alta": 480, "media": 1440, "baja": 2880,
    })
    politica_prioridad: str = "estricta"
    minutos_envejecimiento: int = 240
    factor_demanda: float = 1.0
    factor_servicio: float = 1.0
    factor_escalamiento_n1: float = 1.0
    incidente_masivo: int = 0


CONFIG = Configuracion()


# ENTIDADES Y GENERACIÓN DE DEMANDA

@dataclass
class Ticket:
    id: int
    llegada: float
    prioridad: str
    categoria: str
    complejidad: str
    servicio: dict
    u_escala_n1: float
    u_escala_n2: float
    u_reabrir: float
    demora_reapertura: float
    equipo: str = "N1"
    estado: str = "nuevo"
    ingreso_cola: float = 0.0
    primera_respuesta: float = None
    cierre: float = None
    espera_acumulada: float = 0.0
    trabajo_terminado: float = 0.0
    escaladas: int = 0
    reaperturas: int = 0


@dataclass
class Tecnico:
    id: str
    equipo: str
    turno_inicio: int
    turno_fin: int
    ticket: int = None
    asignaciones: list = field(default_factory=list)


def validar_config(c):
    assert c.dias >= 1 and c.replicas >= 1
    assert len(c.tasas_horarias) == 10 and all(x >= 0 for x in c.tasas_horarias)
    assert len(c.factores_dia) > 0 and all(x >= 0 for x in c.factores_dia)
    assert len(c.proporcion_categorias) == 4 and abs(sum(c.proporcion_categorias) - 1) < 1e-7
    assert len(c.proporcion_complejidad) == 3 and abs(sum(c.proporcion_complejidad) - 1) < 1e-7
    assert set(c.turnos) == set(EQUIPOS) and set(c.servicio_triangular) == set(EQUIPOS)
    assert c.factor_demanda >= 0 and c.factor_servicio > 0 and c.factor_escalamiento_n1 >= 0
    assert c.incidente_masivo >= 0 and c.minutos_envejecimiento > 0
    assert c.politica_prioridad in ("estricta", "envejecimiento")
    for equipo in EQUIPOS:
        a, m, b = c.servicio_triangular[equipo]
        assert 0 < a <= m <= b
        for inicio, fin in c.turnos[equipo]:
            assert 0 <= inicio < fin <= MINUTOS_ATENCION
    for p in PRIORIDADES:
        assert c.sla_respuesta[p] > 0 and c.sla_resolucion[p] > 0
    assert all(0 <= x <= 1 for x in c.prob_escalamiento_n1.values())
    assert 0 <= c.prob_escalamiento_n2 <= 1 and 0 <= c.prob_reapertura <= 1


def generar_tickets(c, semilla):
    rng = np.random.default_rng(semilla)
    tickets = []
    probabilidades_prioridad = {
        "simple": [0.015, 0.12, 0.58, 0.285],
        "media": [0.045, 0.25, 0.60, 0.105],
        "compleja": [0.18, 0.45, 0.35, 0.02],
    }

    def crear(llegada, categoria=None, prioridad=None, complejidad=None):
        cat = categoria or str(rng.choice(CATEGORIAS, p=c.proporcion_categorias))
        comp = complejidad or str(rng.choice(["simple", "media", "compleja"], p=c.proporcion_complejidad))
        pri = prioridad or str(rng.choice(PRIORIDADES, p=probabilidades_prioridad[comp]))
        tiempos = {}
        for equipo, (a, m, b) in c.servicio_triangular.items():
            tiempos[equipo] = max(0.1, float(rng.triangular(a, m, b)) * c.factor_complejidad[comp] * c.factor_servicio)
        tickets.append(Ticket(
            id=len(tickets) + 1, llegada=float(llegada), prioridad=pri,
            categoria=cat, complejidad=comp, servicio=tiempos,
            u_escala_n1=float(rng.random()), u_escala_n2=float(rng.random()),
            u_reabrir=float(rng.random()), demora_reapertura=float(rng.uniform(30, 180)),
        ))

    for dia in range(c.dias):
        for hora, tasa in enumerate(c.tasas_horarias):
            media = tasa * c.factor_demanda * c.factores_dia[dia % len(c.factores_dia)]
            for instante in rng.uniform(0, 60, size=rng.poisson(media)):
                crear(dia * MINUTOS_DIA + hora * 60 + instante)

    if c.incidente_masivo:
        dia = min(2, c.dias - 1)
        for instante in rng.uniform(0, 4, size=c.incidente_masivo):
            crear(dia * MINUTOS_DIA + 30 + instante,
                  categoria="aplicaciones", prioridad="critica", complejidad="compleja")

    tickets.sort(key=lambda x: (x.llegada, x.id))
    for i, ticket in enumerate(tickets, start=1):
        ticket.id = i
    return tickets


# RELOJ, TURNOS Y CALENDARIO DE EVENTOS

def esta_de_turno(tecnico, instante):
    minuto = instante % MINUTOS_DIA
    return tecnico.turno_inicio <= minuto < tecnico.turno_fin


def fin_del_trabajo(tecnico, inicio, duracion, dias):
    restante = duracion
    dia = int(inicio // MINUTOS_DIA)
    instante = inicio
    while dia < dias:
        abre = dia * MINUTOS_DIA + tecnico.turno_inicio
        cierra = dia * MINUTOS_DIA + tecnico.turno_fin
        comienzo = max(instante, abre)
        if comienzo < cierra:
            disponible = cierra - comienzo
            if restante <= disponible + 1e-9:
                return comienzo + restante
            restante -= disponible
        dia += 1
        instante = dia * MINUTOS_DIA
    return math.inf


def trabajo_en_horizonte(tecnico, inicio, fin, horizonte, dias):
    total = 0.0
    hasta = min(fin, horizonte)
    if hasta <= inicio:
        return 0.0
    for dia in range(dias):
        abre = dia * MINUTOS_DIA + tecnico.turno_inicio
        cierra = dia * MINUTOS_DIA + tecnico.turno_fin
        total += max(0.0, min(hasta, cierra) - max(inicio, abre))
    return total


def simular(c, tickets, guardar_traza=False):
    horizonte = (c.dias - 1) * MINUTOS_DIA + MINUTOS_ATENCION
    calendario = []
    correlativo = 0
    tecnicos = {equipo: [] for equipo in EQUIPOS}
    colas = {equipo: [] for equipo in EQUIPOS}
    historial = []
    evolucion = []
    max_cola = 0
    ultimo_instante = 0.0
    areas_cola = {e: 0.0 for e in EQUIPOS}
    por_id = {t.id: t for t in tickets}

    def evento(t, tipo, identificador):
        nonlocal correlativo
        heapq.heappush(calendario, (float(t), correlativo, tipo, identificador))
        correlativo += 1

    def trazar(t, tipo, ticket, equipo):
        if guardar_traza:
            historial.append({"minuto": t, "dia": int(t // MINUTOS_DIA) + 1,
                              "evento": tipo, "ticket": ticket, "equipo": equipo})

    def ingresar_cola(ticket, equipo, instante):
        ticket.equipo = equipo
        ticket.estado = "en_cola"
        ticket.ingreso_cola = instante
        colas[equipo].append(ticket.id)

    for equipo in EQUIPOS:
        for i, (inicio, fin) in enumerate(c.turnos[equipo], 1):
            tecnico = Tecnico(f"{equipo}_{i}", equipo, inicio, fin)
            tecnicos[equipo].append(tecnico)
            for dia in range(c.dias):
                evento(dia * MINUTOS_DIA + inicio, "inicio_turno", tecnico.id)
                evento(dia * MINUTOS_DIA + fin, "fin_turno", tecnico.id)
    for ticket in tickets:
        evento(ticket.llegada, "llegada", ticket.id)

    def seleccionar(equipo, instante):
        def clave(ticket_id):
            ticket = por_id[ticket_id]
            prioridad = PRIORIDADES.index(ticket.prioridad)
            if c.politica_prioridad == "envejecimiento":
                prioridad = max(0, prioridad - int((instante - ticket.ingreso_cola) // c.minutos_envejecimiento))
            return (prioridad, ticket.ingreso_cola, ticket.id)
        elegido = min(colas[equipo], key=clave)
        colas[equipo].remove(elegido)
        return por_id[elegido]

    def minutos_operativos(desde, hasta):
        return sum(max(0.0, min(hasta, dia * MINUTOS_DIA + MINUTOS_ATENCION)
                       - max(desde, dia * MINUTOS_DIA)) for dia in range(c.dias))

    while calendario and calendario[0][0] <= horizonte + 1e-8:
        instante = calendario[0][0]
        transcurrido = minutos_operativos(ultimo_instante, instante)
        for equipo in EQUIPOS:
            areas_cola[equipo] += len(colas[equipo]) * transcurrido
        ultimo_instante = instante
        lote = []
        while calendario and abs(calendario[0][0] - instante) < 1e-8:
            lote.append(heapq.heappop(calendario))

        for _, _, tipo, identificador in lote:
            if tipo == "llegada":
                ticket = por_id[identificador]
                ingresar_cola(ticket, "N1", instante)
                trazar(instante, "apertura", ticket.id, "N1")
            elif tipo == "reabrir":
                ticket = por_id[identificador]
                assert ticket.estado == "resuelto"
                ticket.reaperturas += 1
                ticket.cierre = None
                ingresar_cola(ticket, "N1", instante)
                trazar(instante, "reapertura", ticket.id, "N1")
            elif tipo == "fin_servicio":
                equipo, id_tecnico, id_ticket = identificador
                tecnico = next(t for t in tecnicos[equipo] if t.id == id_tecnico)
                ticket = por_id[id_ticket]
                assert tecnico.ticket == id_ticket and ticket.estado == "en_atencion"
                tecnico.ticket = None
                ticket.trabajo_terminado += ticket.servicio[equipo]
                if equipo == "N1":
                    p = c.prob_escalamiento_n1[ticket.categoria] * c.factor_escalamiento_n1
                    p *= c.factor_complejidad[ticket.complejidad]
                    escalar = ticket.u_escala_n1 < min(1.0, p)
                    destino = ("N2_aplicaciones" if ticket.categoria in ("accesos", "aplicaciones")
                               else "N2_infraestructura")
                elif equipo.startswith("N2"):
                    p = c.prob_escalamiento_n2 * c.factor_complejidad[ticket.complejidad]
                    escalar = ticket.u_escala_n2 < min(1.0, p)
                    destino = "N3"
                else:
                    escalar = False
                    destino = None
                if escalar:
                    ticket.escaladas += 1
                    ingresar_cola(ticket, destino, instante)
                    trazar(instante, "escalamiento", ticket.id, destino)
                else:
                    ticket.estado = "resuelto"
                    ticket.cierre = instante
                    trazar(instante, "resolucion", ticket.id, equipo)
                    if ticket.reaperturas == 0 and ticket.u_reabrir < c.prob_reapertura:
                        retraso = instante + ticket.demora_reapertura
                        if retraso <= horizonte:
                            evento(retraso, "reabrir", ticket.id)
            elif tipo == "inicio_turno":
                trazar(instante, "inicio_turno", None, identificador)
            elif tipo == "fin_turno":
                trazar(instante, "fin_turno", None, identificador)

        for equipo in EQUIPOS:
            libres = [t for t in tecnicos[equipo] if t.ticket is None and esta_de_turno(t, instante)]
            for tecnico in libres:
                if not colas[equipo]:
                    break
                ticket = seleccionar(equipo, instante)
                assert ticket.estado == "en_cola" and ticket.equipo == equipo
                ticket.espera_acumulada += instante - ticket.ingreso_cola
                if ticket.primera_respuesta is None:
                    ticket.primera_respuesta = instante
                ticket.estado = "en_atencion"
                tecnico.ticket = ticket.id
                fin = fin_del_trabajo(tecnico, instante, ticket.servicio[equipo], c.dias)
                tecnico.asignaciones.append((instante, fin))
                if math.isfinite(fin):
                    evento(fin, "fin_servicio", (equipo, tecnico.id, ticket.id))
                trazar(instante, "inicio_atencion", ticket.id, tecnico.id)

        max_cola = max(max_cola, sum(len(cola) for cola in colas.values()))
        if guardar_traza:
            evolucion.append({"minuto": instante, **{e: len(colas[e]) for e in EQUIPOS},
                              "pendientes": sum(len(cola) for cola in colas.values())})

    transcurrido = minutos_operativos(ultimo_instante, horizonte)
    for equipo in EQUIPOS:
        areas_cola[equipo] += len(colas[equipo]) * transcurrido

    assert all(t.estado in ("en_cola", "en_atencion", "resuelto") for t in tickets)
    assert len(tickets) == sum(t.estado == "resuelto" for t in tickets) + sum(
        t.estado == "en_cola" for t in tickets) + sum(t.estado == "en_atencion" for t in tickets)
    assert all(t.espera_acumulada >= -1e-8 for t in tickets)
    assert all(sum(tecnico.ticket is not None for tecnico in tecnicos[e]) <= len(tecnicos[e]) for e in EQUIPOS)

    resueltos = [t for t in tickets if t.estado == "resuelto"]
    respondidos = [t for t in tickets if t.primera_respuesta is not None]
    pendientes = [t for t in tickets if t.estado != "resuelto"]
    demoras_resp = [t.primera_respuesta - t.llegada for t in respondidos]
    demoras_res = [t.cierre - t.llegada for t in resueltos]

    def promedio(valores):
        return float(np.mean(valores)) if valores else math.nan

    def percentil(valores, p):
        return float(np.percentile(valores, p)) if valores else math.nan

    def incumplimiento(tipo):
        incumple, evaluables = 0, 0
        for t in tickets:
            limite = (c.sla_respuesta if tipo == "respuesta" else c.sla_resolucion)[t.prioridad]
            final = t.primera_respuesta if tipo == "respuesta" else t.cierre
            if final is not None:
                evaluables += 1
                incumple += final - t.llegada > limite
            elif horizonte - t.llegada >= limite:
                evaluables += 1
                incumple += 1
        return 100 * incumple / evaluables if evaluables else math.nan

    ocupacion = {}
    for equipo, personas in tecnicos.items():
        capacidad = sum((t.turno_fin - t.turno_inicio) * c.dias for t in personas)
        trabajo = sum(trabajo_en_horizonte(t, inicio, fin, horizonte, c.dias)
                      for t in personas for inicio, fin in t.asignaciones)
        assert trabajo <= capacidad + 1e-6
        ocupacion[equipo] = 100 * trabajo / capacidad if capacidad else math.nan

    resumen = {
        "ingresados": len(tickets), "resueltos": len(resueltos),
        "pendientes": len(pendientes), "en_cola": sum(t.estado == "en_cola" for t in tickets),
        "en_atencion": sum(t.estado == "en_atencion" for t in tickets),
        "respuesta_media_min": promedio(demoras_resp),
        "respuesta_p90_min": percentil(demoras_resp, 90),
        "respuesta_p95_min": percentil(demoras_resp, 95),
        "resolucion_media_min": promedio(demoras_res),
        "resolucion_p90_min": percentil(demoras_res, 90),
        "resolucion_p95_min": percentil(demoras_res, 95),
        "espera_media_resueltos_min": promedio([t.espera_acumulada for t in resueltos]),
        "throughput_por_hora": len(resueltos) / (c.dias * 10),
        "escaladas": sum(t.escaladas for t in tickets),
        "reaperturas": sum(t.reaperturas for t in tickets),
        "incumplimiento_respuesta_pct": incumplimiento("respuesta"),
        "incumplimiento_resolucion_pct": incumplimiento("resolucion"),
        "cola_maxima": max_cola,
        "cola_promedio": sum(areas_cola.values()) / (c.dias * MINUTOS_ATENCION),
        **{f"cola_promedio_{e}": areas_cola[e] / (c.dias * MINUTOS_ATENCION) for e in EQUIPOS},
        "horas_dotacion": sum((t.turno_fin - t.turno_inicio) * c.dias / 60
                               for personas in tecnicos.values() for t in personas),
        **{f"utilizacion_{e}_pct": ocupacion[e] for e in EQUIPOS},
        **{f"pendientes_{e}": sum(t.equipo == e and t.estado != "resuelto" for t in pendientes)
           for e in EQUIPOS},
    }

    detalle = pd.DataFrame([{
        "ticket": t.id, "dia_apertura": int(t.llegada // MINUTOS_DIA) + 1,
        "minuto_apertura": t.llegada, "categoria": t.categoria,
        "prioridad": t.prioridad, "complejidad": t.complejidad,
        "estado_final": t.estado, "equipo_final": t.equipo,
        "minuto_primera_respuesta": t.primera_respuesta,
        "minuto_cierre": t.cierre,
        "tiempo_respuesta_min": (t.primera_respuesta - t.llegada if t.primera_respuesta is not None else np.nan),
        "tiempo_resolucion_min": (t.cierre - t.llegada if t.cierre is not None else np.nan),
        "espera_acumulada_min": t.espera_acumulada,
        "escaladas": t.escaladas, "reaperturas": t.reaperturas,
    } for t in tickets])
    return resumen, detalle, pd.DataFrame(historial), pd.DataFrame(evolucion)


# EXPERIMENTOS, MONTE CARLO Y SENSIBILIDAD

def escenarios(c):
    base = replace(c)
    turnos_equilibrados = {e: [list(t) for t in turnos] for e, turnos in c.turnos.items()}
    turnos_equilibrados["N1"] = [[0, 480]] * 3 + [[120, 600]] * 3
    refuerzo = {e: [list(t) for t in turnos] for e, turnos in c.turnos.items()}
    refuerzo["N2_aplicaciones"].append([0, 480])
    adverso = {e: [list(t) for t in turnos] for e, turnos in c.turnos.items()}
    if adverso["N2_aplicaciones"]:
        adverso["N2_aplicaciones"].pop()
    return {
        "Base": base,
        "Turnos redistribuidos": replace(c, turnos=turnos_equilibrados),
        "Menos escalamiento N1": replace(c, factor_escalamiento_n1=0.75 * c.factor_escalamiento_n1),
        "Refuerzo especialistas": replace(c, turnos=refuerzo),
        "Estrés +30% e incidente": replace(c, turnos=adverso,
                                             factor_demanda=1.30 * c.factor_demanda,
                                             incidente_masivo=40 + c.incidente_masivo),
    }


def ejecutar_experimentos(c, lista_escenarios):
    filas = []
    for nombre, cfg in lista_escenarios.items():
        validar_config(cfg)
        for replica in range(c.replicas):
            semilla = c.semilla + replica * 101
            datos = generar_tickets(cfg, semilla)
            r, _, _, _ = simular(cfg, datos)
            filas.append({"escenario": nombre, "replica": replica + 1,
                          "semilla": semilla, **r})
    df = pd.DataFrame(filas)
    columnas = [x for x in df.select_dtypes(include="number").columns
                if x not in ("replica", "semilla")]
    resumen = df.groupby("escenario", sort=False)[columnas].mean().reset_index()
    for campo in ("respuesta_p95_min", "resolucion_p95_min", "pendientes"):
        desv = df.groupby("escenario", sort=False)[campo].std().reindex(resumen.escenario)
        margen = 1.96 * desv.to_numpy() / math.sqrt(c.replicas) if c.replicas > 1 else np.full(len(resumen), np.nan)
        resumen[f"ic95_inf_{campo}"] = resumen[campo] - margen
        resumen[f"ic95_sup_{campo}"] = resumen[campo] + margen
    return df, resumen


def sensibilidad(c, replicas):
    modificaciones = {
        "Demanda": ("factor_demanda", 0.8, 1.2),
        "Tiempos de servicio": ("factor_servicio", 0.8, 1.2),
        "Escalamiento N1": ("factor_escalamiento_n1", 0.8, 1.2),
    }
    filas = []
    for factor, (campo, minimo, maximo) in modificaciones.items():
        for etiqueta, multiplo in (("-20%", minimo), ("+20%", maximo)):
            cfg = replace(c, **{campo: getattr(c, campo) * multiplo})
            valores = []
            for rep in range(replicas):
                r, _, _, _ = simular(cfg, generar_tickets(cfg, c.semilla + rep * 101))
                valores.append(r["resolucion_p95_min"])
            filas.append({"factor": factor, "variacion": etiqueta,
                          "resolucion_p95_min": float(np.nanmean(valores))})
    return pd.DataFrame(filas)


# REPORTES Y GRÁFICOS

def graficos(comparacion, evolucion, sens, destino):
    etiquetas = comparacion["escenario"].tolist()
    xs = np.arange(len(etiquetas))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    axes[0].bar(xs, comparacion["resolucion_p95_min"])
    axes[0].set_title("P95 de resolución (minutos)")
    axes[1].bar(xs, comparacion["pendientes"])
    axes[1].set_title("Tickets pendientes al final")
    for ax in axes:
        ax.set_xticks(xs, etiquetas, rotation=22, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Mesa de ayuda bancaria · Promedio entre réplicas")
    fig.savefig(destino / "comparacion_escenarios.png", dpi=160)
    plt.close(fig)

    if not evolucion.empty:
        fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
        horas = (evolucion["minuto"] // MINUTOS_DIA) * 10 + np.minimum(
            evolucion["minuto"] % MINUTOS_DIA, MINUTOS_ATENCION) / 60
        for equipo in EQUIPOS:
            ax.step(horas, evolucion[equipo], where="post", label=equipo)
        ax.set(xlabel="Horas operativas acumuladas", ylabel="Tickets en cola",
               title="Evolución de las colas · Una réplica del escenario base")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.savefig(destino / "evolucion_colas.png", dpi=160)
        plt.close(fig)

    if not sens.empty:
        tabla = sens.pivot(index="factor", columns="variacion", values="resolucion_p95_min")
        tabla = tabla.loc[(tabla["+20%"] - tabla["-20%"]).abs().sort_values().index]
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        y = np.arange(len(tabla))
        for i, (_, fila) in enumerate(tabla.iterrows()):
            ax.plot([fila["-20%"], fila["+20%"]], [i, i], "o-", linewidth=5)
        ax.set_yticks(y, tabla.index)
        ax.set(xlabel="P95 promedio de resolución (minutos)",
               title="Sensibilidad · Un factor por vez (±20%)")
        ax.grid(axis="x", alpha=0.25)
        fig.savefig(destino / "sensibilidad.png", dpi=160)
        plt.close(fig)


def pruebas_verificacion():
    c = Configuracion(dias=2, turnos={"N1": [[0, 480]],
                                      "N2_aplicaciones": [[0, 480]],
                                      "N2_infraestructura": [], "N3": []},
                      prob_escalamiento_n1={k: 0 for k in CATEGORIAS},
                      prob_escalamiento_n2=0, prob_reapertura=0)

    def ticket(i, llegada, prioridad="media", escalamiento=1.0):
        return Ticket(i, llegada, prioridad, "aplicaciones", "simple",
                      {e: 2.0 for e in EQUIPOS}, escalamiento, 1.0, 1.0, 30)

    r, df, log, _ = simular(c, [ticket(1, 0, "baja"), ticket(2, 0, "critica")], True)
    inicios = log.loc[log.evento == "inicio_atencion", "ticket"].tolist()
    assert inicios == [2, 1], inicios
    r2, _, log2, _ = simular(c, [ticket(1, 0), ticket(2, 0)], True)
    assert log2.loc[log2.evento == "inicio_atencion", "ticket"].tolist() == [1, 2]
    assert r["resueltos"] == 2 and r["pendientes"] == 0
    assert df.loc[df.ticket == 1, "tiempo_respuesta_min"].iloc[0] == 2

    r, df, log, _ = simular(c, [ticket(1, 0, escalamiento=0)], True)
    assert r["escaladas"] == 0
    c2 = replace(c, prob_escalamiento_n1={k: 1 for k in CATEGORIAS})
    r, df, log, _ = simular(c2, [ticket(1, 0, escalamiento=0)], True)
    assert r["escaladas"] == 1 and r["resueltos"] == 1
    assert abs(df["tiempo_resolucion_min"].iloc[0] - 4) < 1e-9
    tecnico = Tecnico("prueba", "N1", 0, 480)
    assert abs(fin_del_trabajo(tecnico, 470, 20, 2) - 1450) < 1e-9
    print("Verificación: prioridad, FIFO, escalamiento, tiempos y turnos: OK")


def main():
    parser = argparse.ArgumentParser(description="Simulación de tickets de una mesa de ayuda bancaria")
    parser.add_argument("--replicas", type=int, default=None)
    parser.add_argument("--dias", type=int, default=None)
    parser.add_argument("--semilla", type=int, default=None)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--exportar-config", action="store_true")
    parser.add_argument("--solo-base", action="store_true")
    parser.add_argument("--sin-graficos", action="store_true")
    parser.add_argument("--probar", action="store_true")
    parser.add_argument("--salida", type=Path, default=Path("resultados_tickets_banco"))
    args = parser.parse_args()

    if args.probar:
        pruebas_verificacion()
        return

    c = replace(CONFIG)
    if args.config:
        cambios = json.loads(args.config.read_text(encoding="utf-8"))
        desconocidos = set(cambios) - set(asdict(c))
        if desconocidos:
            parser.error(f"Parámetros desconocidos: {sorted(desconocidos)}")
        c = replace(c, **cambios)
    for campo in ("replicas", "dias", "semilla"):
        valor = getattr(args, campo)
        if valor is not None:
            setattr(c, campo, valor)
    validar_config(c)
    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / "configuracion_utilizada.json").write_text(
        json.dumps(asdict(c), ensure_ascii=False, indent=2), encoding="utf-8")
    if args.exportar_config:
        print("Configuración guardada en:", args.salida / "configuracion_utilizada.json")
        return

    lista = {"Base": c} if args.solo_base else escenarios(c)
    replicas, comparacion = ejecutar_experimentos(c, lista)
    base, detalle, eventos, evolucion = simular(c, generar_tickets(c, c.semilla), guardar_traza=True)
    sens = pd.DataFrame() if args.solo_base else sensibilidad(c, min(c.replicas, 10))

    replicas.to_csv(args.salida / "replicas.csv", index=False, encoding="utf-8-sig")
    comparacion.to_csv(args.salida / "comparacion_escenarios.csv", index=False, encoding="utf-8-sig")
    detalle.to_csv(args.salida / "tickets_base.csv", index=False, encoding="utf-8-sig")
    eventos.to_csv(args.salida / "eventos_base.csv", index=False, encoding="utf-8-sig")
    evolucion.to_csv(args.salida / "evolucion_colas_base.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(args.salida / "sensibilidad.csv", index=False, encoding="utf-8-sig")
    if not args.sin_graficos:
        graficos(comparacion, evolucion, sens, args.salida)

    columnas = ["escenario", "ingresados", "resueltos", "pendientes",
                "respuesta_p95_min", "resolucion_p95_min", "incumplimiento_resolucion_pct"]
    print("\nRESULTADOS PROMEDIO POR ESCENARIO (minutos corridos)")
    print(comparacion[columnas].round(1).to_string(index=False))
    print("\nArchivos CSV" + (" y PNG" if not args.sin_graficos else "") + " en:", args.salida.resolve())
    print("SLA, distribuciones, dotación y tasas son SUPUESTOS EDITABLES, no datos bancarios reales.")
    print("Los IC95 aproximan la incertidumbre entre réplicas; una corrida no valida el modelo real.")


if __name__ == "__main__":
    main()
