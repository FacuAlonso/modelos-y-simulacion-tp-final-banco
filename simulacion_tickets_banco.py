"""Simulación de eventos discretos de una mesa de ayuda bancaria.

Unidades: minutos corridos; las atenciones solo avanzan durante el turno del técnico.
Datos y SLA de ejemplo: supuestos didácticos, no información real de un banco.

Versión 2 (copia local ampliada):
  - Calibración y validación operativa, con historial real (--historico)
    o, si no hay, con una referencia sintética que solo prueba el procedimiento.
  - Parte predictiva: demanda por hora y probabilidad de escalamiento
    con baseline, regresión logística y red neuronal pequeña, más su acople al simulador.
  - Salida organizada según las preguntas del informe teórico (informe_resultados.md).
"""

import argparse
import copy
import heapq
import json
import math
import sys
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
TRABAJO_DUPLICADO = 2.0   # minutos que N1 tarda en detectar y cerrar un duplicado
FRACCION_TRAS_PAUSA = 0.3  # parte del trabajo de N1 que queda cuando el usuario responde


@dataclass
class Configuracion:
    dias: int = 5
    dias_calentamiento: int = 1   # días simulados antes de medir (evita arrancar con colas vacías)
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
    max_reaperturas: int = 2
    prob_pausa_usuario: float = 0.10       # N1 necesita información del empleado
    pausa_usuario_min: list = field(default_factory=lambda: [30, 240])  # demora uniforme
    prob_duplicado: float = 0.03           # se cierra administrativamente, no es resolución
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
    distribucion_llegadas: str = "poisson"       # o "deterministica"
    distribucion_servicio: str = "triangular"    # o "fija" (usa la moda)


CONFIG = Configuracion()


def verificar(condicion, mensaje):
    """Reemplaza a assert: sigue activo aunque Python se ejecute con -O."""
    if not condicion:
        raise ValueError(mensaje)


T_STUDENT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
                9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
                16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 24: 2.064, 29: 2.045,
                39: 2.023, 49: 2.010, 59: 2.001, 99: 1.984}


def t_critico(n):
    """Valor t de Student (95 %, dos colas) para n réplicas. Con n grande tiende a 1,96."""
    gl = n - 1
    if gl < 1:
        return math.nan
    menores = [k for k in T_STUDENT_95 if k <= gl]
    return T_STUDENT_95[max(menores)] if gl <= 99 else 1.96


def semiamplitud_ic(valores):
    v = np.asarray([x for x in valores if np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return math.nan
    return float(t_critico(len(v)) * v.std(ddof=1) / math.sqrt(len(v)))


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
    u_pausa: float = 1.0
    demora_pausa: float = 0.0
    duplicado: bool = False
    u_reabrir_extra: list = field(default_factory=lambda: [1.0, 1.0])
    equipo: str = "N1"
    estado: str = "nuevo"
    ingreso_cola: float = 0.0      # para medir la espera de este paso por la cola
    orden_cola: float = 0.0        # para FIFO; un ticket interrumpido conserva su lugar
    restante: float = 0.0          # trabajo que falta en la etapa actual
    etapa_n1: str = "inicial"      # inicial / tras_pausa / reapertura
    primera_respuesta: float = None
    cierre: float = None
    espera_acumulada: float = 0.0
    trabajo_terminado: float = 0.0
    tiempo_pausa: float = 0.0
    escaladas: int = 0
    reaperturas: int = 0
    interrupciones: int = 0
    cola_n1_al_llegar: int = 0      # carga conocida al abrir (variable predictora)
    paso_por_n1: bool = False       # la etiqueta de escalamiento solo se conoce si N1 terminó
    escalo_desde_n1: bool = False   # etiqueta para la parte predictiva
    derivado_temprano: bool = False # enviado directo a N2 por el predictor


@dataclass
class Tecnico:
    id: str
    equipo: str
    turno_inicio: int
    turno_fin: int
    ticket: int = None
    inicio_actual: float = None
    fin_programado: float = None   # None con ticket asignado = se interrumpe al fin del turno
    asignaciones: list = field(default_factory=list)


def validar_config(c):
    verificar(c.dias >= 1 and c.replicas >= 1 and c.dias_calentamiento >= 0, "días o réplicas inválidos")
    verificar(len(c.tasas_horarias) == 10 and all(x >= 0 for x in c.tasas_horarias), "tasas_horarias inválidas")
    verificar(len(c.factores_dia) > 0 and all(x >= 0 for x in c.factores_dia), "factores_dia inválidos")
    verificar(len(c.proporcion_categorias) == 4 and abs(sum(c.proporcion_categorias) - 1) < 1e-7,
              "proporcion_categorias debe sumar 1")
    verificar(len(c.proporcion_complejidad) == 3 and abs(sum(c.proporcion_complejidad) - 1) < 1e-7,
              "proporcion_complejidad debe sumar 1")
    verificar(set(c.turnos) == set(EQUIPOS) and set(c.servicio_triangular) == set(EQUIPOS),
              "turnos y servicio deben cubrir todos los equipos")
    verificar(c.factor_demanda >= 0 and c.factor_servicio > 0 and c.factor_escalamiento_n1 >= 0,
              "factores inválidos")
    verificar(c.incidente_masivo >= 0 and c.minutos_envejecimiento > 0, "incidente o envejecimiento inválidos")
    verificar(c.politica_prioridad in ("estricta", "envejecimiento"), "política de prioridad desconocida")
    verificar(c.distribucion_llegadas in ("poisson", "deterministica"), "distribucion_llegadas inválida")
    verificar(c.distribucion_servicio in ("triangular", "fija"), "distribucion_servicio inválida")
    for equipo in EQUIPOS:
        a, m, b = c.servicio_triangular[equipo]
        verificar(0 < a <= m <= b, f"triangular inválida en {equipo}")
        for inicio, fin in c.turnos[equipo]:
            verificar(0 <= inicio < fin <= MINUTOS_ATENCION, f"turno fuera de horario en {equipo}")
    for p in PRIORIDADES:
        verificar(c.sla_respuesta[p] > 0 and c.sla_resolucion[p] > 0, f"SLA inválido para {p}")
    verificar(all(0 <= x <= 1 for x in c.prob_escalamiento_n1.values()), "prob_escalamiento_n1 fuera de [0, 1]")
    for nombre in ("prob_escalamiento_n2", "prob_reapertura", "prob_pausa_usuario", "prob_duplicado"):
        verificar(0 <= getattr(c, nombre) <= 1, f"{nombre} fuera de [0, 1]")
    verificar(0 <= c.max_reaperturas <= 3, "max_reaperturas debe estar entre 0 y 3")
    verificar(0 <= c.pausa_usuario_min[0] <= c.pausa_usuario_min[1], "pausa_usuario_min inválida")


def dias_totales(c):
    return c.dias_calentamiento + c.dias


def inicio_medicion(c):
    return c.dias_calentamiento * MINUTOS_DIA


def dia_semana(c, instante):
    """Día de la semana (0 = primer día medido) usado por los factores de demanda y los predictores."""
    return (int(instante // MINUTOS_DIA) - c.dias_calentamiento) % len(c.factores_dia)


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
            valor = float(rng.triangular(a, m, b)) if c.distribucion_servicio == "triangular" else float(m)
            tiempos[equipo] = max(0.1, valor * c.factor_complejidad[comp] * c.factor_servicio)
        tickets.append(Ticket(
            id=len(tickets) + 1, llegada=float(llegada), prioridad=pri,
            categoria=cat, complejidad=comp, servicio=tiempos,
            u_escala_n1=float(rng.random()), u_escala_n2=float(rng.random()),
            u_reabrir=float(rng.random()), demora_reapertura=float(rng.uniform(30, 180)),
            u_pausa=float(rng.random()),
            demora_pausa=float(rng.uniform(*c.pausa_usuario_min)),
            duplicado=bool(rng.random() < c.prob_duplicado),
            u_reabrir_extra=[float(rng.random()), float(rng.random())],
        ))

    for dia in range(dias_totales(c)):
        factor_dia = c.factores_dia[(dia - c.dias_calentamiento) % len(c.factores_dia)]
        for hora, tasa in enumerate(c.tasas_horarias):
            media = tasa * c.factor_demanda * factor_dia
            cantidad = int(rng.poisson(media)) if c.distribucion_llegadas == "poisson" else int(round(media))
            for instante in rng.uniform(0, 60, size=cantidad):
                crear(dia * MINUTOS_DIA + hora * 60 + instante)

    if c.incidente_masivo:
        dia = c.dias_calentamiento + min(2, c.dias - 1)
        for instante in rng.uniform(0, 4, size=c.incidente_masivo):
            crear(dia * MINUTOS_DIA + 30 + instante,
                  categoria="aplicaciones", prioridad="critica", complejidad="compleja")

    tickets.sort(key=lambda x: (x.llegada, x.id))
    for i, ticket in enumerate(tickets, start=1):
        ticket.id = i
    return tickets


CAMPOS_ENTRADA = ("id", "llegada", "prioridad", "categoria", "complejidad", "servicio",
                  "u_escala_n1", "u_escala_n2", "u_reabrir", "demora_reapertura",
                  "u_pausa", "demora_pausa", "duplicado", "u_reabrir_extra")
CAMPOS_REQUERIDOS = CAMPOS_ENTRADA[:10]


def ticket_a_dict(t):
    """Solo los atributos de entrada (lo que se sortea al crear el ticket), sin su estado."""
    return {k: getattr(t, k) for k in CAMPOS_ENTRADA}


def tickets_desde_dicts(filas):
    """Reconstruye un lote de tickets (por ejemplo, importado de un JSON) con el estado inicial."""
    datos = []
    for fila in filas:
        if not isinstance(fila, dict) or not set(CAMPOS_REQUERIDOS).issubset(fila):
            raise ValueError("Formato de datos de entrada incompleto")
        datos.append(Ticket(**{k: copy.deepcopy(fila[k]) for k in CAMPOS_ENTRADA if k in fila}))
    verificar(len(datos) <= 100000, "Máximo 100.000 tickets importados")
    verificar(len({t.id for t in datos}) == len(datos), "ID de ticket duplicado")
    return datos


# RELOJ Y TURNOS

def esta_de_turno(tecnico, instante):
    minuto = instante % MINUTOS_DIA
    return tecnico.turno_inicio <= minuto < tecnico.turno_fin


def cierre_del_turno(tecnico, instante):
    return int(instante // MINUTOS_DIA) * MINUTOS_DIA + tecnico.turno_fin


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


def destino_n2(ticket):
    return "N2_aplicaciones" if ticket.categoria in ("accesos", "aplicaciones") else "N2_infraestructura"


# MOTOR DE EVENTOS DISCRETOS
# Estados: en_cola -> en_atencion -> resuelto. Además: pausado_usuario (espera información
# del empleado, no consume técnicos) y cerrado_admin (duplicado, no cuenta como resolución).
# Supuesto que se mantiene: un técnico atiende un ticket por vez (atender en paralelo no
# duplica la capacidad real; con 1 no se sobreestima).

def simular(c, tickets, guardar_traza=False, derivar=None):
    """Simula una réplica. derivar: función opcional (ticket, colas) -> bool; si devuelve True
    el ticket va directo a N2 al abrirse (política de derivación temprana)."""
    total_dias = dias_totales(c)
    inicio = inicio_medicion(c)
    horizonte = (total_dias - 1) * MINUTOS_DIA + MINUTOS_ATENCION
    calendario, correlativo = [], [0]
    tecnicos = {equipo: [] for equipo in EQUIPOS}
    por_tecnico = {}
    colas = {equipo: [] for equipo in EQUIPOS}
    historial, evolucion = [], []
    estado_global = {"max_cola": 0, "ultimo": 0.0}
    areas_cola = {e: 0.0 for e in EQUIPOS}
    por_id = {t.id: t for t in tickets}

    def evento(t, tipo, identificador):
        heapq.heappush(calendario, (float(t), correlativo[0], tipo, identificador))
        correlativo[0] += 1

    def trazar(t, tipo, ticket, equipo):
        if guardar_traza:
            historial.append({"minuto": t, "dia": int(t // MINUTOS_DIA) + 1,
                              "evento": tipo, "ticket": ticket, "equipo": equipo})

    def ingresar_cola(ticket, equipo, instante, trabajo, conservar_orden=False):
        ticket.equipo = equipo
        ticket.estado = "en_cola"
        ticket.ingreso_cola = instante
        if not conservar_orden:
            ticket.orden_cola = instante
        ticket.restante = trabajo
        colas[equipo].append(ticket.id)

    def minutos_operativos(desde, hasta):
        desde = max(desde, inicio)  # solo se mide después del calentamiento
        return sum(max(0.0, min(hasta, dia * MINUTOS_DIA + MINUTOS_ATENCION)
                       - max(desde, dia * MINUTOS_DIA)) for dia in range(total_dias))

    def seleccionar(equipo, instante):
        def clave(ticket_id):
            ticket = por_id[ticket_id]
            prioridad = PRIORIDADES.index(ticket.prioridad)
            if c.politica_prioridad == "envejecimiento":
                prioridad = max(0, prioridad - int((instante - ticket.orden_cola) // c.minutos_envejecimiento))
            return (prioridad, ticket.orden_cola, ticket.id)
        elegido = min(colas[equipo], key=clave)
        colas[equipo].remove(elegido)
        return por_id[elegido]

    def liberar(tecnico, instante):
        tecnico.asignaciones.append((tecnico.inicio_actual, instante))
        trabajado = instante - tecnico.inicio_actual
        ticket = por_id[tecnico.ticket]
        ticket.restante -= trabajado
        ticket.trabajo_terminado += trabajado
        tecnico.ticket = tecnico.inicio_actual = tecnico.fin_programado = None
        return ticket

    # --- Manejadores de eventos ---
    def al_llegar(ticket, instante):
        ticket.cola_n1_al_llegar = len(colas["N1"])
        if derivar is not None and not ticket.duplicado and derivar(ticket, colas):
            destino = destino_n2(ticket)
            ticket.derivado_temprano = True
            ingresar_cola(ticket, destino, instante, ticket.servicio[destino])
            trazar(instante, "derivacion_temprana", ticket.id, destino)
        else:
            trabajo = TRABAJO_DUPLICADO if ticket.duplicado else ticket.servicio["N1"]
            ingresar_cola(ticket, "N1", instante, trabajo)
            trazar(instante, "apertura", ticket.id, "N1")

    def al_terminar(ticket, equipo, instante):
        if ticket.duplicado:
            ticket.estado = "cerrado_admin"
            ticket.cierre = instante
            trazar(instante, "cierre_duplicado", ticket.id, equipo)
            return
        if (equipo == "N1" and ticket.etapa_n1 == "inicial" and ticket.reaperturas == 0
                and ticket.u_pausa < c.prob_pausa_usuario):
            ticket.estado = "pausado_usuario"
            ticket.etapa_n1 = "tras_pausa"
            ticket.tiempo_pausa += ticket.demora_pausa
            evento(instante + ticket.demora_pausa, "reanudar", ticket.id)
            trazar(instante, "pausa_usuario", ticket.id, equipo)
            return
        if equipo == "N1":
            p = c.prob_escalamiento_n1[ticket.categoria] * c.factor_escalamiento_n1
            escalar = ticket.u_escala_n1 < min(1.0, p * c.factor_complejidad[ticket.complejidad])
            destino = destino_n2(ticket)
            if ticket.reaperturas == 0:
                ticket.paso_por_n1 = True
                ticket.escalo_desde_n1 = escalar
        elif equipo.startswith("N2"):
            escalar = ticket.u_escala_n2 < min(1.0, c.prob_escalamiento_n2 * c.factor_complejidad[ticket.complejidad])
            destino = "N3"
        else:
            escalar, destino = False, None
        if escalar:
            ticket.escaladas += 1
            ingresar_cola(ticket, destino, instante, ticket.servicio[destino])
            trazar(instante, "escalamiento", ticket.id, destino)
            return
        ticket.estado = "resuelto"
        ticket.cierre = instante
        trazar(instante, "resolucion", ticket.id, equipo)
        k = ticket.reaperturas
        u = ticket.u_reabrir if k == 0 else ticket.u_reabrir_extra[k - 1]
        if k < c.max_reaperturas and u < c.prob_reapertura:
            momento = instante + ticket.demora_reapertura
            if momento <= horizonte:
                evento(momento, "reabrir", ticket.id)

    def al_terminar_turno(tecnico, instante):
        # Trabajo que no terminó en el turno: vuelve a la cola conservando su lugar,
        # para que lo tome otro técnico de turno o el mismo al día siguiente.
        if tecnico.ticket is not None and tecnico.fin_programado is None:
            ticket = liberar(tecnico, instante)
            ticket.interrupciones += 1
            ingresar_cola(ticket, ticket.equipo, instante, ticket.restante, conservar_orden=True)
            trazar(instante, "interrupcion_por_turno", ticket.id, tecnico.id)
        trazar(instante, "fin_turno", None, tecnico.id)

    def asignar(instante):
        for equipo in EQUIPOS:
            libres = [t for t in tecnicos[equipo] if t.ticket is None and esta_de_turno(t, instante)]
            for tecnico in libres:
                if not colas[equipo]:
                    break
                ticket = seleccionar(equipo, instante)
                verificar(ticket.estado == "en_cola" and ticket.equipo == equipo, "asignación de un ticket que no estaba en cola")
                ticket.espera_acumulada += instante - ticket.ingreso_cola
                if ticket.primera_respuesta is None:
                    ticket.primera_respuesta = instante
                ticket.estado = "en_atencion"
                tecnico.ticket = ticket.id
                tecnico.inicio_actual = instante
                fin = instante + ticket.restante
                if fin <= cierre_del_turno(tecnico, instante) + 1e-9:
                    tecnico.fin_programado = fin
                    evento(fin, "fin_servicio", (equipo, tecnico.id, ticket.id))
                else:
                    tecnico.fin_programado = None  # se interrumpe al terminar el turno
                trazar(instante, "inicio_atencion", ticket.id, tecnico.id)

    # --- Estado inicial y calendario ---
    for equipo in EQUIPOS:
        for i, (ini, fin) in enumerate(c.turnos[equipo], 1):
            tecnico = Tecnico(f"{equipo}_{i}", equipo, ini, fin)
            tecnicos[equipo].append(tecnico)
            por_tecnico[tecnico.id] = tecnico
            for dia in range(total_dias):
                evento(dia * MINUTOS_DIA + ini, "inicio_turno", tecnico.id)
                evento(dia * MINUTOS_DIA + fin, "fin_turno", tecnico.id)
    for ticket in tickets:
        evento(ticket.llegada, "llegada", ticket.id)

    # --- Ciclo principal ---
    while calendario and calendario[0][0] <= horizonte + 1e-8:
        instante = calendario[0][0]
        verificar(instante >= estado_global["ultimo"] - 1e-9, "el reloj retrocedió")
        transcurrido = minutos_operativos(estado_global["ultimo"], instante)
        for equipo in EQUIPOS:
            areas_cola[equipo] += len(colas[equipo]) * transcurrido
        estado_global["ultimo"] = instante
        lote = []
        while calendario and abs(calendario[0][0] - instante) < 1e-8:
            lote.append(heapq.heappop(calendario))

        for _, _, tipo, identificador in lote:
            if tipo == "llegada":
                al_llegar(por_id[identificador], instante)
            elif tipo == "reabrir":
                ticket = por_id[identificador]
                verificar(ticket.estado == "resuelto", "se reabre un ticket no resuelto")
                ticket.reaperturas += 1
                ticket.cierre = None
                ticket.etapa_n1 = "reapertura"
                ingresar_cola(ticket, "N1", instante, ticket.servicio["N1"])
                trazar(instante, "reapertura", ticket.id, "N1")
            elif tipo == "reanudar":
                ticket = por_id[identificador]
                verificar(ticket.estado == "pausado_usuario", "se reanuda un ticket no pausado")
                ingresar_cola(ticket, "N1", instante, FRACCION_TRAS_PAUSA * ticket.servicio["N1"])
                trazar(instante, "reanudacion", ticket.id, "N1")
            elif tipo == "fin_servicio":
                equipo, id_tecnico, id_ticket = identificador
                tecnico = por_tecnico[id_tecnico]
                verificar(tecnico.ticket == id_ticket and por_id[id_ticket].estado == "en_atencion",
                          "fin de servicio de un ticket que el técnico no tenía")
                ticket = liberar(tecnico, instante)
                verificar(abs(ticket.restante) < 1e-6, "el trabajo no quedó completo")
                al_terminar(ticket, equipo, instante)
            elif tipo == "inicio_turno":
                trazar(instante, "inicio_turno", None, identificador)
            elif tipo == "fin_turno":
                al_terminar_turno(por_tecnico[identificador], instante)

        asignar(instante)

        if instante >= inicio:
            estado_global["max_cola"] = max(estado_global["max_cola"], sum(len(q) for q in colas.values()))
        if guardar_traza:
            evolucion.append({"minuto": instante, **{e: len(colas[e]) for e in EQUIPOS},
                              "pendientes": sum(len(cola) for cola in colas.values())})

    transcurrido = minutos_operativos(estado_global["ultimo"], horizonte)
    for equipo in EQUIPOS:
        areas_cola[equipo] += len(colas[equipo]) * transcurrido
    for tecnico in por_tecnico.values():  # atenciones en curso al terminar
        if tecnico.ticket is not None:
            tecnico.asignaciones.append((tecnico.inicio_actual, horizonte))

    # --- Invariantes ---
    estados = ("en_cola", "en_atencion", "resuelto", "pausado_usuario", "cerrado_admin")
    verificar(all(t.estado in estados for t in tickets), "ticket con estado inválido")
    en_cola = sum(len(q) for q in colas.values())
    en_atencion = sum(t.ticket is not None for t in por_tecnico.values())
    verificar(en_cola == sum(t.estado == "en_cola" for t in tickets), "las colas no coinciden con los estados")
    verificar(en_atencion == sum(t.estado == "en_atencion" for t in tickets), "ocupación inconsistente")
    verificar(all(t.espera_acumulada >= -1e-8 and t.restante >= -1e-6 for t in tickets), "tiempo negativo")

    return resumir(c, tickets, tecnicos, areas_cola, estado_global["max_cola"], horizonte, inicio,
                   total_dias, historial, evolucion)


# MÉTRICAS DE UNA RÉPLICA

def resumir(c, tickets, tecnicos, areas_cola, max_cola, horizonte, inicio, total_dias, historial, evolucion):
    medidos = [t for t in tickets if t.llegada >= inicio]       # se descarta el calentamiento
    validos = [t for t in medidos if not t.duplicado]           # duplicados se informan aparte
    resueltos = [t for t in validos if t.estado == "resuelto"]
    respondidos = [t for t in validos if t.primera_respuesta is not None]
    pendientes = [t for t in validos if t.estado != "resuelto"]

    def promedio(valores):
        return float(np.mean(valores)) if len(valores) else math.nan

    def percentil(valores, p):
        return float(np.percentile(valores, p)) if len(valores) else math.nan

    def resolucion(lista):
        return [t.cierre - t.llegada for t in lista]

    def resolucion_con_abiertos(lista):
        # Los abiertos aportan el tiempo que llevan abiertos: su resolución real será mayor,
        # así que el percentil resultante es una cota inferior sin sesgo por censura.
        return [(t.cierre if t.estado == "resuelto" else horizonte) - t.llegada for t in lista]

    def incumplimiento(tipo, prioridad=None):
        incumple, evaluables = 0, 0
        for t in validos:
            if prioridad is not None and t.prioridad != prioridad:
                continue
            limite = (c.sla_respuesta if tipo == "respuesta" else c.sla_resolucion)[t.prioridad]
            final = t.primera_respuesta if tipo == "respuesta" else (t.cierre if t.estado == "resuelto" else None)
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
        trabajo = sum(trabajo_en_horizonte(t, max(ini, inicio), fin, horizonte, total_dias)
                      for t in personas for ini, fin in t.asignaciones)
        verificar(trabajo <= capacidad + 1e-6, f"{equipo} trabajó más que su capacidad")
        ocupacion[equipo] = 100 * trabajo / capacidad if capacidad else math.nan

    def por_prioridad(p):
        return [t for t in validos if t.prioridad == p]

    resumen = {
        "ingresados": len(validos), "resueltos": len(resueltos),
        "pendientes": len(pendientes), "en_cola": sum(t.estado == "en_cola" for t in validos),
        "en_atencion": sum(t.estado == "en_atencion" for t in validos),
        "pausados_usuario": sum(t.estado == "pausado_usuario" for t in validos),
        "duplicados": sum(t.duplicado for t in medidos),
        "respuesta_media_min": promedio([t.primera_respuesta - t.llegada for t in respondidos]),
        "respuesta_p90_min": percentil([t.primera_respuesta - t.llegada for t in respondidos], 90),
        "respuesta_p95_min": percentil([t.primera_respuesta - t.llegada for t in respondidos], 95),
        "resolucion_media_min": promedio(resolucion(resueltos)),
        "resolucion_p90_min": percentil(resolucion(resueltos), 90),
        "resolucion_p95_min": percentil(resolucion(resueltos), 95),
        "resolucion_p95_con_abiertos_min": percentil(resolucion_con_abiertos(validos), 95),
        "resolucion_neta_p95_min": percentil([t.cierre - t.llegada - t.tiempo_pausa for t in resueltos], 95),
        "espera_media_resueltos_min": promedio([t.espera_acumulada for t in resueltos]),
        "pausa_media_resueltos_min": promedio([t.tiempo_pausa for t in resueltos]),
        "throughput_por_hora": len(resueltos) / (c.dias * 10),
        "escaladas": sum(t.escaladas for t in validos),
        "reaperturas": sum(t.reaperturas for t in validos),
        "interrupciones_por_turno": sum(t.interrupciones for t in validos),
        "incumplimiento_respuesta_pct": incumplimiento("respuesta"),
        "incumplimiento_resolucion_pct": incumplimiento("resolucion"),
        **{f"respuesta_p95_{p}_min": percentil([t.primera_respuesta - t.llegada for t in respondidos
                                                if t.prioridad == p], 95) for p in PRIORIDADES},
        **{f"resolucion_media_{p}_min": promedio(resolucion([t for t in resueltos if t.prioridad == p]))
           for p in PRIORIDADES},
        **{f"resolucion_p95_{p}_min": percentil(resolucion([t for t in resueltos if t.prioridad == p]), 95)
           for p in PRIORIDADES},
        **{f"resolucion_p95_con_abiertos_{p}_min": percentil(resolucion_con_abiertos(por_prioridad(p)), 95)
           for p in PRIORIDADES},
        **{f"incumplimiento_respuesta_{p}_pct": incumplimiento("respuesta", p) for p in PRIORIDADES},
        **{f"incumplimiento_resolucion_{p}_pct": incumplimiento("resolucion", p) for p in PRIORIDADES},
        "derivados_temprano": sum(t.derivado_temprano for t in validos),
        "cola_maxima": max_cola,
        "cola_promedio": sum(areas_cola.values()) / (c.dias * MINUTOS_ATENCION),
        **{f"cola_promedio_{e}": areas_cola[e] / (c.dias * MINUTOS_ATENCION) for e in EQUIPOS},
        "horas_dotacion": sum((t.turno_fin - t.turno_inicio) * c.dias / 60
                              for personas in tecnicos.values() for t in personas),
        **{f"utilizacion_{e}_pct": ocupacion[e] for e in EQUIPOS},
        **{f"pendientes_{e}": sum(t.equipo == e and t.estado in ("en_cola", "en_atencion") for t in pendientes)
           for e in EQUIPOS},
    }

    def relativo(x):
        return x - inicio if x is not None else None

    detalle = pd.DataFrame([{
        "ticket": t.id, "dia_apertura": int((t.llegada - inicio) // MINUTOS_DIA) + 1,
        "minuto_apertura": t.llegada - inicio, "categoria": t.categoria,
        "prioridad": t.prioridad, "complejidad": t.complejidad,
        "estado_final": t.estado, "equipo_final": t.equipo,
        "minuto_primera_respuesta": relativo(t.primera_respuesta),
        "minuto_cierre": relativo(t.cierre),
        "tiempo_respuesta_min": (t.primera_respuesta - t.llegada if t.primera_respuesta is not None else np.nan),
        "tiempo_resolucion_min": (t.cierre - t.llegada if t.estado == "resuelto" else np.nan),
        "tiempo_pausa_min": t.tiempo_pausa,
        "espera_acumulada_min": t.espera_acumulada,
        "escaladas": t.escaladas, "reaperturas": t.reaperturas, "interrupciones": t.interrupciones,
        "duplicado": t.duplicado,
        "hora_apertura": int((t.llegada % MINUTOS_DIA) // 60),
        "dia_semana": dia_semana(c, t.llegada),
        "cola_n1_al_llegar": t.cola_n1_al_llegar,
        "trabajo_N1_min": t.servicio["N1"] if t.paso_por_n1 else np.nan,
        "paso_por_n1": t.paso_por_n1, "escalo_desde_n1": t.escalo_desde_n1,
        "derivado_temprano": t.derivado_temprano,
    } for t in medidos])
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
        "Prioridad con envejecimiento": replace(c, politica_prioridad="envejecimiento"),
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
    for campo in ("respuesta_p95_min", "resolucion_p95_min", "pendientes",
                  "resolucion_p95_critica_min", "incumplimiento_resolucion_pct"):
        margen = (df.groupby("escenario", sort=False)[campo].apply(semiamplitud_ic)
                  .reindex(resumen.escenario).to_numpy())  # t de Student, no 1,96
        resumen[f"ic95_inf_{campo}"] = resumen[campo] - margen
        resumen[f"ic95_sup_{campo}"] = resumen[campo] + margen
    return df, resumen


def sensibilidad(c, replicas):
    """Un factor por vez, ±20 %, con las mismas semillas. Incluye el valor base para el tornado."""
    modificaciones = {
        "Demanda": ("factor_demanda", 0.8, 1.2),
        "Tiempos de servicio": ("factor_servicio", 0.8, 1.2),
        "Escalamiento N1": ("factor_escalamiento_n1", 0.8, 1.2),
    }

    def p95_medio(cfg):
        valores = [simular(cfg, generar_tickets(cfg, c.semilla + rep * 101))[0]["resolucion_p95_min"]
                   for rep in range(replicas)]
        return float(np.nanmean(valores))

    base = p95_medio(c)
    filas = []
    for factor, (campo, minimo, maximo) in modificaciones.items():
        filas.append({"factor": factor, "variacion": "base", "resolucion_p95_min": base})
        for etiqueta, multiplo in (("-20%", minimo), ("+20%", maximo)):
            cfg = replace(c, **{campo: getattr(c, campo) * multiplo})
            filas.append({"factor": factor, "variacion": etiqueta, "resolucion_p95_min": p95_medio(cfg)})
    return pd.DataFrame(filas)


def desglosar_tickets(detalle, agrupacion, c):
    """Métricas por grupo (categoría, prioridad, equipo final...) a partir del detalle por ticket.
    Los duplicados se excluyen: su cierre es administrativo, no una resolución."""
    if detalle.empty:
        return pd.DataFrame()
    horizonte = (c.dias - 1) * MINUTOS_DIA + MINUTOS_ATENCION  # en minutos relativos al inicio de la medición
    detalle = detalle[~detalle["duplicado"].astype(bool)]
    filas = []
    for valor, grupo in detalle.groupby(agrupacion, sort=False):
        def sla(tipo):
            limites = c.sla_respuesta if tipo == "respuesta" else c.sla_resolucion
            columna = "tiempo_respuesta_min" if tipo == "respuesta" else "tiempo_resolucion_min"
            incumple = evaluables = 0
            for _, t in grupo.iterrows():
                limite = limites[t["prioridad"]]
                if pd.notna(t[columna]):
                    evaluables += 1
                    incumple += t[columna] > limite
                elif horizonte - t["minuto_apertura"] >= limite:
                    evaluables += 1
                    incumple += 1
            return 100 * incumple / evaluables if evaluables else np.nan

        filas.append({
            agrupacion: valor,
            "ingresados": len(grupo),
            "resueltos": int((grupo.estado_final == "resuelto").sum()),
            "pendientes": int((grupo.estado_final != "resuelto").sum()),
            "respuesta_media_min": grupo.tiempo_respuesta_min.mean(),
            "respuesta_p95_min": grupo.tiempo_respuesta_min.quantile(0.95),
            "resolucion_media_min": grupo.tiempo_resolucion_min.mean(),
            "resolucion_p95_min": grupo.tiempo_resolucion_min.quantile(0.95),
            "escaladas": int(grupo.escaladas.sum()),
            "incumplimiento_respuesta_pct": sla("respuesta"),
            "incumplimiento_resolucion_pct": sla("resolucion"),
        })
    return pd.DataFrame(filas)


# PRUEBA DE HIPÓTESIS: "si quiero simular X, lo hago así"
# Uso por consola:  python simulacion_tickets_banco.py --hipotesis menos_escalamiento --magnitud 25
# El dashboard web usa exactamente estas mismas funciones.

INDICADORES = {  # en todos, un valor menor es mejor
    "resolucion_p95_min": "P95 de resolución (min)",
    "resolucion_p95_con_abiertos_min": "P95 de resolución incluyendo abiertos (min)",
    "respuesta_p95_min": "P95 de primera respuesta (min)",
    "pendientes": "Tickets pendientes al final",
    "incumplimiento_resolucion_pct": "Incumplimiento del SLA de resolución (%)",
    "cola_maxima": "Cola máxima (tickets)",
}

HIPOTESIS = {
    "menos_escalamiento": {
        "etiqueta": "−{m:g} % de escaladas desde N1",
        "titulo": "Capacitar a N1 para que escale menos",
        "magnitud": "Reducción de la probabilidad de escalar desde N1 (%)", "por_defecto": 25,
        "efecto": "mejora",
        "como": "Multiplica la probabilidad de escalar desde N1 por (1 − magnitud/100).",
    },
    "refuerzo_n2": {
        "etiqueta": "+{m:g} técnico(s)",
        "titulo": "Reforzar N2 de aplicaciones",
        "magnitud": "Técnicos adicionales en N2 de aplicaciones (1 a 5)", "por_defecto": 1,
        "efecto": "mejora",
        "como": "Agrega técnicos con turno de 0 a 480 minutos al equipo N2 de aplicaciones.",
    },
    "redistribucion": {
        "titulo": "Redistribuir los turnos de N1",
        "magnitud": None, "por_defecto": 0, "efecto": "mejora",
        "como": "Reparte N1 en mitades iguales entre el turno temprano (0–480) y el tarde (120–600), "
                "con la misma cantidad de personas.",
    },
    "servicio": {
        "etiqueta": "−{m:g} % de tiempo de atención",
        "titulo": "Reducir los tiempos de atención",
        "magnitud": "Reducción de los tiempos de servicio (%)", "por_defecto": 20, "efecto": "mejora",
        "como": "Multiplica todos los tiempos de atención por (1 − magnitud/100).",
    },
    "demanda": {
        "etiqueta": "+{m:g} % de demanda",
        "titulo": "Aumentar la demanda",
        "magnitud": "Aumento de la demanda (%)", "por_defecto": 30, "efecto": "empeora",
        "como": "Multiplica las tasas de llegada por (1 + magnitud/100). Acá la hipótesis es que el "
                "indicador empeora.",
    },
    "prioridad": {
        "titulo": "Usar prioridad con envejecimiento",
        "magnitud": None, "por_defecto": 0, "efecto": "mejora",
        "como": "Cambia la política a prioridad con envejecimiento: un ticket sube un nivel por cada "
                "240 minutos en cola.",
    },
}

EFECTOS_SECUNDARIOS = ("pendientes", "incumplimiento_resolucion_pct", "respuesta_p95_min",
                       "resolucion_p95_min", "utilizacion_N1_pct", "utilizacion_N2_aplicaciones_pct")
NOMBRES_CORTOS = {"pendientes": "Pendientes", "incumplimiento_resolucion_pct": "Incumpl. SLA resolución",
                  "respuesta_p95_min": "P95 respuesta", "resolucion_p95_min": "P95 resolución",
                  "resolucion_p95_con_abiertos_min": "P95 resol. incl. abiertos", "cola_maxima": "Cola máxima",
                  "utilizacion_N1_pct": "Utilización N1", "utilizacion_N2_aplicaciones_pct": "Utilización N2 apl."}


def aplicar_hipotesis(c, tipo, magnitud):
    verificar(tipo in HIPOTESIS, f"Hipótesis desconocida: {tipo}. Opciones: {', '.join(HIPOTESIS)}")
    verificar(0 <= magnitud <= 100, "La magnitud debe estar entre 0 y 100")
    cfg = copy.deepcopy(c)
    if tipo == "menos_escalamiento":
        cfg.factor_escalamiento_n1 *= 1 - magnitud / 100
    elif tipo == "refuerzo_n2":
        verificar(1 <= magnitud <= 5 and int(magnitud) == magnitud, "El refuerzo debe ser de 1 a 5 técnicos")
        cfg.turnos["N2_aplicaciones"] = list(cfg.turnos["N2_aplicaciones"]) + [[0, 480]] * int(magnitud)
    elif tipo == "redistribucion":
        total = len(cfg.turnos["N1"])
        cfg.turnos["N1"] = [[0, 480]] * (total // 2) + [[120, 600]] * (total - total // 2)
    elif tipo == "demanda":
        cfg.factor_demanda *= 1 + magnitud / 100
    elif tipo == "servicio":
        cfg.factor_servicio *= 1 - magnitud / 100
    elif tipo == "prioridad":
        cfg.politica_prioridad = "envejecimiento"
    validar_config(cfg)
    return cfg


def enunciado_hipotesis(tipo, magnitud, indicador):
    h = HIPOTESIS[tipo]
    detalle = f" ({h['etiqueta'].format(m=magnitud)})" if h.get("etiqueta") else ""
    verbo = "reduce" if h["efecto"] == "mejora" else "aumenta"
    return f"{h['titulo']}{detalle} {verbo} el indicador «{INDICADORES[indicador]}»"


def probar_hipotesis(c, tipo, magnitud=None, indicador="resolucion_p95_min", entrada=None):
    """Compara base y alternativa con las mismas semillas y devuelve un veredicto claro.
    entrada: lote de tickets importado (lista de dicts); si se da, se usa el mismo lote siempre."""
    verificar(indicador in INDICADORES, f"Indicador desconocido: {indicador}")
    if magnitud is None:
        magnitud = HIPOTESIS.get(tipo, {}).get("por_defecto", 0)
    validar_config(c)
    alt = aplicar_hipotesis(c, tipo, float(magnitud))
    replicas, primera = [], None
    for i in range(c.replicas):
        semilla = c.semilla + i * 101
        datos_b = tickets_desde_dicts(entrada) if entrada is not None else generar_tickets(c, semilla)
        datos_a = tickets_desde_dicts(entrada) if entrada is not None else generar_tickets(alt, semilla)
        if i == 0:
            primera = {"entrada": [ticket_a_dict(t) for t in datos_b]}
        rb, db, _, eb = simular(c, datos_b, guardar_traza=(i == 0))
        ra, _, _, _ = simular(alt, datos_a)
        if i == 0:
            primera.update({"detalle": db, "evolucion": eb})
        replicas.append({"replica": i + 1, "semilla": semilla, "base": rb, "alternativa": ra})

    def resumen_dif(clave):
        d = np.array([r["alternativa"][clave] - r["base"][clave] for r in replicas], dtype=float)
        d = d[np.isfinite(d)]
        m = float(d.mean()) if len(d) else math.nan
        hw = semiamplitud_ic(d) if entrada is None else math.nan
        return d, m, hw

    dif, m, hw = resumen_dif(indicador)
    base_vals = [r["base"][indicador] for r in replicas]
    alt_vals = [r["alternativa"][indicador] for r in replicas]
    efecto = HIPOTESIS[tipo]["efecto"]
    nombre = INDICADORES[indicador]
    if entrada is not None:
        # Con un lote fijo todas las réplicas son iguales: es una comparación puntual, sin IC.
        mejora = m < 0 if efecto == "mejora" else m > 0
        estado = "SE CUMPLE EN ESTE LOTE" if mejora else "NO SE CUMPLE EN ESTE LOTE"
        explicacion = (f"Con el lote importado, la alternativa cambia «{nombre}» en {m:+.1f}. Como siempre se "
                       "usa el mismo lote, no hay variabilidad de demanda entre réplicas y no se puede "
                       "generalizar: para validar, usar datos generados con varias réplicas.")
    elif not np.isfinite(hw):
        estado = "NO CONCLUYENTE"
        explicacion = "Con una sola réplica no se puede estimar la incertidumbre: usar 2 o más (se recomiendan 30)."
    else:
        baja, sube = m + hw < 0, m - hw > 0
        if (efecto == "mejora" and baja) or (efecto == "empeora" and sube):
            estado = "VALIDADA"
            lado = "por debajo" if baja else "por encima"
            explicacion = (f"Todo el intervalo de la diferencia está {lado} de 0: el efecto esperado aparece en "
                           f"forma consistente en las {len(dif)} réplicas pareadas.")
        elif (efecto == "mejora" and sube) or (efecto == "empeora" and baja):
            estado = "REFUTADA"
            explicacion = "Todo el intervalo de la diferencia está del lado contrario al esperado: el efecto es el opuesto."
        else:
            estado = "NO CONCLUYENTE"
            explicacion = ("El intervalo de la diferencia incluye el 0: con estas réplicas no se distingue un "
                           "efecto real del ruido. Se puede aumentar la cantidad de réplicas.")

    efectos = []
    for clave in EFECTOS_SECUNDARIOS:
        if clave == indicador:
            continue
        _, me, he = resumen_dif(clave)
        base_m = float(np.nanmean([r["base"][clave] for r in replicas]))
        if np.isfinite(he) and me - he > 0:
            lectura = "empeora"
        elif np.isfinite(he) and me + he < 0:
            lectura = "mejora"
        else:
            lectura = "sin cambio claro"
        if clave.startswith("utilizacion_"):
            lectura = "sube" if lectura == "empeora" else ("baja" if lectura == "mejora" else lectura)
        efectos.append({"indicador": clave, "nombre": NOMBRES_CORTOS.get(clave, clave), "base": base_m,
                        "diferencia": me, "semiamplitud": he,
                        "cambio_pct": 100 * me / base_m if base_m else math.nan, "lectura": lectura})
    alertas = [e["nombre"] for e in efectos if e["lectura"] == "empeora"]
    if estado == "VALIDADA" and efecto == "mejora" and alertas:
        explicacion += f" Atención: empeoran {', '.join(alertas)}; revisar si la mejora traslada el problema."

    return {
        "tipo": tipo, "magnitud": float(magnitud), "indicador": indicador, "nombre_indicador": nombre,
        "enunciado": enunciado_hipotesis(tipo, float(magnitud), indicador), "como": HIPOTESIS[tipo]["como"],
        "efecto_esperado": efecto, "replicas_usadas": len(replicas), "lote_fijo": entrada is not None,
        "base_media": float(np.nanmean(base_vals)), "base_ic": semiamplitud_ic(base_vals),
        "alternativa_media": float(np.nanmean(alt_vals)), "alternativa_ic": semiamplitud_ic(alt_vals),
        "diferencias": [float(x) for x in dif], "diferencia_media": m,
        "ic95": (m - hw, m + hw) if np.isfinite(hw) else (None, None),
        "estado": estado, "explicacion": explicacion, "efectos_secundarios": efectos,
        "replicas": replicas, "primera": primera,
        "configuracion": c, "configuracion_alternativa": alt,
    }


COLOR_ESTADO = {"VALIDADA": "#1a7f4b", "REFUTADA": "#b42318", "NO CONCLUYENTE": "#8a6d00",
                "SE CUMPLE EN ESTE LOTE": "#1a7f4b", "NO SE CUMPLE EN ESTE LOTE": "#b42318"}
AZUL, GRIS, VERDE, ROJO = "#2f6fdb", "#7b8794", "#1a7f4b", "#b42318"


def imprimir_hipotesis(res, ruta_grafico=None):
    ic = res["ic95"]
    marca = {"VALIDADA": "✔", "REFUTADA": "✖"}.get(res["estado"], "≈")
    lineas = [
        "", "=" * 78,
        f"HIPÓTESIS: {res['enunciado']}",
        f"Cómo se simula: {res['como']}",
        f"Réplicas: {res['replicas_usadas']} con semillas pareadas (misma demanda en base y alternativa)",
        "-" * 78,
        f"Base:        {res['base_media']:10.1f}" + (f"  ± {res['base_ic']:.1f}" if np.isfinite(res['base_ic']) else ""),
        f"Alternativa: {res['alternativa_media']:10.1f}" + (f"  ± {res['alternativa_ic']:.1f}" if np.isfinite(res['alternativa_ic']) else ""),
        f"Diferencia:  {res['diferencia_media']:+10.1f}" + (f"  (IC 95 %: {ic[0]:+.1f} a {ic[1]:+.1f})" if ic[0] is not None else ""),
        "",
        f"RESULTADO: {marca} HIPÓTESIS {res['estado']}",
        res["explicacion"],
        "",
        "Efectos secundarios (alternativa − base):",
    ]
    for e in res["efectos_secundarios"]:
        hw = f" ± {e['semiamplitud']:.1f}" if np.isfinite(e["semiamplitud"]) else ""
        lineas.append(f"  · {e['nombre']:<24} {e['diferencia']:+8.1f}{hw:<9} → {e['lectura']}")
    if ruta_grafico:
        lineas += ["", f"Gráfico: {ruta_grafico}"]
    lineas.append("=" * 78)
    print("\n".join(lineas))


def grafico_hipotesis(res, ruta):
    color = COLOR_ESTADO.get(res["estado"], GRIS)
    fig = plt.figure(figsize=(15, 5.6), layout="constrained")
    ejes = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1.3, 1.3]})

    # 1. Base vs alternativa con IC
    ax = ejes[0]
    medias = [res["base_media"], res["alternativa_media"]]
    errores = [res["base_ic"] if np.isfinite(res["base_ic"]) else 0,
               res["alternativa_ic"] if np.isfinite(res["alternativa_ic"]) else 0]
    barras = ax.bar(["Base", "Alternativa"], medias, yerr=errores, capsize=8, color=[GRIS, AZUL], width=0.6)
    for b, v in zip(barras, medias):
        ax.text(b.get_x() + b.get_width() / 2, v / 2, f"{v:,.1f}", ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
    ax.set_title("Promedio entre réplicas (± IC 95 %)", fontsize=11)
    ax.set_ylabel(res["nombre_indicador"])
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # 2. Diferencias pareadas por réplica
    ax = ejes[1]
    d = np.array(res["diferencias"])
    rng = np.random.default_rng(0)
    y = rng.uniform(-0.25, 0.25, len(d))
    limite = max(1e-9, np.nanmax(np.abs(d)) * 1.25) if len(d) else 1
    if res["efecto_esperado"] == "mejora":
        ax.axvspan(-limite, 0, color=VERDE, alpha=0.07)
        ax.axvspan(0, limite, color=ROJO, alpha=0.05)
        ax.text(-limite * 0.97, 0.55, "← la alternativa mejora", color=VERDE, fontsize=9, va="center")
        ax.text(limite * 0.97, 0.55, "empeora →", color=ROJO, fontsize=9, va="center", ha="right")
    else:
        ax.axvspan(0, limite, color=VERDE, alpha=0.07)
        ax.text(limite * 0.97, 0.55, "efecto esperado (aumenta) →", color=VERDE, fontsize=9, va="center", ha="right")
    ax.scatter(d, y, s=26, color=AZUL, alpha=0.6, label="Una réplica")
    ax.axvline(0, color="black", lw=1.2, ls="--")
    if res["ic95"][0] is not None:
        ax.errorbar([res["diferencia_media"]], [-0.55], xerr=[[res["diferencia_media"] - res["ic95"][0]],
                    [res["ic95"][1] - res["diferencia_media"]]], fmt="D", color=color, ms=9, capsize=7, lw=2.5,
                    label="Media e IC 95 %")
    else:
        ax.plot([res["diferencia_media"]], [-0.55], "D", color=color, ms=9, label="Diferencia")
    ax.set_xlim(-limite, limite)
    ax.set_ylim(-0.8, 0.75)
    ax.set_yticks([])
    ax.set_xlabel(f"Alternativa − base ({res['nombre_indicador']})")
    ax.set_title("¿La diferencia está claramente de un lado del 0?", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)

    # 3. Efectos secundarios en %
    ax = ejes[2]
    ef = [e for e in res["efectos_secundarios"] if np.isfinite(e["cambio_pct"])]
    nombres = [e["nombre"] for e in ef]
    valores = [e["cambio_pct"] for e in ef]
    errs = [100 * e["semiamplitud"] / e["base"] if np.isfinite(e["semiamplitud"]) and e["base"] else 0 for e in ef]
    colores = [ROJO if e["lectura"] in ("empeora",) else VERDE if e["lectura"] == "mejora" else GRIS for e in ef]
    pos = np.arange(len(ef))
    ax.barh(pos, valores, xerr=errs, color=colores, alpha=0.85, capsize=4)
    ax.set_yticks(pos, nombres)
    ax.axvline(0, color="black", lw=1)
    for p, v, e in zip(pos, valores, errs):
        x = v + e if v >= 0 else v - e
        ax.annotate(f"{v:+.0f} %", (x, p), xytext=(6 if v >= 0 else -6, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", fontsize=9, fontweight="bold")
    extremo = max([abs(v) + e for v, e in zip(valores, errs)] + [1])
    ax.set_xlim(-extremo * 1.35, extremo * 1.35)
    ax.invert_yaxis()
    ax.set_xlabel("Cambio respecto de la base (%)")
    ax.set_title("Efectos secundarios (rojo = empeora, verde = mejora)", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    marca = {"VALIDADA": "✔", "REFUTADA": "✖"}.get(res["estado"], "≈")
    fig.suptitle(f"Hipótesis: {res['enunciado']}", fontsize=13, fontweight="bold")
    fig.text(0.5, -0.04, f"{marca} HIPÓTESIS {res['estado']}  ·  {res['explicacion']}", ha="center",
             fontsize=11, color="white", wrap=True,
             bbox=dict(boxstyle="round,pad=0.6", facecolor=color, edgecolor="none"))
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


# REPORTES Y GRÁFICOS

def graficos(comparacion, evolucion, sens, destino, c=None, replicas=None):
    # 1. Comparación de escenarios con IC, coloreada según mejore o empeore a la base
    if not comparacion.empty:
        base = comparacion.loc[comparacion["escenario"] == "Base"]
        fig, axes = plt.subplots(1, 2, figsize=(15, 0.65 * len(comparacion) + 2.2), layout="constrained")
        for ax, campo, titulo in ((axes[0], "resolucion_p95_min", "P95 de resolución (min)"),
                                  (axes[1], "pendientes", "Tickets pendientes al final")):
            vals = comparacion[campo].to_numpy()
            err = (comparacion[f"ic95_sup_{campo}"] - comparacion[campo]).to_numpy() if f"ic95_sup_{campo}" in comparacion else None
            ref = float(base[campo].iloc[0]) if len(base) else np.nan
            colores = []
            for e in comparacion["escenario"]:
                if e == "Base" or replicas is None:
                    colores.append(GRIS)
                    continue
                m, hw = diferencia_pareada(replicas, e, campo)
                colores.append(VERDE if np.isfinite(hw) and m + hw < 0 else
                               ROJO if np.isfinite(hw) and m - hw > 0 else GRIS)
            pos = np.arange(len(vals))
            ax.barh(pos, vals, xerr=err, color=colores, alpha=0.85, capsize=4)
            ax.set_yticks(pos, comparacion["escenario"])
            if np.isfinite(ref):
                ax.axvline(ref, color="black", ls="--", lw=1, label="Base")
            for p, v, e in zip(pos, vals, err if err is not None else np.zeros(len(vals))):
                ax.annotate(f"{v:,.1f}", (v + (e if np.isfinite(e) else 0), p), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=9, fontweight="bold")
            ax.set_xlim(0, np.nanmax(vals + np.nan_to_num(err if err is not None else 0)) * 1.18)
            ax.invert_yaxis()
            ax.set_title(titulo, fontsize=11)
            ax.grid(axis="x", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Comparación de escenarios · promedio entre réplicas con IC 95 %\n"
                     "verde = mejora significativa frente a la base · rojo = empeora · gris = sin diferencia clara "
                     "(diferencia pareada)", fontsize=12, fontweight="bold")
        fig.savefig(destino / "comparacion_escenarios.png", dpi=150)
        plt.close(fig)

    # 2. Evolución de las colas en horas operativas, con separación por día
    if not evolucion.empty:
        fig, ax = plt.subplots(figsize=(13, 4.8), layout="constrained")
        dia = evolucion["minuto"] // MINUTOS_DIA
        horas = dia * 10 + np.minimum(evolucion["minuto"] % MINUTOS_DIA, MINUTOS_ATENCION) / 60
        ax.stackplot(horas, *[evolucion[e] for e in EQUIPOS], labels=EQUIPOS, alpha=0.85, step="post")
        for d in range(1, int(dia.max()) + 1):
            ax.axvline(d * 10, color="black", lw=0.6, alpha=0.4)
        if c is not None and c.dias_calentamiento:
            ax.axvspan(0, 10 * c.dias_calentamiento, color="grey", alpha=0.15, label="Calentamiento (no se mide)")
        ax.set(xlabel="Horas operativas acumuladas (cada línea vertical es un día)", ylabel="Tickets en cola",
               title="Evolución de las colas por equipo · una réplica del escenario base")
        ax.legend(loc="upper right", fontsize=9, ncols=3)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(destino / "evolucion_colas.png", dpi=150)
        plt.close(fig)

    # 3. Tornado de sensibilidad
    if sens is not None and not sens.empty:
        tabla = sens.pivot(index="factor", columns="variacion", values="resolucion_p95_min")
        base_v = float(tabla["base"].iloc[0])
        tabla = tabla.loc[(tabla["+20%"] - tabla["-20%"]).abs().sort_values().index]
        fig, ax = plt.subplots(figsize=(10, 1.1 * len(tabla) + 1.8), layout="constrained")
        pos = np.arange(len(tabla))
        ax.barh(pos, tabla["-20%"] - base_v, left=base_v, color="#7aa6f0", label="Parámetro −20 %")
        ax.barh(pos, tabla["+20%"] - base_v, left=base_v, color="#f08a7a", label="Parámetro +20 %")
        ax.axvline(base_v, color="black", lw=1.2)
        for p, (lo, hi) in enumerate(zip(tabla["-20%"], tabla["+20%"])):
            ax.annotate(f"{lo:,.0f}", (lo, p), xytext=(-5 if lo < base_v else 5, 0), textcoords="offset points",
                        ha="right" if lo < base_v else "left", va="center", fontsize=9)
            ax.annotate(f"{hi:,.0f}", (hi, p), xytext=(5 if hi > base_v else -5, 0), textcoords="offset points",
                        ha="left" if hi > base_v else "right", va="center", fontsize=9)
        ax.set_yticks(pos, tabla.index)
        ax.set(xlabel=f"P95 de resolución (min) · base = {base_v:,.0f}",
               title="Sensibilidad: cuánto cambia el P95 si cada supuesto varía ±20 % (arriba, el más influyente)")
        ax.legend(fontsize=9, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(destino / "sensibilidad.png", dpi=150)
        plt.close(fig)


def pruebas_verificacion():
    """Casos chicos con resultado calculado a mano."""
    vacio = {e: [] for e in EQUIPOS}
    simple = dict(dias=2, dias_calentamiento=0, prob_pausa_usuario=0, prob_duplicado=0, prob_reapertura=0,
                  prob_escalamiento_n1={k: 0 for k in CATEGORIAS}, prob_escalamiento_n2=0)
    c = Configuracion(turnos={**vacio, "N1": [[0, 480]], "N2_aplicaciones": [[0, 480]]}, **simple)

    def ticket(i, llegada, prioridad="media", escalamiento=1.0, servicio=2.0, **extra):
        return Ticket(i, llegada, prioridad, "aplicaciones", "simple",
                      {e: servicio for e in EQUIPOS}, escalamiento, 1.0, 1.0, 30, **extra)

    # Prioridad y FIFO
    r, df, log, _ = simular(c, [ticket(1, 0, "baja"), ticket(2, 0, "critica")], True)
    inicios = log.loc[log.evento == "inicio_atencion", "ticket"].tolist()
    verificar(inicios == [2, 1], f"prioridad: se esperaba [2, 1] y se obtuvo {inicios}")
    _, _, log2, _ = simular(c, [ticket(1, 0), ticket(2, 0)], True)
    verificar(log2.loc[log2.evento == "inicio_atencion", "ticket"].tolist() == [1, 2], "FIFO no respetado")
    verificar(r["resueltos"] == 2 and r["pendientes"] == 0, "conteo de resueltos incorrecto")
    verificar(df.loc[df.ticket == 1, "tiempo_respuesta_min"].iloc[0] == 2, "tiempo de respuesta incorrecto")

    # Escalamiento
    r, _, _, _ = simular(c, [ticket(1, 0, escalamiento=0)])
    verificar(r["escaladas"] == 0, "escaló con probabilidad 0")
    c2 = replace(c, prob_escalamiento_n1={k: 1 for k in CATEGORIAS})
    r, df, _, _ = simular(c2, [ticket(1, 0, escalamiento=0)])
    verificar(r["escaladas"] == 1 and r["resueltos"] == 1, "no escaló con probabilidad 1")
    verificar(abs(df["tiempo_resolucion_min"].iloc[0] - 4) < 1e-9, "resolución con escalamiento != 2 + 2")

    # Fin de turno: el trabajo pendiente pasa a otro técnico de turno...
    c3 = replace(c, turnos={**vacio, "N1": [[0, 480], [120, 600]]})
    r, df, _, _ = simular(c3, [ticket(1, 470, servicio=20)])
    verificar(abs(df["tiempo_resolucion_min"].iloc[0] - 20) < 1e-9 and r["interrupciones_por_turno"] == 1,
              "la interrupción por turno no se reasignó al otro técnico")
    # ...o, si no hay nadie, al mismo técnico al día siguiente (10 min hoy + 10 min a las 1440)
    r, df, _, _ = simular(c, [ticket(1, 470, servicio=20)])
    verificar(abs(df["minuto_cierre"].iloc[0] - 1450) < 1e-9, "el trabajo no continuó al día siguiente")

    # Pausa por el usuario: 10 min, espera 60, completa el 30 % restante (3 min) -> cierre a los 73
    c4 = replace(c, prob_pausa_usuario=1.0)
    r, df, _, _ = simular(c4, [ticket(1, 0, servicio=10, u_pausa=0.0, demora_pausa=60)])
    verificar(abs(df["tiempo_resolucion_min"].iloc[0] - 73) < 1e-9 and df["tiempo_pausa_min"].iloc[0] == 60,
              "pausa por usuario mal calculada")

    # Duplicado: cierre administrativo, no cuenta como resolución
    r, df, _, _ = simular(c, [ticket(1, 0, duplicado=True)])
    verificar(r["resueltos"] == 0 and r["duplicados"] == 1 and df["estado_final"].iloc[0] == "cerrado_admin",
              "duplicado mal tratado")

    # Calentamiento: el ticket del día 0 no se mide
    c5 = replace(c, dias=1, dias_calentamiento=1)
    r, _, _, _ = simular(c5, [ticket(1, 10), ticket(2, MINUTOS_DIA + 10)])
    verificar(r["ingresados"] == 1, "el calentamiento no se descartó")

    # Valores extremos
    r, _, _, _ = simular(CONFIG, [])
    verificar(r["resueltos"] == 0 and all(r[f"utilizacion_{e}_pct"] == 0 for e in EQUIPOS),
              "sin tickets no debería haber resoluciones")
    mucha = replace(CONFIG, turnos={e: [[0, 600]] * 60 for e in EQUIPOS})
    r, _, _, _ = simular(mucha, generar_tickets(mucha, 1))
    verificar(r["respuesta_p95_min"] < 1e-9, "con capacidad muy alta la espera debería ser 0")
    sin_n2 = replace(CONFIG, prob_escalamiento_n1={k: 1 for k in CATEGORIAS}, prob_pausa_usuario=0,
                     factor_complejidad={"simple": 1.0, "media": 1.0, "compleja": 1.0}, prob_duplicado=0,
                     turnos={**CONFIG.turnos, "N2_aplicaciones": [], "N2_infraestructura": []})
    r, _, _, _ = simular(sin_n2, generar_tickets(sin_n2, 1))
    verificar(r["resueltos"] == 0, "sin especialistas no debería resolverse ningún ticket escalado")
    print("Verificación: prioridad, FIFO, escalamiento, turnos, pausas, duplicados, calentamiento "
          "y valores extremos: OK")


# BARRIDO DE DEMANDA (umbral de saturación)

def barrido_demanda(c, replicas, multiplos=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5)):
    filas = []
    for m in multiplos:
        cfg = replace(c, factor_demanda=c.factor_demanda * m)
        for rep in range(replicas):
            r, _, _, _ = simular(cfg, generar_tickets(cfg, c.semilla + rep * 101))
            filas.append({"factor_demanda": m, "replica": rep + 1, **r})
    df = pd.DataFrame(filas)
    return df.groupby("factor_demanda")[["ingresados", "pendientes", "resolucion_p95_min",
                                         "resolucion_p95_critica_min",
                                         "incumplimiento_resolucion_pct"]].mean().reset_index()


# CALIBRACIÓN Y VALIDACIÓN
# Con datos reales: --historico archivo.csv, con una fila por ticket y las columnas
#   minuto_apertura (minutos corridos desde el día 0, atención entre 0 y 600 de cada día),
#   categoria, complejidad, prioridad, escalo_desde_n1 (0/1),
#   tiempo_respuesta_min, tiempo_resolucion_min (vacíos si el ticket sigue abierto)
#   y opcionalmente paso_por_n1 (0/1) y trabajo_N1_min (trabajo efectivo, sin pausas).
# Sin datos reales se usa una "mesa de referencia" sintética con parámetros distintos
# a los supuestos. Eso solo prueba que el procedimiento funciona: NO valida el modelo.

OBLIGATORIAS_HISTORICO = ["minuto_apertura", "categoria", "complejidad", "prioridad",
                          "escalo_desde_n1", "tiempo_respuesta_min", "tiempo_resolucion_min"]


def referencia_sintetica(c):
    return replace(
        c,
        tasas_horarias=[x * f for x, f in zip(c.tasas_horarias,
                                              [1.15, 1.2, 1.1, 1.0, 0.95, 1.0, 1.05, 1.0, 0.9, 0.9])],
        proporcion_categorias=[0.28, 0.42, 0.18, 0.12],
        prob_escalamiento_n1={"accesos": 0.10, "aplicaciones": 0.50,
                              "conectividad": 0.30, "hardware": 0.22},
        servicio_triangular={**c.servicio_triangular, "N1": [7, 14, 28]},
    )


def preparar_historico(df):
    faltan = [x for x in OBLIGATORIAS_HISTORICO if x not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el histórico: {faltan}")
    df = df.copy()
    df["dia_apertura"] = (df["minuto_apertura"] // MINUTOS_DIA).astype(int) + 1
    df["escalo_desde_n1"] = df["escalo_desde_n1"].astype(bool)
    df["paso_por_n1"] = df["paso_por_n1"].astype(bool) if "paso_por_n1" in df else True
    if "trabajo_N1_min" not in df:
        df["trabajo_N1_min"] = np.nan
    if "duplicado" in df:
        df = df[~df["duplicado"].astype(bool)]  # los duplicados no son demanda real de soporte
    df["hora_apertura"] = ((df["minuto_apertura"] % MINUTOS_DIA) // 60).astype(int).clip(0, 9)
    df["dia_semana"] = (df["dia_apertura"] - 1) % 5
    df["tiene_cola"] = "cola_n1_al_llegar" in df
    if "cola_n1_al_llegar" not in df:
        df["cola_n1_al_llegar"] = 0  # sin dato de carga: la variable queda neutra
    return df


def calibrar(c, hist):
    """Estima parámetros de entrada a partir del período de calibración."""
    hora = ((hist["minuto_apertura"] % MINUTOS_DIA) // 60).astype(int).clip(0, 9)
    conteo = (hist.groupby([hist["dia_apertura"], hora]).size().unstack(fill_value=0)
              .reindex(columns=range(10), fill_value=0))
    totales = conteo.sum(axis=1)
    dia_sem = (conteo.index.to_numpy() - 1) % 5
    factores = []
    for k in range(5):
        sel = totales[dia_sem == k]
        factores.append(float(sel.mean() / totales.mean()) if len(sel) else 1.0)
    ajustado = conteo.div([factores[d] for d in dia_sem], axis=0)
    tasas = [float(x) for x in ajustado.mean()]

    categorias = hist["categoria"].value_counts(normalize=True).reindex(CATEGORIAS, fill_value=0)
    complejidad = (hist["complejidad"].value_counts(normalize=True)
                   .reindex(["simple", "media", "compleja"], fill_value=0))

    # Escalamiento: tasa observada / factor medio de complejidad de esa categoría
    paso = hist[hist["paso_por_n1"]]
    prob = {}
    for k in CATEGORIAS:
        sub = paso[paso["categoria"] == k]
        if len(sub) < 20:
            prob[k] = c.prob_escalamiento_n1[k]  # pocos datos: se conserva el supuesto
            continue
        f_medio = sub["complejidad"].map(c.factor_complejidad).mean()
        prob[k] = float(min(1.0, sub["escalo_desde_n1"].mean() / f_medio))

    # Servicio N1: triangular ajustada al trabajo efectivo (sin el efecto de la complejidad)
    servicio = {e: list(v) for e, v in c.servicio_triangular.items()}
    trabajo = (hist["trabajo_N1_min"] / hist["complejidad"].map(c.factor_complejidad)).dropna()
    if len(trabajo) > 30:
        a, b = np.percentile(trabajo, 1), np.percentile(trabajo, 99)
        moda = float(np.clip(3 * trabajo.mean() - a - b, a, b))  # media triangular = (a+m+b)/3
        servicio["N1"] = [round(float(a), 1), round(moda, 1), round(float(b), 1)]

    return replace(c, tasas_horarias=tasas, factores_dia=factores,
                   proporcion_categorias=[float(x) for x in categorias / categorias.sum()],
                   proporcion_complejidad=[float(x) for x in complejidad / complejidad.sum()],
                   prob_escalamiento_n1=prob, servicio_triangular=servicio)


def metricas_operativas(det):
    paso = det[det["paso_por_n1"].astype(bool)]
    return {
        "tickets_por_dia": len(det) / max(1, det["dia_apertura"].nunique()),
        "pct_escalado_desde_N1": 100 * paso["escalo_desde_n1"].astype(bool).mean(),
        "respuesta_p50_min": det["tiempo_respuesta_min"].median(),
        "respuesta_p90_min": det["tiempo_respuesta_min"].quantile(0.9),
        "resolucion_p50_min": det["tiempo_resolucion_min"].median(),
        "resolucion_p90_min": det["tiempo_resolucion_min"].quantile(0.9),
    }


def banda_simulada(cfg, replicas, semilla):
    filas = []
    for rep in range(replicas):
        _, det, _, _ = simular(cfg, generar_tickets(cfg, semilla + rep * 101))
        det = det.rename(columns={"dia_apertura": "dia_apertura"})
        filas.append(metricas_operativas(det))
    return pd.DataFrame(filas)


def calibracion_y_validacion(c, ruta_historico, replicas):
    if ruta_historico:
        hist = preparar_historico(pd.read_csv(ruta_historico))
        origen = f"histórico real ({ruta_historico})"
        referencia = None
    else:
        referencia = replace(referencia_sintetica(c), dias=20)
        _, hist, _, _ = simular(referencia, generar_tickets(referencia, 777))
        hist = preparar_historico(hist)
        origen = "referencia SINTÉTICA (solo prueba el procedimiento; no es validación real)"

    dias = sorted(hist["dia_apertura"].unique())
    corte = dias[int(len(dias) * 0.75) - 1]  # 75 % calibración, 25 % reservado
    cal = hist[hist["dia_apertura"] <= corte]
    reservado = hist[hist["dia_apertura"] > corte]
    c_cal = calibrar(c, cal)
    dias_val = int(reservado["dia_apertura"].nunique())

    observado = metricas_operativas(reservado)
    sim_cal = banda_simulada(replace(c_cal, dias=dias_val), replicas, c.semilla)
    sim_sup = banda_simulada(replace(c, dias=dias_val), replicas, c.semilla)
    filas = []
    for k, v in observado.items():
        fila = {"metrica": k, "observado": v}
        for nombre, sim in (("calibrado", sim_cal), ("supuesto", sim_sup)):
            lo, hi = sim[k].quantile(0.025), sim[k].quantile(0.975)
            fila[f"{nombre}_media"] = sim[k].mean()
            fila[f"{nombre}_banda95"] = f"[{lo:.1f}; {hi:.1f}]"
            fila[f"{nombre}_consistente"] = bool(lo <= v <= hi)
        filas.append(fila)
    validacion = pd.DataFrame(filas)

    parametros = [
        ("Tickets esperados por día", sum(c.tasas_horarias), sum(c_cal.tasas_horarias),
         sum(referencia.tasas_horarias) if referencia else np.nan),
        ("Proporción aplicaciones", c.proporcion_categorias[1], c_cal.proporcion_categorias[1],
         referencia.proporcion_categorias[1] if referencia else np.nan),
    ]
    for k in CATEGORIAS:
        parametros.append((f"Prob. escalar N1 · {k}", c.prob_escalamiento_n1[k],
                           c_cal.prob_escalamiento_n1[k],
                           referencia.prob_escalamiento_n1[k] if referencia else np.nan))
    parametros.append(("Servicio N1 (moda, min)", c.servicio_triangular["N1"][1],
                       c_cal.servicio_triangular["N1"][1],
                       referencia.servicio_triangular["N1"][1] if referencia else np.nan))
    recuperacion = pd.DataFrame(parametros, columns=["parametro", "supuesto", "calibrado", "referencia_real"])
    return {"origen": origen, "sintetico": referencia is not None, "dias_calibracion": len(dias) - dias_val,
            "dias_validacion": dias_val, "validacion": validacion, "parametros": recuperacion,
            "config_calibrada": c_cal}


# PARTE PREDICTIVA
# Solo se usan variables conocidas al abrir el ticket. La complejidad NO se
# usa: en la realidad se conoce recién después del diagnóstico (evita fuga de información).

def matriz_x(categoria, prioridad, hora, dia, cola, stats):
    categoria, prioridad = np.asarray(categoria), np.asarray(prioridad)
    hora, dia = np.asarray(hora), np.asarray(dia)
    cola = np.asarray(cola, dtype=float)
    columnas = [(categoria == k) for k in CATEGORIAS] + [(prioridad == p) for p in PRIORIDADES]
    columnas += [(hora == h) for h in range(10)] + [(dia == d) for d in range(5)]
    columnas.append((cola - stats[0]) / stats[1])  # estandarizada con datos de entrenamiento
    return np.column_stack(columnas).astype(float)


def matriz_desde_df(df, stats):
    return matriz_x(df["categoria"], df["prioridad"], df["hora_apertura"], df["dia_semana"],
                    df["cola_n1_al_llegar"], stats)


def sigmoide(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def entrenar_logistica(X, y, l2=1e-3, lr=0.5, epocas=3000):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epocas):
        g = sigmoide(X @ w + b) - y
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return lambda Xn: sigmoide(Xn @ w + b)


def entrenar_red(Xtr, ytr, Xva, yva, capas=(32, 16), lr=1e-3, lote=64,
                 max_epocas=300, paciencia=20, l2=1e-4, semilla=2026):
    """Red feedforward: ReLU en capas ocultas, sigmoide en la salida,
    entropía cruzada binaria, Adam en minibatches y early stopping."""
    rng = np.random.default_rng(semilla)
    tam = [Xtr.shape[1], *capas, 1]
    W = [rng.normal(0, np.sqrt(2 / tam[i]), (tam[i], tam[i + 1])) for i in range(len(tam) - 1)]
    B = [np.zeros(tam[i + 1]) for i in range(len(tam) - 1)]
    m1 = [np.zeros_like(p) for p in W + B]
    m2 = [np.zeros_like(p) for p in W + B]

    def adelante(X, W, B):
        acts, h = [X], X
        for i in range(len(W) - 1):
            h = np.maximum(0, h @ W[i] + B[i])
            acts.append(h)
        return acts, sigmoide(h @ W[-1] + B[-1]).ravel()

    def perdida(X, y):
        p = np.clip(adelante(X, W, B)[1], 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    mejor, guardado, sin_mejora, paso, historia = np.inf, None, 0, 0, []
    for epoca in range(max_epocas):
        orden = rng.permutation(len(ytr))
        for s in range(0, len(orden), lote):
            j = orden[s:s + lote]
            acts, p = adelante(Xtr[j], W, B)
            delta = ((p - ytr[j]) / len(j))[:, None]
            gW, gB = [None] * len(W), [None] * len(B)
            for i in reversed(range(len(W))):
                gW[i] = acts[i].T @ delta + l2 * W[i]
                gB[i] = delta.sum(axis=0)
                if i > 0:
                    delta = (delta @ W[i].T) * (acts[i] > 0)
            paso += 1
            for k, (P, G) in enumerate(zip(W + B, gW + gB)):
                m1[k] = 0.9 * m1[k] + 0.1 * G
                m2[k] = 0.999 * m2[k] + 0.001 * G * G
                P -= lr * (m1[k] / (1 - 0.9 ** paso)) / (np.sqrt(m2[k] / (1 - 0.999 ** paso)) + 1e-8)
        ltr, lva = perdida(Xtr, ytr), perdida(Xva, yva)
        historia.append({"epoca": epoca + 1, "perdida_entrenamiento": ltr, "perdida_validacion": lva})
        if lva < mejor - 1e-5:
            mejor, sin_mejora = lva, 0
            guardado = ([w.copy() for w in W], [b.copy() for b in B])
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    Wf, Bf = guardado
    return (lambda Xn: adelante(Xn, Wf, Bf)[1]), pd.DataFrame(historia)


def auc(y, p):
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metricas_clasificacion(y, p, umbral):
    pred = p >= umbral
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    fn, tn = int(np.sum(~pred & (y == 1))), int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {"umbral": umbral, "AUC": auc(y, p), "precision": precision, "recall": recall, "F1": f1,
            "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            "VP": tp, "FP": fp, "FN": fn, "VN": tn}


def umbral_f1(y, p):
    grilla = np.arange(0.05, 0.96, 0.01)
    return float(max(grilla, key=lambda u: metricas_clasificacion(y, p, u)["F1"]))


def prediccion_demanda(det, dias_entrenamiento):
    """Demanda por hora: tres modelos simples evaluados en días posteriores."""
    dias = int(det["dia_apertura"].max())
    conteo = (det.groupby(["dia_apertura", "hora_apertura"]).size().unstack(fill_value=0)
              .reindex(index=range(1, dias + 1), columns=range(10), fill_value=0))
    dia_sem = (conteo.index.to_numpy() - 1) % 5
    entrenamiento = conteo.loc[:dias_entrenamiento]
    por_hora = entrenamiento.mean()
    por_hora_dia = entrenamiento.groupby(dia_sem[:dias_entrenamiento]).mean()
    errores = {"Promedio por hora": [], "Promedio por hora y día de semana": [],
               "Misma hora de la semana anterior": []}
    for d in range(dias_entrenamiento + 1, dias + 1):
        real = conteo.loc[d]
        errores["Promedio por hora"].append(np.abs(real - por_hora).mean())
        errores["Promedio por hora y día de semana"].append(np.abs(real - por_hora_dia.loc[(d - 1) % 5]).mean())
        errores["Misma hora de la semana anterior"].append(np.abs(real - conteo.loc[d - 5]).mean())
    return pd.DataFrame([{"modelo": k, "MAE_tickets_por_hora": float(np.mean(v))} for k, v in errores.items()]
                        + [{"modelo": "(referencia) tickets promedio por hora",
                            "MAE_tickets_por_hora": float(conteo.loc[dias_entrenamiento + 1:].to_numpy().mean())}])


def parte_predictiva(c, replicas, historico=None, dias_historia=40):
    # Con --historico se entrena con datos reales; si no, con un historial simulado del escenario base.
    if historico is not None:
        det = historico.copy()
        dias_historia = int(det["dia_apertura"].max())
        origen = "histórico real"
    else:
        cfg_hist = replace(c, dias=dias_historia)
        _, det, _, _ = simular(cfg_hist, generar_tickets(cfg_hist, 4242))
        det = det[~det["duplicado"]]
        origen = "historial simulado con el escenario base"
    n_tr, n_va = int(dias_historia * 0.6), int(dias_historia * 0.8)

    demanda = prediccion_demanda(det, n_va)

    # Escalamiento: solo tickets con etiqueta conocida (terminaron su paso por N1)
    datos = det[det["paso_por_n1"]].copy()
    y_all = datos["escalo_desde_n1"].astype(float).to_numpy()
    tr = (datos["dia_apertura"] <= n_tr).to_numpy()
    va = ((datos["dia_apertura"] > n_tr) & (datos["dia_apertura"] <= n_va)).to_numpy()
    te = (datos["dia_apertura"] > n_va).to_numpy()
    cola = datos.loc[tr, "cola_n1_al_llegar"].astype(float)
    stats = (float(cola.mean()), float(cola.std()) if cola.std() > 0 else 1.0)
    X = matriz_desde_df(datos, stats)

    # Baseline 1: tasa histórica por categoría y prioridad
    tasas = datos[tr].groupby(["categoria", "prioridad"])["escalo_desde_n1"].mean()
    global_tr = float(y_all[tr].mean())
    def tasa_historica(cat, pri):
        return float(tasas.get((cat, pri), global_tr))
    p_base = np.array([tasa_historica(a, b) for a, b in zip(datos["categoria"], datos["prioridad"])])

    logistica = entrenar_logistica(X[tr], y_all[tr])
    red, curva = entrenar_red(X[tr], y_all[tr], X[va], y_all[va])

    modelos = {"Tasa histórica (baseline)": p_base, "Regresión logística": logistica(X), "Red neuronal 32-16": red(X)}
    filas, umbrales = [], {}
    for nombre, p in modelos.items():
        u = umbral_f1(y_all[va], p[va])  # el umbral se elige con validación, no con prueba
        umbrales[nombre] = u
        filas.append({"modelo": nombre, **metricas_clasificacion(y_all[te], p[te], u)})
    resultados = pd.DataFrame(filas)

    # Acople al simulador: derivación temprana a N2
    auc_va = {k: auc(y_all[va], v[va]) for k, v in modelos.items() if k != "Tasa histórica (baseline)"}
    elegido = max(auc_va, key=auc_va.get)
    funcion = logistica if elegido == "Regresión logística" else red
    # Umbral de derivación: el mismo elegido con validación (máximo F1)
    umbral_derivacion = umbrales[elegido]

    def con_predictor(t, colas):
        x = matriz_x([t.categoria], [t.prioridad], [int((t.llegada % MINUTOS_DIA) // 60)],
                     [dia_semana(c, t.llegada)], [t.cola_n1_al_llegar], stats)
        return float(funcion(x)[0]) >= umbral_derivacion

    def oraculo(t, colas):
        p = c.prob_escalamiento_n1[t.categoria] * c.factor_escalamiento_n1 * c.factor_complejidad[t.complejidad]
        return t.u_escala_n1 < min(1.0, p)

    if historico is not None:
        # El acople necesita el simulador calibrado con esos mismos datos: se omite aquí.
        return {"demanda": demanda, "clasificacion": resultados, "curva": curva, "elegido": elegido,
                "umbral_derivacion": umbral_derivacion, "acople": None, "acople_replicas": None,
                "n": {"entrenamiento": int(tr.sum()), "validacion": int(va.sum()), "prueba": int(te.sum())},
                "tasa_escalamiento": float(y_all.mean()), "dias": (n_tr, n_va, dias_historia), "origen": origen}

    filas = []
    for nombre, regla in (("Sin predictor (base)", None),
                          (f"Derivación con {elegido.lower()}", con_predictor),
                          ("Oráculo (predicción perfecta)", oraculo)):
        for rep in range(replicas):
            r, _, _, _ = simular(c, generar_tickets(c, c.semilla + rep * 101), derivar=regla)
            filas.append({"politica": nombre, "replica": rep + 1, **r})
    acople_rep = pd.DataFrame(filas)
    columnas = ["derivados_temprano", "respuesta_p95_min", "resolucion_p95_min", "resolucion_p95_critica_min",
                "pendientes", "utilizacion_N1_pct", "utilizacion_N2_aplicaciones_pct",
                "utilizacion_N2_infraestructura_pct", "cola_promedio_N2_aplicaciones"]
    acople = acople_rep.groupby("politica", sort=False)[columnas].mean().reset_index()

    return {"demanda": demanda, "clasificacion": resultados, "curva": curva, "elegido": elegido,
            "umbral_derivacion": umbral_derivacion, "acople": acople, "acople_replicas": acople_rep,
            "n": {"entrenamiento": int(tr.sum()), "validacion": int(va.sum()), "prueba": int(te.sum())},
            "tasa_escalamiento": float(y_all.mean()), "dias": (n_tr, n_va, dias_historia), "origen": origen}


# INFORME ORGANIZADO SEGÚN LAS PREGUNTAS DEL INFORME TEÓRICO

def diferencia_pareada(df, escenario, campo, base="Base"):
    a = df[df["escenario"] == base].set_index("replica")[campo]
    b = df[df["escenario"] == escenario].set_index("replica")[campo]
    d = (b - a).dropna()
    if len(d) < 2:
        return np.nan, np.nan
    return float(d.mean()), semiamplitud_ic(d)


def veredicto(m, hw):
    if not np.isfinite(hw):
        return "sin réplicas suficientes"
    if m + hw < 0:
        return "mejora significativa"
    if m - hw > 0:
        return "empeora significativamente"
    return "diferencia no concluyente"


def tabla_md(df, decimales=1):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda x: "–" if pd.isna(x) else f"{x:,.{decimales}f}")
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    filas = ["| " + " | ".join(map(str, f)) + " |" for f in df.to_numpy()]
    return "\n".join([cab, sep] + filas)


def armar_informe(c, comparacion, replicas_df, sens, barrido, val, pred):
    L = []
    def h(t): L.extend(["", "## " + t, ""])
    def p(t): L.append(t)
    def esc(nombre, campo):
        fila = comparacion[comparacion["escenario"] == nombre]
        return float(fila[campo].iloc[0]) if len(fila) else np.nan
    hay = set(comparacion["escenario"])

    L += ["# Resultados de la simulación · Mesa de ayuda bancaria",
          "",
          f"Configuración: {c.dias} días medidos tras {c.dias_calentamiento} de calentamiento, {c.replicas} "
          f"réplicas por escenario, semilla {c.semilla}. Tiempos en minutos corridos. IC 95 % con t de Student. "
          "Todos los parámetros son supuestos editables, no datos reales.",
          "",
          f"Supuestos de operación: un técnico atiende un ticket a la vez; el trabajo no terminado al final del "
          f"turno vuelve a la cola sin perder su lugar; {100 * c.prob_pausa_usuario:.0f} % de los tickets espera "
          f"información del empleado; {100 * c.prob_duplicado:.0f} % son duplicados (cierre administrativo, no "
          f"cuentan como resueltos); hasta {c.max_reaperturas} reaperturas por ticket."]

    # P1 · pregunta principal
    h("1. ¿Qué combinación de dotación, prioridad y escalamiento reduce mejor los tiempos sin aumentar el backlog?")
    cols = ["escenario", "horas_dotacion", "respuesta_p95_min", "resolucion_p95_min", "resolucion_p95_critica_min",
            "incumplimiento_resolucion_pct", "pendientes"]
    p(tabla_md(comparacion[cols].rename(columns={
        "horas_dotacion": "horas dotación", "respuesta_p95_min": "P95 respuesta",
        "resolucion_p95_min": "P95 resolución", "resolucion_p95_critica_min": "P95 resol. crítica",
        "incumplimiento_resolucion_pct": "% incumpl. SLA resol.", "pendientes": "pendientes"})))
    alternativas = [e for e in comparacion["escenario"] if e not in ("Base", "Estrés +30% e incidente")]
    if "Base" in hay and alternativas:
        p("")
        p("Diferencia pareada contra la base (mismas semillas, IC 95 %):")
        p("")
        mejores = []
        for e in alternativas:
            m1, h1 = diferencia_pareada(replicas_df, e, "resolucion_p95_min")
            m2, h2 = diferencia_pareada(replicas_df, e, "resolucion_p95_critica_min")
            m3, h3 = diferencia_pareada(replicas_df, e, "pendientes")
            p(f"- **{e}**: P95 resolución {m1:+.1f} ± {h1:.1f} min ({veredicto(m1, h1)}); "
              f"P95 críticos {m2:+.1f} ± {h2:.1f} ({veredicto(m2, h2)}); pendientes {m3:+.1f} ± {h3:.1f}.")
            if m1 + h1 < 0 and m3 - h3 <= 0:
                mejores.append((m1, e))
        p("")
        if mejores:
            mejores.sort()
            p(f"**Respuesta:** con estos supuestos, la alternativa que más reduce el P95 de resolución sin aumentar "
              f"el backlog es **{mejores[0][1]}**"
              + (f", seguida de {', '.join(e for _, e in mejores[1:])}" if len(mejores) > 1 else "")
              + ". Las horas de dotación sirven como aproximación del costo.")
        else:
            p("**Respuesta:** ninguna alternativa mejora de forma significativa el P95 sin aumentar el backlog.")

    # P2 · hipótesis
    if {"Base", "Turnos redistribuidos", "Menos escalamiento N1"} <= hay:
        h("2. ¿Redistribuir turnos y resolver más en N1 reduce colas y escaladas? ¿Se traslada el cuello de botella a N2?")
        filas = []
        for e in ("Base", "Turnos redistribuidos", "Menos escalamiento N1", "Refuerzo especialistas"):
            if e in hay:
                filas.append({"escenario": e, "cola prom. N1": esc(e, "cola_promedio_N1"),
                              "cola prom. N2 apl.": esc(e, "cola_promedio_N2_aplicaciones"),
                              "util. N1 %": esc(e, "utilizacion_N1_pct"),
                              "util. N2 apl. %": esc(e, "utilizacion_N2_aplicaciones_pct"),
                              "escaladas": esc(e, "escaladas")})
        p(tabla_md(pd.DataFrame(filas)))
        d_n2 = esc("Turnos redistribuidos", "cola_promedio_N2_aplicaciones") - esc("Base", "cola_promedio_N2_aplicaciones")
        p("")
        p(f"**Respuesta:** redistribuir turnos cambia la cola de N1 de {esc('Base', 'cola_promedio_N1'):.1f} a "
          f"{esc('Turnos redistribuidos', 'cola_promedio_N1'):.1f} tickets en promedio, y la cola de N2 de aplicaciones "
          f"{'aumenta' if d_n2 > 0 else 'no aumenta'} ({d_n2:+.1f}). Resolver más en N1 baja las escaladas de "
          f"{esc('Base', 'escaladas'):.0f} a {esc('Menos escalamiento N1', 'escaladas'):.0f} y la cola de N2 de "
          f"aplicaciones a {esc('Menos escalamiento N1', 'cola_promedio_N2_aplicaciones'):.1f}. "
          + ("La hipótesis de traslado del cuello de botella se confirma para la redistribución."
             if d_n2 > 0 else "En este caso no se observa traslado del cuello de botella hacia N2."))

    # P3 · cuello de botella
    if "Base" in hay:
        h("3. ¿Dónde está el cuello de botella?")
        filas = [{"equipo": e, "utilización %": esc("Base", f"utilizacion_{e}_pct"),
                  "cola promedio": esc("Base", f"cola_promedio_{e}"),
                  "pendientes al final": esc("Base", f"pendientes_{e}")} for e in EQUIPOS]
        df = pd.DataFrame(filas)
        p(tabla_md(df))
        top_u = df.loc[df["utilización %"].idxmax(), "equipo"]
        top_c = df.loc[df["cola promedio"].idxmax(), "equipo"]
        p("")
        p(f"**Respuesta:** en el escenario base el equipo más utilizado es **{top_u}** y la cola promedio más larga "
          f"está en **{top_c}**. "
          + ("Coinciden, por lo que ese equipo es el cuello de botella." if top_u == top_c else
             "No coinciden: una cola larga con utilización moderada indica un problema de turnos o de capacidad en "
             "ciertos horarios más que de dotación total."))

    # P4 · promedio vs percentiles
    if "Base" in hay:
        h("4. ¿El promedio oculta demoras importantes? (tiempos por prioridad, escenario base)")
        filas = [{"prioridad": pr, "resolución media": esc("Base", f"resolucion_media_{pr}_min"),
                  "resolución P95": esc("Base", f"resolucion_p95_{pr}_min"),
                  "P95 incl. abiertos": esc("Base", f"resolucion_p95_con_abiertos_{pr}_min"),
                  "respuesta P95": esc("Base", f"respuesta_p95_{pr}_min"),
                  "% incumpl. respuesta": esc("Base", f"incumplimiento_respuesta_{pr}_pct"),
                  "% incumpl. resolución": esc("Base", f"incumplimiento_resolucion_{pr}_pct")} for pr in PRIORIDADES]
        df = pd.DataFrame(filas)
        p(tabla_md(df))
        ratio = (df["resolución P95"] / df["resolución media"]).max()
        p("")
        p(f"**Respuesta:** sí. En la prioridad más dispersa, el P95 de resolución es {ratio:.1f} veces la media: "
          "un promedio global no muestra los casos demorados. Por eso las decisiones se apoyan en P95 e "
          "incumplimiento de SLA por prioridad.")
        p("")
        p(f"El P95 que incluye los tickets todavía abiertos (con el tiempo que llevan abiertos) es una cota "
          f"inferior sin sesgo por censura: si es mayor que el P95 de los resueltos, los casos más demorados "
          f"quedaron fuera del horizonte. Las pausas por espera del empleado suman en promedio "
          f"{esc('Base', 'pausa_media_resueltos_min'):.0f} min por ticket resuelto; sin ellas, el P95 de "
          f"resolución sería {esc('Base', 'resolucion_neta_p95_min'):.0f} min (no son demoras atribuibles al personal).")

    # P5 · política de prioridad
    if {"Base", "Prioridad con envejecimiento"} <= hay:
        h("5. ¿Conviene la prioridad con envejecimiento para no postergar a los tickets de baja prioridad?")
        filas = [{"política": e, "P95 resol. crítica": esc(e, "resolucion_p95_critica_min"),
                  "P95 resol. baja": esc(e, "resolucion_p95_baja_min"),
                  "% incumpl. resol. crítica": esc(e, "incumplimiento_resolucion_critica_pct"),
                  "% incumpl. resol. baja": esc(e, "incumplimiento_resolucion_baja_pct")}
                 for e in ("Base", "Prioridad con envejecimiento")]
        p(tabla_md(pd.DataFrame(filas)))
        m_b, h_b = diferencia_pareada(replicas_df, "Prioridad con envejecimiento", "resolucion_p95_baja_min")
        m_c, h_c = diferencia_pareada(replicas_df, "Prioridad con envejecimiento", "resolucion_p95_critica_min")
        p("")
        p(f"**Respuesta:** con envejecimiento ({c.minutos_envejecimiento} min por nivel) el P95 de la prioridad baja "
          f"cambia {m_b:+.1f} ± {h_b:.1f} min ({veredicto(m_b, h_b)}) y el de los críticos {m_c:+.1f} ± {h_c:.1f} "
          f"({veredicto(m_c, h_c)}). Conviene si protege a los casos de baja prioridad sin perjudicar a los críticos.")

    # P6 · sensibilidad (c9)
    if sens is not None and not sens.empty:
        h("6. ¿Qué supuesto influye más en los resultados? (sensibilidad ±20 %, un factor por vez)")
        tabla = sens.pivot(index="factor", columns="variacion", values="resolucion_p95_min")
        tabla["rango"] = (tabla["+20%"] - tabla["-20%"]).abs()
        tabla = tabla.sort_values("rango", ascending=False).reset_index()
        p(tabla_md(tabla.rename(columns={"factor": "factor (P95 resolución)"})))
        p("")
        p(f"**Respuesta:** el factor más influyente es **{tabla['factor'].iloc[0]}**. Es el dato que más conviene "
          "medir con precisión antes de recomendar cambios (valor de la información).")

    # P7 · umbral de saturación y estrés
    if barrido is not None and not barrido.empty:
        h("7. ¿A partir de qué aumento de demanda el sistema se satura?")
        p(tabla_md(barrido.rename(columns={"factor_demanda": "demanda ×", "resolucion_p95_min": "P95 resolución",
                                           "resolucion_p95_critica_min": "P95 resol. crítica",
                                           "incumplimiento_resolucion_pct": "% incumpl. resol."})))
        limite = 0.10 * barrido["ingresados"] / c.dias
        satura = barrido[barrido["pendientes"] > limite]
        p("")
        if len(satura):
            p(f"**Respuesta:** tomando como criterio ilustrativo un backlog final mayor al 10 % de los ingresos "
              f"diarios, la mesa se satura a partir de **demanda × {satura['factor_demanda'].iloc[0]:.1f}**.")
            i = satura.index[0]
            if i > 0 and barrido.loc[i, "pendientes"] > 1.5 * barrido.loc[i - 1, "pendientes"]:
                p(f"El crecimiento no es lineal: entre × {barrido.loc[i - 1, 'factor_demanda']:.1f} y "
                  f"× {barrido.loc[i, 'factor_demanda']:.1f} los pendientes pasan de "
                  f"{barrido.loc[i - 1, 'pendientes']:.1f} a {barrido.loc[i, 'pendientes']:.1f}.")
        else:
            p("**Respuesta:** en el rango probado el backlog no supera el criterio ilustrativo (10 % de los ingresos diarios).")
        if "Estrés +30% e incidente" in hay:
            p("")
            p(f"En el escenario de estrés (+30 %, incidente de 40 tickets y un especialista menos) quedan "
              f"{esc('Estrés +30% e incidente', 'pendientes'):.0f} pendientes y el incumplimiento de SLA de "
              f"resolución llega a {esc('Estrés +30% e incidente', 'incumplimiento_resolucion_pct'):.1f} %. "
              f"El P95 de resolución de los resueltos es {esc('Estrés +30% e incidente', 'resolucion_p95_min'):.0f} "
              f"min, pero incluyendo los abiertos sube al menos a "
              f"{esc('Estrés +30% e incidente', 'resolucion_p95_con_abiertos_min'):.0f} min.")

    # P8 · calibración y validación (c10)
    if val is not None:
        h("8. ¿El modelo está calibrado y validado?")
        p(f"Datos usados: **{val['origen']}**. Calibración con {val['dias_calibracion']} días y validación "
          f"con {val['dias_validacion']} días reservados, no usados para calibrar.")
        p("")
        p("Parámetros estimados:")
        p("")
        p(tabla_md(val["parametros"], 3))
        p("")
        p("Validez operativa en el período reservado (banda del 95 % de las réplicas simuladas):")
        p("")
        v = val["validacion"][["metrica", "observado", "calibrado_media", "calibrado_banda95",
                               "calibrado_consistente", "supuesto_media", "supuesto_banda95",
                               "supuesto_consistente"]]
        p(tabla_md(v))
        ok_cal = int(val["validacion"]["calibrado_consistente"].sum())
        ok_sup = int(val["validacion"]["supuesto_consistente"].sum())
        n = len(val["validacion"])
        p("")
        if val["sintetico"]:
            p(f"**Respuesta:** todavía **no**. El procedimiento funciona: con la referencia sintética, el modelo "
              f"calibrado reproduce {ok_cal} de {n} métricas del período reservado, contra {ok_sup} de {n} con los "
              "parámetros supuestos. Pero la referencia la generó el propio simulador, así que esto verifica el "
              "método de calibración y no la representatividad del modelo. Para validarlo hay que correr el script "
              "con `--historico` usando datos reales y sumar la validación de cara con supervisores.")
        else:
            p(f"**Respuesta:** con el histórico real, el modelo calibrado reproduce {ok_cal} de {n} métricas del "
              f"período reservado (los parámetros supuestos, {ok_sup} de {n}). Las métricas fuera de banda señalan "
              "qué parte del modelo revisar antes de usarlo para decidir.")

    # P9 a P11 · parte predictiva (c11, c12)
    if pred is not None:
        n_tr, n_va, n_tot = pred["dias"]
        h("9. ¿Se puede anticipar qué tickets van a escalar? ¿La red neuronal supera a modelos más simples?")
        p(f"Datos: {pred['origen']} de {n_tot} días. Partición temporal: días 1–{n_tr} entrenamiento "
          f"({pred['n']['entrenamiento']} tickets), {n_tr + 1}–{n_va} validación ({pred['n']['validacion']}) y "
          f"{n_va + 1}–{n_tot} prueba ({pred['n']['prueba']}). Escalan desde N1 el {100 * pred['tasa_escalamiento']:.1f} %. "
          "Variables: categoría, prioridad, hora, día y cola de N1 al abrir. La complejidad no se usa porque se "
          "conoce recién después del diagnóstico. El umbral de cada modelo se elige con validación.")
        p("")
        res = pred["clasificacion"][["modelo", "AUC", "precision", "recall", "F1", "brier", "umbral",
                                     "VP", "FP", "FN", "VN"]]
        p(tabla_md(res, 3))
        auc_b = float(res.loc[res["modelo"] == "Tasa histórica (baseline)", "AUC"].iloc[0])
        auc_l = float(res.loc[res["modelo"] == "Regresión logística", "AUC"].iloc[0])
        auc_r = float(res.loc[res["modelo"] == "Red neuronal 32-16", "AUC"].iloc[0])
        ep = int(pred["curva"]["epoca"].max())
        p("")
        p(f"La red se detuvo por early stopping en la época {ep} "
          f"(pérdida de validación mínima {pred['curva']['perdida_validacion'].min():.3f}).")
        p("")
        if auc_r - max(auc_b, auc_l) > 0.01:
            p(f"**Respuesta:** la red (AUC {auc_r:.3f}) supera al baseline ({auc_b:.3f}) y a la logística "
              f"({auc_l:.3f}). Antes de adoptarla hay que confirmar que ese margen se mantiene con datos reales.")
        else:
            p(f"**Respuesta:** la predicción es posible pero limitada (AUC ≈ {max(auc_b, auc_l, auc_r):.2f}; "
              f"0,5 sería azar y 1 una predicción perfecta). La red ({auc_r:.3f}) **no supera** de forma relevante a modelos "
              f"más simples (baseline {auc_b:.3f}, logística {auc_l:.3f}), así que no se justifica: conviene un "
              "modelo interpretable. El límite viene de que lo que más determina el escalamiento, "
              "la complejidad, no se conoce al abrir el ticket.")

        h("10. Si se usa el predictor para derivar tickets directo a especialistas, ¿mejora el sistema?")
        if pred["acople"] is None:
            p("**Respuesta pendiente:** con datos reales, este análisis requiere primero el simulador calibrado con "
              "esos mismos datos (pregunta 8).")
        else:
            p(f"Modelo usado: {pred['elegido']} (mejor AUC en validación), umbral {pred['umbral_derivacion']:.2f} "
              "(el de máximo F1 en validación). Los tickets con probabilidad mayor o igual al umbral van directo a N2. "
              "El oráculo indica el máximo beneficio posible con una predicción perfecta.")
            p("")
            p(tabla_md(pred["acople"].rename(columns={
                "politica": "política", "derivados_temprano": "derivados", "respuesta_p95_min": "P95 respuesta",
                "resolucion_p95_min": "P95 resolución", "resolucion_p95_critica_min": "P95 resol. crítica",
                "utilizacion_N1_pct": "util. N1 %", "utilizacion_N2_aplicaciones_pct": "util. N2 apl. %",
                "utilizacion_N2_infraestructura_pct": "util. N2 infra. %",
                "cola_promedio_N2_aplicaciones": "cola N2 apl."})))
            ac = pred["acople"]
            b, q, o = ac.iloc[0], ac.iloc[1], ac.iloc[2]
            reps = pred["acople_replicas"]
            nombres = reps["politica"].unique()
            a_ = reps[reps["politica"] == nombres[0]].set_index("replica")["resolucion_p95_min"]
            b_ = reps[reps["politica"] == nombres[1]].set_index("replica")["resolucion_p95_min"]
            d = (b_ - a_).dropna()
            m, hw = float(d.mean()), semiamplitud_ic(d)
            p("")
            p(f"**Respuesta:** el predictor deriva {q['derivados_temprano']:.0f} tickets por semana. El P95 de resolución "
              f"pasa de {b['resolucion_p95_min']:.0f} a {q['resolucion_p95_min']:.0f} min ({m:+.1f} ± {hw:.1f}, "
              f"{veredicto(m, hw)}), con predicción perfecta llegaría a {o['resolucion_p95_min']:.0f}. La utilización de "
              f"N2 de aplicaciones va de {b['utilizacion_N2_aplicaciones_pct']:.1f} % a "
              f"{q['utilizacion_N2_aplicaciones_pct']:.1f} %: los falsos positivos cargan a los especialistas con casos "
              "que N1 podía resolver. La distancia al oráculo muestra que el beneficio depende de "
              "la calidad de la predicción, por lo que debería usarse primero como recomendación al técnico, con "
              "auditoría, y no como regla automática.")

        h("11. ¿Se puede estimar la demanda por hora?")
        p(tabla_md(pred["demanda"], 2))
        d = pred["demanda"].iloc[:3]
        mejor = d.loc[d["MAE_tickets_por_hora"].idxmin(), "modelo"]
        p("")
        p(f"**Respuesta:** el mejor modelo simple es **{mejor}**. El error que queda es en gran parte aleatorio "
          "(variabilidad Poisson de las llegadas), por lo que un modelo más complejo difícilmente lo reduzca "
          "mucho. Sí sirve para planificar turnos por franja.")

    h("Qué falta para que las conclusiones sean aplicables")
    p("- Calibrar con el historial real de tickets (`--historico`) y repetir la validación operativa.")
    p("- Validación de cara con supervisores y técnicos de la mesa de ayuda.")
    p("- Costos reales por hora de dotación para comparar alternativas en términos económicos.")
    p("- SLA reales del banco por prioridad (los usados son ilustrativos).")
    p("- Entrenar y evaluar los predictores con tickets reales, incluyendo el texto del incidente.")
    return "\n".join(L)


def graficos_extra(barrido, pred, destino, c):
    if barrido is not None and not barrido.empty:
        fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
        limite = 0.10 * barrido["ingresados"] / c.dias
        ax.plot(barrido["factor_demanda"], barrido["pendientes"], "o-", color=AZUL, lw=2, label="Pendientes al final")
        ax.plot(barrido["factor_demanda"], limite, "--", color=ROJO, label="Umbral: 10 % de los ingresos diarios")
        satura = barrido[barrido["pendientes"] > limite]
        if len(satura):
            x0 = satura["factor_demanda"].iloc[0]
            ax.axvspan(x0 - 0.05, barrido["factor_demanda"].max() + 0.05, color=ROJO, alpha=0.08)
            ax.annotate(f"Satura desde × {x0:.1f}", (x0, satura["pendientes"].iloc[0]), xytext=(-90, 25),
                        textcoords="offset points", arrowprops=dict(arrowstyle="->"), fontsize=10)
        ax2 = ax.twinx()
        ax2.plot(barrido["factor_demanda"], barrido["resolucion_p95_min"], "s:", color=GRIS, label="P95 resolución (eje der.)")
        ax2.set_ylabel("P95 de resolución (min)", color=GRIS)
        ax.set(xlabel="Demanda (× escenario base)", ylabel="Tickets pendientes",
               title="¿Desde qué demanda se satura la mesa?")
        lineas = ax.get_legend_handles_labels()
        lineas2 = ax2.get_legend_handles_labels()
        ax.legend(lineas[0] + lineas2[0], lineas[1] + lineas2[1], fontsize=9, loc="upper left", frameon=False)
        ax.grid(alpha=0.25)
        fig.savefig(destino / "barrido_demanda.png", dpi=150)
        plt.close(fig)
    if pred is not None:
        curva = pred["curva"]
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        ax.plot(curva["epoca"], curva["perdida_entrenamiento"], label="Entrenamiento", color=AZUL)
        ax.plot(curva["epoca"], curva["perdida_validacion"], label="Validación", color=ROJO)
        mejor = curva.loc[curva["perdida_validacion"].idxmin()]
        ax.axvline(mejor["epoca"], color="black", ls="--", lw=1)
        ax.annotate(f"Mejor época: {int(mejor['epoca'])}", (mejor["epoca"], mejor["perdida_validacion"]),
                    xytext=(10, 20), textcoords="offset points", fontsize=9)
        ax.set(xlabel="Época", ylabel="Entropía cruzada binaria",
               title="Red neuronal · curva de aprendizaje (se detiene cuando validación deja de mejorar)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.savefig(destino / "red_curva_aprendizaje.png", dpi=150)
        plt.close(fig)


def resumen_consola(comparacion):
    columnas = ["escenario", "resolucion_p95_min", "pendientes", "incumplimiento_resolucion_pct", "utilizacion_N1_pct"]
    print("\nMESA DE AYUDA | COMPARACIÓN DE ESCENARIOS")
    print("P95 de resolución en minutos; promedios entre réplicas")
    print(comparacion[columnas].round(1).to_string(index=False))
    print("\nInterpretar P95 junto con pendientes y SLA; los valores provienen de supuestos.")


def a_json(obj):
    """Convierte resultados (numpy, NaN, DataFrames, dataclasses) a algo serializable en JSON."""
    if isinstance(obj, Configuracion):
        return a_json(asdict(obj))
    if isinstance(obj, pd.DataFrame):
        return a_json(obj.replace({np.nan: None}).to_dict("records"))
    if isinstance(obj, dict):
        return {str(k): a_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [a_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    return obj


def paquete_hipotesis(res, nombre_fuente=None):
    """Experimento reutilizable (mismo formato que guarda el dashboard)."""
    p = res["primera"]
    resumen_b = {k: float(np.nanmean([r["base"][k] for r in res["replicas"]])) for k in res["replicas"][0]["base"]}
    resumen_a = {k: float(np.nanmean([r["alternativa"][k] for r in res["replicas"]])) for k in res["replicas"][0]["alternativa"]}
    return a_json({
        "formato": "mesa_ayuda_v2", "origen": "importado" if res["lote_fijo"] else "sintetico",
        "nombre_fuente": nombre_fuente, "hipotesis": res["tipo"], "magnitud": res["magnitud"],
        "indicador": res["indicador"], "configuracion": res["configuracion"],
        "configuracion_alternativa": res["configuracion_alternativa"],
        "datos_entrada": p["entrada"], "tickets_resultado": p["detalle"], "evolucion": p["evolucion"],
        "replicas": res["replicas"], "resumen": {"base": resumen_b, "alternativa": resumen_a},
        "diferencia_pareada": {"promedio": res["diferencia_media"], "ic95_inf": res["ic95"][0],
                               "ic95_sup": res["ic95"][1]},
        "veredicto": {k: res[k] for k in ("estado", "enunciado", "explicacion", "como", "efecto_esperado",
                                          "diferencias", "efectos_secundarios", "base_media", "base_ic",
                                          "alternativa_media", "alternativa_ic")},
    })


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Simulación de la mesa de ayuda bancaria. Sin argumentos corre el estudio completo; "
                    "con --hipotesis prueba una hipótesis puntual.",
        epilog="Ejemplos:\n"
               "  python simulacion_tickets_banco.py                        # estudio completo + informe\n"
               "  python simulacion_tickets_banco.py --listar-hipotesis\n"
               "  python simulacion_tickets_banco.py --hipotesis menos_escalamiento --magnitud 25\n"
               "  python simulacion_tickets_banco.py --hipotesis refuerzo_n2 --magnitud 2 --indicador pendientes\n"
               "  python simulacion_tickets_banco.py --probar                 # pruebas de verificación\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dias", type=int)
    parser.add_argument("--replicas", type=int)
    parser.add_argument("--semilla", type=int)
    parser.add_argument("--config", type=Path, help="JSON con parámetros a modificar")
    parser.add_argument("--hipotesis", choices=list(HIPOTESIS), help="Probar una hipótesis puntual")
    parser.add_argument("--magnitud", type=float, help="Magnitud de la hipótesis (ver --listar-hipotesis)")
    parser.add_argument("--indicador", choices=list(INDICADORES), default="resolucion_p95_min")
    parser.add_argument("--listar-hipotesis", action="store_true")
    parser.add_argument("--guardar", type=Path, help="Guardar el experimento en JSON (se puede abrir en el dashboard)")
    parser.add_argument("--historico", type=Path, help="CSV de tickets reales para calibrar, validar y entrenar")
    parser.add_argument("--solo-base", action="store_true")
    parser.add_argument("--sin-graficos", action="store_true")
    parser.add_argument("--sin-validacion", action="store_true")
    parser.add_argument("--sin-prediccion", action="store_true")
    parser.add_argument("--probar", action="store_true")
    parser.add_argument("--salida", type=Path, default=Path("resultados_tickets_banco"))
    args = parser.parse_args()

    if args.probar:
        pruebas_verificacion()
        return
    if args.listar_hipotesis:
        print("Hipótesis disponibles (--hipotesis TIPO --magnitud N):\n")
        for k, h in HIPOTESIS.items():
            mag = f"magnitud: {h['magnitud']}, por defecto {h['por_defecto']}" if h["magnitud"] else "sin magnitud"
            print(f"  {k:<20} {h['titulo']} ({mag})\n  {'':<20} {h['como']}\n")
        print("Indicadores (--indicador):")
        for k, v in INDICADORES.items():
            print(f"  {k:<34} {v}")
        return

    c = Configuracion()
    if args.config:
        cambios = json.loads(args.config.read_text(encoding="utf-8"))
        desconocidos = set(cambios) - set(asdict(c))
        if desconocidos:
            parser.error(f"Parámetros desconocidos: {sorted(desconocidos)}")
        c = replace(c, **cambios)
    for campo in ("dias", "replicas", "semilla"):
        v = getattr(args, campo)
        if v is not None:
            setattr(c, campo, v)
    validar_config(c)
    args.salida.mkdir(parents=True, exist_ok=True)

    # Modo hipótesis: una pregunta puntual con veredicto claro
    if args.hipotesis:
        res = probar_hipotesis(c, args.hipotesis, args.magnitud, args.indicador)
        ruta = None
        if not args.sin_graficos:
            ruta = args.salida / f"hipotesis_{args.hipotesis}.png"
            grafico_hipotesis(res, ruta)
        imprimir_hipotesis(res, ruta)
        if args.guardar:
            args.guardar.write_text(json.dumps(paquete_hipotesis(res), ensure_ascii=False, indent=2), encoding="utf-8")
            print("Experimento guardado:", args.guardar)
        return

    # Estudio completo
    (args.salida / "configuracion_utilizada.json").write_text(
        json.dumps(asdict(c), ensure_ascii=False, indent=2), encoding="utf-8")
    print("Corriendo escenarios...")
    lista = {"Base": c} if args.solo_base else escenarios(c)
    replicas, comparacion = ejecutar_experimentos(c, lista)
    base, detalle, eventos, evolucion = simular(c, generar_tickets(c, c.semilla), guardar_traza=True)
    sens = pd.DataFrame() if args.solo_base else sensibilidad(c, min(c.replicas, 10))
    barrido = None if args.solo_base else barrido_demanda(c, min(c.replicas, 10))
    val = None
    if not args.sin_validacion:
        print("Calibrando y validando...")
        val = calibracion_y_validacion(c, args.historico, min(c.replicas, 30))
    pred = None
    if not args.sin_prediccion:
        print("Entrenando modelos predictivos...")
        historico = preparar_historico(pd.read_csv(args.historico)) if args.historico else None
        pred = parte_predictiva(c, min(c.replicas, 30), historico)

    replicas.to_csv(args.salida / "replicas.csv", index=False, encoding="utf-8-sig")
    comparacion.to_csv(args.salida / "comparacion_escenarios.csv", index=False, encoding="utf-8-sig")
    detalle.to_csv(args.salida / "tickets_base.csv", index=False, encoding="utf-8-sig")
    for agrupacion in ("prioridad", "categoria"):
        desglosar_tickets(detalle, agrupacion, c).to_csv(args.salida / f"desglose_por_{agrupacion}.csv",
                                                         index=False, encoding="utf-8-sig")
    eventos.to_csv(args.salida / "eventos_base.csv", index=False, encoding="utf-8-sig")
    evolucion.to_csv(args.salida / "evolucion_colas_base.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(args.salida / "sensibilidad.csv", index=False, encoding="utf-8-sig")
    if barrido is not None:
        barrido.to_csv(args.salida / "barrido_demanda.csv", index=False, encoding="utf-8-sig")
    if val is not None:
        val["validacion"].to_csv(args.salida / "validacion_operativa.csv", index=False, encoding="utf-8-sig")
        val["parametros"].to_csv(args.salida / "parametros_calibrados.csv", index=False, encoding="utf-8-sig")
        (args.salida / "configuracion_calibrada.json").write_text(
            json.dumps(asdict(val["config_calibrada"]), ensure_ascii=False, indent=2), encoding="utf-8")
    if pred is not None:
        pred["clasificacion"].to_csv(args.salida / "prediccion_escalamiento.csv", index=False, encoding="utf-8-sig")
        pred["demanda"].to_csv(args.salida / "prediccion_demanda.csv", index=False, encoding="utf-8-sig")
        if pred["acople_replicas"] is not None:
            pred["acople_replicas"].to_csv(args.salida / "acople_predictor_replicas.csv", index=False, encoding="utf-8-sig")
        pred["curva"].to_csv(args.salida / "red_curva_aprendizaje.csv", index=False, encoding="utf-8-sig")
    if not args.sin_graficos:
        graficos(comparacion, evolucion, sens, args.salida, c, replicas)
        graficos_extra(barrido, pred, args.salida, c)

    informe = armar_informe(c, comparacion, replicas, sens, barrido, val, pred)
    (args.salida / "informe_resultados.md").write_text(informe, encoding="utf-8")
    resumen_consola(comparacion)
    print(informe)
    if args.guardar:
        paquete = a_json({"formato": "mesa_ayuda_v2", "origen": "sintetico", "configuracion": c,
                          "datos_entrada": [ticket_a_dict(t) for t in generar_tickets(c, c.semilla)],
                          "resultados": comparacion, "replicas": replicas, "tickets_resultado": detalle,
                          "eventos": eventos, "evolucion": evolucion})
        args.guardar.write_text(json.dumps(paquete, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Experimento guardado:", args.guardar)
    print("\nArchivos CSV" + (" y PNG" if not args.sin_graficos else "") + " e informe_resultados.md en:",
          args.salida.resolve())


if __name__ == "__main__":
    main()
