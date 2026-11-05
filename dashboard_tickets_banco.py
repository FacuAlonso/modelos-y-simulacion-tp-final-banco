"""Dashboard web local para experimentar con la simulación de mesa de ayuda."""
import copy
import hashlib
import json
import math
import os
import webbrowser
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread, Timer
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from simulacion_tickets_banco import (
    Configuracion, Ticket, generar_tickets, simular, validar_config, EQUIPOS,
    diferencia_pareada, ejecutar_experimentos, escenarios, semiamplitud_ic, veredicto,
)
from estudio_redes import ejecutar_estudio

ESTADO = {"paquete": None, "entrada_cargada": None, "nombre_entrada": None}
RESUMEN = {"estado": "inactivo", "etapa": None, "resultado": None, "error": None}
RESUMEN_LOCK = Lock()
BASE = Path(__file__).parent
CACHE_RESUMEN = BASE / "resultados_tickets_banco" / "resumen_dashboard.json"
MULTIPLOS_DEMANDA = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8)
UMBRAL_UTILIZACION = 85.0       # criterio ilustrativo de cuello de botella
UMBRAL_BACKLOG = 0.10           # backlog final mayor al 10 % de los ingresos diarios
UMBRAL_SLA = 5.0                # criterio ilustrativo de incumplimiento de SLA de resolución


def limpiar(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, dict):
        return {str(k): limpiar(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [limpiar(v) for v in obj]
    return obj


def configurar(valores):
    c = Configuracion()
    permitidos = set(asdict(c))
    for k, v in valores.items():
        if k not in permitidos:
            raise ValueError(f"Parámetro desconocido: {k}")
        setattr(c, k, v)
    validar_config(c)
    if c.dias > 30 or c.replicas > 100:
        raise ValueError("Máximo: 30 días y 100 réplicas para el dashboard")
    return c


def alternativa(c, tipo, cantidad):
    cfg = copy.deepcopy(c)
    if tipo == "menos_escalamiento":
        cfg.factor_escalamiento_n1 *= 1 - cantidad / 100
    elif tipo == "refuerzo_n2":
        cfg.turnos["N2_aplicaciones"].extend([[0, 480]] * int(cantidad))
    elif tipo == "redistribucion":
        total = len(cfg.turnos["N1"])
        cfg.turnos["N1"] = [[0, 480]] * (total // 2) + [[120, 600]] * (total - total // 2)
    elif tipo == "demanda":
        cfg.factor_demanda *= 1 + cantidad / 100
    elif tipo == "servicio":
        cfg.factor_servicio *= 1 - cantidad / 100
    elif tipo == "prioridad":
        cfg.politica_prioridad = "envejecimiento"
    else:
        raise ValueError("Hipótesis no válida")
    validar_config(cfg)
    return cfg


def crear_tickets(filas):
    atributos = set(Ticket.__dataclass_fields__)
    requeridos = {"id", "llegada", "prioridad", "categoria", "complejidad", "servicio",
                  "u_escala_n1", "u_escala_n2", "u_reabrir", "demora_reapertura"}
    datos = []
    for fila in filas:
        if not isinstance(fila, dict) or not requeridos.issubset(fila):
            raise ValueError("Formato de datos de entrada incompleto")
        t = Ticket(**{k: v for k, v in fila.items() if k in atributos})
        t.estado, t.equipo, t.primera_respuesta, t.cierre = "nuevo", "N1", None, None
        t.ingreso_cola, t.espera_acumulada, t.trabajo_terminado = 0.0, 0.0, 0.0
        t.escaladas = t.reaperturas = 0
        datos.append(t)
    if len(datos) > 100000:
        raise ValueError("Máximo 100.000 tickets importados")
    if len({t.id for t in datos}) != len(datos):
        raise ValueError("ID de ticket duplicado")
    return datos


def promedio_metricas(filas):
    claves = [k for k in filas[0] if isinstance(filas[0][k], (int, float)) and not isinstance(filas[0][k], bool)]
    return {k: float(np.nanmean([f[k] for f in filas])) for k in claves}


def margen_ic95(valores):
    valores = np.asarray(valores, dtype=float)
    valores = valores[np.isfinite(valores)]
    if len(valores) < 2:
        return None
    t_criticos = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    t = t_criticos.get(len(valores) - 1, 1.96)
    return float(t * np.std(valores, ddof=1) / math.sqrt(len(valores)))


def ejecutar(peticion):
    c = configurar(peticion.get("configuracion", {}))
    tipo = peticion.get("hipotesis", "menos_escalamiento")
    cantidad = float(peticion.get("magnitud", 25))
    if cantidad < 0 or cantidad > 100:
        raise ValueError("Magnitud fuera de rango (0 a 100)")
    if tipo == "refuerzo_n2" and (cantidad < 1 or cantidad > 5 or int(cantidad) != cantidad):
        raise ValueError("El refuerzo debe ser de 1 a 5 técnicos")
    alt = alternativa(c, tipo, cantidad)
    base_filas, alt_filas, replicas = [], [], []
    primera_entrada = None
    primera_detalle = None
    primera_evolucion = None
    entrada_cargada = ESTADO["entrada_cargada"]
    for i in range(c.replicas):
        semilla = c.semilla + i * 101
        datos_base = crear_tickets(entrada_cargada) if entrada_cargada is not None else generar_tickets(c, semilla)
        datos_alt = (crear_tickets(entrada_cargada) if entrada_cargada is not None
                     else generar_tickets(alt, semilla))
        if i == 0:
            primera_entrada = [asdict(t) for t in datos_base]
        rb, db, _, eb = simular(c, datos_base, guardar_traza=(i == 0))
        ra, _, _, _ = simular(alt, datos_alt)
        if i == 0:
            primera_detalle = db.replace({np.nan: None}).to_dict("records")
            primera_evolucion = eb.to_dict("records")
        base_filas.append(rb)
        alt_filas.append(ra)
        replicas.append({"replica": i + 1, "semilla": semilla, "base": rb, "alternativa": ra})
    metrics_base = promedio_metricas(base_filas)
    metrics_alt = promedio_metricas(alt_filas)
    kpi = peticion.get("indicador", "resolucion_p95_min")
    if kpi not in ("resolucion_p95_min", "resolucion_p95_con_abiertos_min", "respuesta_p95_min",
                   "pendientes", "incumplimiento_resolucion_pct", "cola_maxima"):
        raise ValueError("Indicador no válido")
    diferencias = np.asarray([x["alternativa"][kpi] - x["base"][kpi] for x in replicas], dtype=float)
    diferencias = diferencias[np.isfinite(diferencias)]
    media_delta = float(np.mean(diferencias)) if len(diferencias) else None
    margen = None if entrada_cargada is not None else margen_ic95(diferencias)
    ic95_inf = media_delta - margen if margen is not None else None
    ic95_sup = media_delta + margen if margen is not None else None
    espera_mejora = tipo != "demanda"
    if entrada_cargada is not None:
        cambio_esperado = media_delta < 0 if espera_mejora else media_delta > 0
        estado_validacion = "cumple_lote" if cambio_esperado else (
            "sin_cambio_lote" if abs(media_delta or 0) < 1e-12 else "no_cumple_lote"
        )
        criterio_validacion = "Comparación descriptiva sobre un único lote importado; no permite inferir variabilidad."
    elif ic95_sup is not None and (
        (espera_mejora and ic95_sup < 0) or
        (not espera_mejora and ic95_inf > 0)
    ):
        estado_validacion = "sostenida"
        criterio_validacion = "El IC95% de la diferencia queda del lado esperado para esta hipótesis."
    elif ic95_inf is not None and (
        (espera_mejora and ic95_inf > 0) or
        (not espera_mejora and ic95_sup < 0)
    ):
        estado_validacion = "refutada"
        criterio_validacion = "El IC95% de la diferencia queda del lado contrario al esperado."
    else:
        estado_validacion = "no_concluyente"
        criterio_validacion = "La cantidad de réplicas o la variabilidad no permite distinguir la diferencia de cero."
    paquete = {
        "formato": "mesa_ayuda_v2", "origen": "importado" if entrada_cargada is not None else "sintetico",
        "nombre_fuente": ESTADO["nombre_entrada"], "hipotesis": tipo, "magnitud": cantidad,
        "indicador": kpi, "configuracion": asdict(c), "configuracion_alternativa": asdict(alt),
        "datos_entrada": primera_entrada, "tickets_resultado": primera_detalle,
        "evolucion": primera_evolucion, "replicas": replicas,
        "resumen": {"base": metrics_base, "alternativa": metrics_alt},
        "diferencia_pareada": {"promedio": media_delta,
                               "ic95_inf": ic95_inf, "ic95_sup": ic95_sup,
                               "metodo": "t de Student" if entrada_cargada is None else "sin IC por lote fijo"},
        "validacion": {"estado": estado_validacion, "criterio": criterio_validacion,
                       "metodo": "t de Student" if entrada_cargada is None else "sin IC por lote fijo"},
        "advertencias": ["Los datos sintéticos no constituyen evidencia empírica ni entrenamiento validado.",
                         "Las métricas temporales utilizan minutos corridos y P95 solo de tickets resueltos.",
                         "Con datos importados se reutiliza exactamente la misma entrada en todas las réplicas: los IC no estiman variabilidad de demanda."] if entrada_cargada is not None else [
                         "Los supuestos no están calibrados con registros bancarios reales.",
                         "P95 se calcula solo entre tickets resueltos: comparar también pendientes y SLA."]
    }
    ESTADO["paquete"] = limpiar(paquete)
    return {k: ESTADO["paquete"][k] for k in (
        "resumen", "diferencia_pareada", "validacion", "indicador", "hipotesis",
        "magnitud", "advertencias", "origen", "replicas")}


def responder_pregunta(c):
    """P1: qué alternativa reduce más el P95 de resolución sin aumentar el backlog (mismo criterio del informe)."""
    df, resumen = ejecutar_experimentos(c, escenarios(c))
    fila = resumen.set_index("escenario")
    alternativas = []
    for nombre in fila.index:
        if nombre in ("Base", "Estrés +30% e incidente"):
            continue
        p95, ic_p95 = diferencia_pareada(df, nombre, "resolucion_p95_min")
        pendientes, ic_pendientes = diferencia_pareada(df, nombre, "pendientes")
        critica, ic_critica = diferencia_pareada(df, nombre, "resolucion_p95_critica_min")
        alternativas.append({
            "nombre": nombre, "p95": p95, "ic_p95": ic_p95, "pendientes": pendientes, "ic_pendientes": ic_pendientes,
            "critica": critica, "ic_critica": ic_critica, "veredicto": veredicto(p95, ic_p95),
            "horas_extra": float(fila.loc[nombre, "horas_dotacion"] - fila.loc["Base", "horas_dotacion"]),
            "cumple": bool(p95 + ic_p95 < 0 and pendientes - ic_pendientes <= 0),
        })
    alternativas.sort(key=lambda x: x["p95"])
    campos = ("resolucion_p95_min", "pendientes", "incumplimiento_resolucion_pct", "horas_dotacion")
    return {
        "alternativas": alternativas,
        "mejor": next((x["nombre"] for x in alternativas if x["cumple"]), None),
        "base": {k: float(fila.loc["Base", k]) for k in campos},
        "estres": {k: float(fila.loc["Estrés +30% e incidente", k]) for k in campos},
        "replicas": c.replicas, "dias": c.dias,
    }


def buscar_puntos_criticos(c):
    """P7: barre la demanda y marca dónde se cruza cada criterio."""
    columnas_util = {e: f"utilizacion_{e}_pct" for e in EQUIPOS}
    filas = []
    for m in MULTIPLOS_DEMANDA:
        cfg = replace(c, factor_demanda=c.factor_demanda * m)
        corridas = [simular(cfg, generar_tickets(cfg, c.semilla + r * 101))[0] for r in range(c.replicas)]
        media = lambda k: float(np.nanmean([x[k] for x in corridas]))
        filas.append({
            "demanda": m, "pendientes": media("pendientes"),
            "ic_pendientes": semiamplitud_ic([x["pendientes"] for x in corridas]),
            "limite_pendientes": UMBRAL_BACKLOG * media("ingresados") / c.dias,
            "p95": media("resolucion_p95_min"), "p95_abiertos": media("resolucion_p95_con_abiertos_min"),
            "sla": media("incumplimiento_resolucion_pct"), "cola_maxima": media("cola_maxima"),
            "utilizacion": {e: media(k) for e, k in columnas_util.items()},
        })
    puntos = []
    primera_util = next((f for f in filas if max(f["utilizacion"].values()) >= UMBRAL_UTILIZACION), None)
    if primera_util:
        equipo = max(primera_util["utilizacion"], key=primera_util["utilizacion"].get)
        puntos.append({"demanda": primera_util["demanda"], "tipo": "cuello",
                       "titulo": f"{equipo.replace('_', ' ')} llega a {UMBRAL_UTILIZACION:.0f} % de utilización",
                       "detalle": "Primer cuello de botella: el equipo queda sin margen para absorber picos."})
    primera_sla = next((f for f in filas if f["sla"] > UMBRAL_SLA), None)
    if primera_sla:
        puntos.append({"demanda": primera_sla["demanda"], "tipo": "sla",
                       "titulo": f"El incumplimiento de SLA supera {UMBRAL_SLA:.0f} %",
                       "detalle": "Los tiempos de resolución empiezan a incumplir el compromiso de forma visible."})
    primera_sat = next((f for f in filas if f["pendientes"] > f["limite_pendientes"]), None)
    if primera_sat:
        puntos.append({"demanda": primera_sat["demanda"], "tipo": "saturacion",
                       "titulo": "La mesa se satura",
                       "detalle": f"El backlog final supera {100 * UMBRAL_BACKLOG:.0f} % de los ingresos diarios: "
                                  "la demanda deja de absorberse dentro del horizonte."})
    puntos.sort(key=lambda x: x["demanda"])
    return {"filas": filas, "puntos": puntos, "saturacion": primera_sat["demanda"] if primera_sat else None,
            "umbral_utilizacion": UMBRAL_UTILIZACION, "umbral_backlog_pct": 100 * UMBRAL_BACKLOG,
            "umbral_sla": UMBRAL_SLA}


def huella_resumen(c):
    """El resumen guardado solo vale si no cambiaron el modelo ni los parámetros."""
    h = hashlib.sha256(json.dumps(asdict(c), sort_keys=True).encode("utf-8"))
    for nombre in ("simulacion_tickets_banco.py", "estudio_redes.py"):
        h.update((BASE / nombre).read_bytes())
    return h.hexdigest()


def calcular_resumen(c=None):
    """Publica cada parte apenas está lista, para que la página responda lo antes posible."""
    c = c or Configuracion()
    parcial = {"huella": huella_resumen(c)}

    def etapa(texto):
        with RESUMEN_LOCK:
            RESUMEN["etapa"] = texto

    def publicar(clave, valor):
        parcial[clave] = limpiar(valor)
        with RESUMEN_LOCK:
            RESUMEN["resultado"] = dict(parcial)

    try:
        etapa("Comparando las alternativas contra la base")
        publicar("pregunta", responder_pregunta(c))
        etapa("Buscando los puntos críticos de demanda")
        publicar("criticos", buscar_puntos_criticos(c))
        etapa("Evaluando la red neuronal (la parte más lenta)")
        publicar("redes", ejecutar_estudio(c=c, avisar=etapa))
        CACHE_RESUMEN.parent.mkdir(parents=True, exist_ok=True)
        CACHE_RESUMEN.write_text(json.dumps(parcial, ensure_ascii=False), encoding="utf-8")
        with RESUMEN_LOCK:
            RESUMEN.update(estado="listo", etapa=None, error=None)
    except Exception as exc:
        with RESUMEN_LOCK:
            RESUMEN.update(estado="error", etapa=None, error=str(exc))


def iniciar_resumen(forzar=False):
    """Usa el resumen guardado si sigue vigente; si no, lo calcula en segundo plano."""
    c = Configuracion()
    with RESUMEN_LOCK:
        if RESUMEN["estado"] == "ejecutando":
            raise ValueError("Ya hay un cálculo en curso")
        if not forzar and CACHE_RESUMEN.exists():
            try:
                guardado = json.loads(CACHE_RESUMEN.read_text(encoding="utf-8"))
                if guardado.get("huella") == huella_resumen(c):
                    RESUMEN.update(estado="listo", etapa=None, resultado=guardado, error=None)
                    return
            except (OSError, ValueError):
                pass
        RESUMEN.update(estado="ejecutando", etapa="Iniciando", resultado=None, error=None)
    Thread(target=calcular_resumen, args=(c,), daemon=True).start()


class Servidor(BaseHTTPRequestHandler):
    def log_message(self, formato, *args):
        # http.server escribe cada pedido en stderr. Si la terminal que lanzó el servidor ya no
        # existe, esa escritura falla antes de responder y el navegador ve ERR_EMPTY_RESPONSE.
        pass

    def responder(self, estado, objeto, tipo="application/json; charset=utf-8", adjunto=None):
        cuerpo = objeto if isinstance(objeto, bytes) else json.dumps(objeto, ensure_ascii=False).encode("utf-8")
        self.send_response(estado)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        if adjunto:
            self.send_header("Content-Disposition", f'attachment; filename="{adjunto}"')
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self):
        ruta = urlparse(self.path).path
        if ruta == "/":
            self.responder(200, (BASE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        elif ruta == "/api/config":
            self.responder(200, {"configuracion": asdict(Configuracion()), "equipos": EQUIPOS})
        elif ruta == "/api/guardar":
            if ESTADO["paquete"] is None:
                self.responder(400, {"error": "Primero ejecutá una simulación"})
            else:
                self.responder(200, json.dumps(ESTADO["paquete"], ensure_ascii=False, indent=2).encode("utf-8"),
                               "application/json; charset=utf-8", "experimento_mesa_ayuda.json")
        elif ruta == "/api/estado":
            p = ESTADO["paquete"]
            self.responder(200, {"cargado": ESTADO["nombre_entrada"],
                                   "resultado": p and {k: p.get(k) for k in ("resumen", "diferencia_pareada", "validacion", "indicador", "advertencias", "origen", "replicas")}})
        elif ruta == "/api/resumen":
            with RESUMEN_LOCK:
                copia = dict(RESUMEN)
            self.responder(200, copia)
        else:
            self.responder(404, {"error": "No encontrado"})

    def do_POST(self):
        try:
            tam = int(self.headers.get("Content-Length", "0"))
            if tam <= 0 or tam > 35_000_000:
                raise ValueError("Archivo o petición demasiado grande")
            datos = json.loads(self.rfile.read(tam))
            ruta = urlparse(self.path).path
            if ruta == "/api/ejecutar":
                salida = ejecutar(datos)
            elif ruta == "/api/recalcular":
                iniciar_resumen(forzar=True)
                salida = {"estado": "ejecutando"}
            elif ruta == "/api/cargar":
                if datos.get("formato") != "mesa_ayuda_v2":
                    raise ValueError("Se espera un experimento mesa_ayuda_v2")
                if not isinstance(datos.get("datos_entrada"), list):
                    raise ValueError("No hay datos de entrada en el archivo")
                crear_tickets(datos["datos_entrada"])
                ESTADO["entrada_cargada"] = datos["datos_entrada"]
                ESTADO["nombre_entrada"] = "Archivo JSON importado"
                ESTADO["paquete"] = datos if "resumen" in datos else None
                salida = {"mensaje": f"Cargados {len(datos['datos_entrada'])} tickets",
                          "resultado": {k: datos.get(k) for k in ("resumen", "diferencia_pareada", "validacion", "indicador", "advertencias", "origen", "replicas")} if "resumen" in datos else None,
                          "configuracion": datos.get("configuracion")}
            elif ruta == "/api/limpiar":
                ESTADO["entrada_cargada"] = None
                ESTADO["nombre_entrada"] = None
                salida = {"mensaje": "Generación sintética activada"}
            else:
                self.responder(404, {"error": "No encontrado"})
                return
            self.responder(200, salida)
        except (ValueError, TypeError, AssertionError, KeyError, OverflowError) as exc:
            self.responder(400, {"error": str(exc) or "Parámetros no válidos"})
        except Exception as exc:
            self.responder(500, {"error": f"Error de simulación: {exc}"})


def crear_servidor(puerto, intentos=10):
    # En Windows, SO_REUSEADDR deja a varios procesos escuchar en el mismo puerto y el pedido puede
    # llegarle a un servidor viejo. Sin esa opción, un puerto ocupado se detecta de inmediato.
    ThreadingHTTPServer.allow_reuse_address = os.name != "nt"
    for candidato in range(puerto, puerto + intentos):
        try:
            return ThreadingHTTPServer(("127.0.0.1", candidato), Servidor)
        except OSError:
            continue
    raise SystemExit(f"No hay puertos libres entre {puerto} y {puerto + intentos - 1}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--puerto", type=int, default=8765)
    parser.add_argument("--sin-navegador", action="store_true")
    args = parser.parse_args()
    servidor = crear_servidor(args.puerto)
    puerto = servidor.server_address[1]
    direccion = f"http://127.0.0.1:{puerto}"
    if puerto != args.puerto:
        print(f"El puerto {args.puerto} está ocupado por otro proceso: se usa el {puerto}.")
    print(f"Dashboard de mesa de ayuda: {direccion}")
    print("Presioná Ctrl+C para cerrar")
    iniciar_resumen()
    if not args.sin_navegador:
        Timer(0.8, lambda: webbrowser.open(direccion)).start()
    servidor.serve_forever()


if __name__ == "__main__":
    main()
