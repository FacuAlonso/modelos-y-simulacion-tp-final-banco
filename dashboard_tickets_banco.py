"""Dashboard web local para experimentar con la simulación de mesa de ayuda."""
import copy
import json
import math
import webbrowser
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Timer
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from simulacion_tickets_banco import (
    Configuracion, Ticket, generar_tickets, simular, validar_config, EQUIPOS,
)

ESTADO = {"paquete": None, "entrada_cargada": None, "nombre_entrada": None}
BASE = Path(__file__).parent


def limpiar(obj):
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


class Servidor(BaseHTTPRequestHandler):
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--puerto", type=int, default=8765)
    parser.add_argument("--sin-navegador", action="store_true")
    args = parser.parse_args()
    direccion = f"http://127.0.0.1:{args.puerto}"
    print(f"Dashboard de mesa de ayuda: {direccion}")
    print("Presioná Ctrl+C para cerrar")
    if not args.sin_navegador:
        Timer(0.8, lambda: webbrowser.open(direccion)).start()
    ThreadingHTTPServer(("127.0.0.1", args.puerto), Servidor).serve_forever()


if __name__ == "__main__":
    main()
