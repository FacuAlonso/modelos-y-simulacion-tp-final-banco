"""Estudio: ¿conviene sumar una red neuronal para predecir el escalamiento de tickets?

Se ejecuta en la misma carpeta que simulacion_tickets_banco.py y solo usa sus funciones públicas
(Configuracion, generar_tickets, simular). No modifica el simulador.

    python estudio_redes.py                         # historial simulado, 5 historias
    python estudio_redes.py --historias 10 --dias 80
    python estudio_redes.py --historico tickets.csv  # con datos reales

Pasos:
  1. Techo: con muchísimos datos, ¿cuál es el mejor AUC posible usando solo lo que se sabe al abrir el
     ticket? Si los modelos simples ya están cerca del techo, ninguna red puede mejorar mucho.
  2. Comparación con partición temporal: tasa histórica (baseline), regresión logística y red 32-16
     (ReLU, sigmoide, entropía cruzada binaria, Adam, early stopping). El umbral se elige en validación.
  3. Robustez: se repite con varias historias (semillas) y se calcula un IC bootstrap de la diferencia
     de AUC entre la red y el mejor modelo simple.
  4. Regla de decisión: la red se justifica solo si mejora el AUC en al menos 0,01 con un IC por
     encima de 0 en la mayoría de las historias.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulacion_tickets_banco import Configuracion, generar_tickets, simular

CATEGORIAS = ("accesos", "aplicaciones", "conectividad", "hardware")
PRIORIDADES = ("critica", "alta", "media", "baja")
MEJORA_MINIMA = 0.01


# ---------- Datos ----------

def preparar(det):
    """Variables conocidas al abrir el ticket + etiqueta (escaló o no). La complejidad NO se usa:
    se conoce después del diagnóstico. Solo quedan tickets con etiqueta conocida."""
    d = det.copy()
    if "duplicado" in d:
        d = d[~d["duplicado"].astype(bool)]
    if "escalo_desde_n1" in d and "paso_por_n1" in d:
        d = d[d["paso_por_n1"].astype(bool)]
        d["y"] = d["escalo_desde_n1"].astype(int)
    else:
        d = d[(d["escaladas"] >= 1) | (d["estado_final"] == "resuelto")]
        d["y"] = (d["escaladas"] >= 1).astype(int)
    d["hora"] = ((d["minuto_apertura"] % 1440) // 60).astype(int).clip(0, 9)
    d["dia"] = ((d["minuto_apertura"] // 1440).astype(int)) + 1
    d["dia_semana"] = (d["dia"] - 1) % 5
    return d.reset_index(drop=True)


def historial_simulado(c, dias, semilla):
    cfg = replace(c, dias=dias)
    _, det, _, _ = simular(cfg, generar_tickets(cfg, semilla))
    return preparar(det)


def matriz(d):
    cols = [(d["categoria"] == k) for k in CATEGORIAS] + [(d["prioridad"] == p) for p in PRIORIDADES]
    cols += [(d["hora"] == h) for h in range(10)] + [(d["dia_semana"] == k) for k in range(5)]
    return np.column_stack(cols).astype(float)


# ---------- Modelos ----------

def sigmoide(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def tasa_historica(tr):
    tabla = tr.groupby(["categoria", "prioridad"])["y"].mean()
    media = tr["y"].mean()
    return lambda d: np.array([tabla.get((a, b), media) for a, b in zip(d["categoria"], d["prioridad"])])


def logistica(X, y, l2=1e-3, lr=0.5, epocas=3000):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epocas):
        g = sigmoide(X @ w + b) - y
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return lambda Xn: sigmoide(Xn @ w + b)


def red(Xtr, ytr, Xva, yva, capas=(32, 16), lr=1e-3, lote=64, max_epocas=300, paciencia=20, l2=1e-4, semilla=0):
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

    mejor, guardado, sin_mejora, paso, curva = np.inf, None, 0, 0, []
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
        curva.append((epoca + 1, ltr, lva))
        if lva < mejor - 1e-5:
            mejor, sin_mejora = lva, 0
            guardado = ([w.copy() for w in W], [b.copy() for b in B])
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    Wf, Bf = guardado
    return (lambda Xn: adelante(Xn, Wf, Bf)[1]), curva


# ---------- Métricas ----------

def auc(y, p):
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def f1(y, p, u):
    pred = p >= u
    tp, fp, fn = np.sum(pred & (y == 1)), np.sum(pred & (y == 0)), np.sum(~pred & (y == 1))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def umbral(y, p):
    grilla = np.arange(0.05, 0.96, 0.01)
    return float(max(grilla, key=lambda u: f1(y, p, u)))


def bootstrap_dif_auc(y, p_a, p_b, n=500, semilla=0):
    rng = np.random.default_rng(semilla)
    difs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            difs.append(auc(y[i], p_a[i]) - auc(y[i], p_b[i]))
    return float(np.mean(difs)), float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))


# ---------- Estudio ----------

def techo(c, dias_grande=300):
    """Mejor AUC posible con la información al abrir: tasa por categoría × prioridad estimada con
    muchísimos datos y evaluada en otra muestra independiente."""
    grande = historial_simulado(c, dias_grande, 9001)
    otra = historial_simulado(c, dias_grande, 9002)
    p = tasa_historica(grande)(otra)
    resultado = {"con datos al abrir": auc(otra["y"].to_numpy(), p)}
    det = otra
    if "complejidad" in det:
        verdadera = np.array([min(1.0, c.prob_escalamiento_n1[a] * c.factor_escalamiento_n1 * c.factor_complejidad[b])
                              for a, b in zip(det["categoria"], det["complejidad"])])
        resultado["conociendo la complejidad"] = auc(det["y"].to_numpy(), verdadera)
    return resultado


def evaluar(d, semilla):
    dias = int(d["dia"].max())
    corte_tr, corte_va = int(dias * 0.6), int(dias * 0.8)
    tr, va, te = d[d["dia"] <= corte_tr], d[(d["dia"] > corte_tr) & (d["dia"] <= corte_va)], d[d["dia"] > corte_va]
    if any(parte["y"].nunique() < 2 for parte in (tr, va, te)):
        raise ValueError("El historial es muy corto: cada período necesita tickets que escalen y que no escalen.")
    Xtr, Xva, Xte = matriz(tr), matriz(va), matriz(te)
    ytr, yva, yte = tr["y"].to_numpy(float), va["y"].to_numpy(float), te["y"].to_numpy()
    base = tasa_historica(tr)
    log = logistica(Xtr, ytr)
    nn, curva = red(Xtr, ytr, Xva, yva, semilla=semilla)
    preds = {"Tasa histórica (baseline)": (base(va), base(te)), "Regresión logística": (log(Xva), log(Xte)),
             "Red neuronal 32-16": (nn(Xva), nn(Xte))}
    filas = []
    for nombre, (pva, pte) in preds.items():
        u = umbral(yva, pva)
        filas.append({"modelo": nombre, "AUC": auc(yte, pte), "brier": float(np.mean((pte - yte) ** 2)),
                      "F1": f1(yte, pte, u), "umbral": u})
    res = pd.DataFrame(filas)
    simples = res[res["modelo"] != "Red neuronal 32-16"]
    mejor_simple = simples.loc[simples["AUC"].idxmax(), "modelo"]
    dif = bootstrap_dif_auc(yte, preds["Red neuronal 32-16"][1], preds[mejor_simple][1], semilla=semilla)
    return res, dif, mejor_simple, curva, {"entrenamiento": len(tr), "validacion": len(va), "prueba": len(te),
                                            "tasa_escalamiento": float(d["y"].mean())}


def comparar(historias, avisar=print):
    """Pasos 2, 3 y 4. Devuelve las tablas por historia, el promedio y la decisión."""
    todas, difs, curva0, tamanos = [], [], None, []
    for h, d in enumerate(historias):
        res, dif, mejor, curva, n = evaluar(d, h)
        res["historia"] = h + 1
        todas.append(res)
        tamanos.append(n)
        difs.append({"historia": h + 1, "vs": mejor, "dif_media": dif[0], "ic_inf": dif[1], "ic_sup": dif[2]})
        curva0 = curva0 or curva
        avisar(f"   Historia {h + 1}: " + " | ".join(f"{m}: AUC {a:.3f}" for m, a in zip(res["modelo"], res["AUC"]))
               + f" | red − {mejor.split(' (')[0].lower()}: {dif[0]:+.3f} [{dif[1]:+.3f}; {dif[2]:+.3f}]")
    todas = pd.concat(todas)
    difs = pd.DataFrame(difs)
    resumen = todas.groupby("modelo", sort=False)[["AUC", "brier", "F1"]].mean().reset_index()
    gana = int(((difs["dif_media"] >= MEJORA_MINIMA) & (difs["ic_inf"] > 0)).sum())
    return {"todas": todas, "difs": difs, "resumen": resumen, "curva": curva0, "tamanos": tamanos,
            "gana": gana, "justificada": bool(gana > len(difs) / 2)}


def ejecutar_estudio(historias=5, dias=60, c=None, avisar=lambda _: None):
    """Corre el estudio completo con historial simulado y devuelve un dict listo para el dashboard."""
    c = c or Configuracion()
    avisar("Calculando el techo de predicción (2 × 300 días simulados)")
    cota = techo(c)
    avisar("Simulando historias y entrenando los modelos")
    r = comparar([historial_simulado(c, dias, 4000 + h) for h in range(historias)], avisar)
    return {
        "modelos": r["resumen"].to_dict("records"),
        "auc_por_historia": r["todas"][["historia", "modelo", "AUC"]].to_dict("records"),
        "diferencias": r["difs"].to_dict("records"),
        "techo": cota,
        "curva": [{"epoca": e, "entrenamiento": a, "validacion": b} for e, a, b in r["curva"]],
        "historias": len(r["difs"]), "dias": dias, "gana": r["gana"], "se_justifica": r["justificada"],
        "mejora_minima": MEJORA_MINIMA, "tamanos": r["tamanos"][0],
        "tasa_escalamiento": r["tamanos"][0]["tasa_escalamiento"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--historias", type=int, default=5, help="historias simuladas independientes")
    ap.add_argument("--dias", type=int, default=60, help="días por historia")
    ap.add_argument("--historico", type=Path, help="CSV real: categoria, prioridad, minuto_apertura, escalo_desde_n1")
    ap.add_argument("--salida", type=Path, default=Path("estudio_redes"))
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)
    c = Configuracion()

    if args.historico:
        d = pd.read_csv(args.historico)
        d["escaladas"] = d["escalo_desde_n1"].astype(int)
        d["estado_final"] = "resuelto"
        historias = [preparar(d)]
        cota = None
        print(f"Datos reales: {len(historias[0])} tickets con etiqueta conocida.")
    else:
        print("1) Calculando el techo de predicción con 2 × 300 días simulados...")
        cota = techo(c)
        for k, v in cota.items():
            print(f"   AUC máximo {k}: {v:.3f}")
        historias = [historial_simulado(c, args.dias, 4000 + h) for h in range(args.historias)]

    print(f"\n2) y 3) Comparando modelos en {len(historias)} historia(s) con partición temporal 60/20/20...")
    r = comparar(historias)
    todas, difs, resumen, curva0, gana, justificada = (r["todas"], r["difs"], r["resumen"], r["curva"],
                                                       r["gana"], r["justificada"])
    todas.to_csv(args.salida / "resultados_por_historia.csv", index=False, encoding="utf-8-sig")
    difs.to_csv(args.salida / "diferencias_auc.csv", index=False, encoding="utf-8-sig")

    # 4) Decisión
    print("\nResumen (promedio de las historias):")
    print(resumen.round(3).to_string(index=False))
    print(f"\n4) Decisión: la red supera al mejor modelo simple en ≥ {MEJORA_MINIMA} de AUC, con IC > 0, "
          f"en {gana} de {len(difs)} historias.")
    if justificada:
        print("   → La red SE JUSTIFICA con estos datos: aporta una mejora consistente fuera de muestra.")
    else:
        print("   → La red NO SE JUSTIFICA: no mejora de forma consistente a un modelo simple e interpretable.")
    if cota:
        mejor_auc = resumen["AUC"].max()
        margen = cota["con datos al abrir"] - mejor_auc
        print(f"   El mejor modelo llega a {mejor_auc:.3f} y el techo con la información al abrir es "
              f"{cota['con datos al abrir']:.3f}. " +
              (f"Queda un margen de {margen:.3f} para cualquier modelo." if margen > 0.005 else
               "El mejor modelo ya está en el techo (las diferencias de ±0,01 son ruido de la muestra de prueba).")
              + " Lo demás es azar que ningún modelo puede predecir.")

    # Gráficos
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    ax = axes[0]
    pos = np.arange(len(resumen))
    err = todas.groupby("modelo", sort=False)["AUC"].std().fillna(0).to_numpy()
    ax.bar(pos, resumen["AUC"], yerr=err, capsize=6, color=["#7b8794", "#2f6fdb", "#b42318"])
    ax.set_xticks(pos, [m.replace(" (baseline)", "\n(baseline)") for m in resumen["modelo"]])
    if cota:
        ax.axhline(cota["con datos al abrir"], color="black", ls="--", label="Techo con datos al abrir")
        if "conociendo la complejidad" in cota:
            ax.axhline(cota["conociendo la complejidad"], color="grey", ls=":", label="Techo conociendo la complejidad")
    ax.axhline(0.5, color="#b42318", lw=0.8, alpha=0.5, label="Azar (0,5)")
    ax.set_ylim(0.45, 0.8)
    ax.set_ylabel("AUC en el período de prueba")
    ax.set_title("¿La red supera a los modelos simples?")
    ax.legend(fontsize=9, frameon=False)
    ax = axes[1]
    ep, ltr, lva = zip(*curva0)
    ax.plot(ep, ltr, label="Entrenamiento", color="#2f6fdb")
    ax.plot(ep, lva, label="Validación", color="#b42318")
    ax.set(xlabel="Época", ylabel="Entropía cruzada binaria", title="Curva de aprendizaje de la red (historia 1)")
    ax.legend(frameon=False)
    veredicto = "SE JUSTIFICA" if justificada else "NO SE JUSTIFICA"
    fig.suptitle(f"Estudio de redes neuronales para predecir escalamiento → la red {veredicto}",
                 fontsize=13, fontweight="bold")
    fig.savefig(args.salida / "estudio_redes.png", dpi=150)
    print(f"\nArchivos en {args.salida.resolve()}")


if __name__ == "__main__":
    main()
