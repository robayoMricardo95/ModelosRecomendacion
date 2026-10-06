"""Gráficos de resultados para el README y el notebook.

Ejecutar después de metodos.py:  python figuras.py
Lee resultados/resultados.json y guarda PNG en resultados/figuras/.
Color: azul = recomendar, rojo = no recomendar, gris = neutro. Cada número va escrito en su celda,
así que el color nunca es la única forma de leer el resultado.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np

BASE = Path(__file__).parent
RES = BASE / "resultados"
FIG = RES / "figuras"
CLIENTES = ["Ana", "Beto", "Caro", "Dani", "Eva"]
PROD6 = ["Pan", "Mantequilla", "Leche", "Cereal", "Pañales", "Toallitas"]
PROD7 = PROD6 + ["Avena"]

SUP, TINTA, TINTA2, LINEA = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"
AZUL, ROJO, GRIS = "#2a78d6", "#e34948", "#f0efec"
DIVERGENTE = LinearSegmentedColormap.from_list("div", [ROJO, GRIS, AZUL])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": SUP,
                     "axes.facecolor": SUP, "savefig.facecolor": SUP, "axes.edgecolor": LINEA,
                     "text.color": TINTA, "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2})


def _tinta(rgba):
    r, g, b = rgba[:3]
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else TINTA


def mapa(ax, M, filas, columnas, escala=None, fmt="{:.2f}", resaltar=None, comprados=None, titulo=None):
    """Mapa de calor anotado. escala: valor que corresponde al azul máximo (por defecto, max |M|)."""
    M = np.array(M, dtype=float)
    norm = np.full_like(M, np.nan)
    for i in range(M.shape[0]):
        e = escala[i] if isinstance(escala, (list, np.ndarray)) else (escala or np.nanmax(np.abs(M)))
        norm[i] = np.clip(M[i] / e, -1, 1) if e else 0
    rgb = DIVERGENTE((norm + 1) / 2)
    rgb[np.isnan(M)] = to_rgb(SUP) + (1,)
    if comprados is not None:  # lo ya comprado no es recomendación: se apaga en gris
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if comprados[i][j]: rgb[i, j] = to_rgb("#e4e3df") + (1,)
    ax.imshow(rgb, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color=TINTA2); continue
            ya = comprados is not None and comprados[i][j]
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=9.5,
                    color=TINTA2 if ya else _tinta(rgb[i, j]), fontweight="normal" if ya else "bold",
                    fontstyle="italic" if ya else "normal")
    ax.set_xticks(range(len(columnas)), columnas)
    ax.set_yticks(range(len(filas)), filas)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, len(columnas)), minor=True); ax.set_yticks(np.arange(-.5, len(filas)), minor=True)
    ax.grid(which="minor", color=SUP, lw=2); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values(): s.set_visible(False)
    if resaltar is not None:
        ax.add_patch(plt.Rectangle((-.48, resaltar - .48), len(columnas) - .04, .96, fill=False, edgecolor=TINTA, lw=2, zorder=5, clip_on=False))
    if titulo: ax.set_title(titulo, loc="left", fontsize=11, fontweight="bold", pad=8)


def _guardar(fig, nombre):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / nombre, dpi=160, bbox_inches="tight")
    return fig


def cargar():
    return json.loads((RES / "resultados.json").read_text())


# ------------------------------------------------------------------ 1. resumen
def resumen_beto(r):
    """Qué le recomienda cada método a Beto. Cada fila se colorea con su propia escala."""
    nan = np.nan
    lift = {x["regla"]: x["lift"] for x in r["1_apriori"]}
    apr = [nan] + [lift.get(f"Pan → {p}", 0.0) for p in PROD6[1:]] + [nan]
    fila = lambda d: [d.get(p, nan) for p in PROD7]
    filas = [
        ("1 Apriori · lift Pan → X", apr),
        ("2 Vecindario", fila(r["2_vecindario"]["puntaje"])),
        ("3 SVD (k=2)", fila(r["3_svd"]["beto"])),
        ("4 SVD de Funk", fila(r["4_funk"]["beto"])),
        ("5 ALS", fila(r["5_als"]["beto"])),
        ("6 ALS implicit", fila(r["6_als_implicit"]["beto"])),
        ("7 Contenido", fila(r["7_contenido"]["beto"])),
        ("8 Two-Tower", fila(r["8_two_tower"]["beto"])),
        ("9 GRU4Rec · carrito Pan → Cereal", fila(r["9_gru4rec"]["siguiente"]["Pan → Cereal"])),
        ("10 SASRec · carrito Pan → Cereal", fila(r["10_sasrec"]["siguiente"]["Pan → Cereal"])),
    ]
    M = np.array([f[1] for f in filas])
    escala = [np.nanmax(np.abs(f)) for f in M]
    comprados = [[j < 2 and i < 8 for j in range(7)] for i in range(10)]
    for i in (8, 9): comprados[i][0] = comprados[i][3] = True  # ya están en el carrito
    fig, ax = plt.subplots(figsize=(10, 6.2))
    mapa(ax, M, [f[0] for f in filas], PROD7, escala=escala, comprados=comprados)
    ax.xaxis.tick_top()
    fig.text(0.01, 0.97, "¿Qué le recomienda cada método a Beto?", fontsize=14, fontweight="bold", ha="left")
    fig.text(0.01, -0.02, "Azul: recomendar · rojo: no recomendar · cada fila usa su propia escala. Gris en cursiva: Beto ya lo compró o ya está en su carrito.\n"
             "La respuesta esperada es Leche y Cereal, nunca Pañales. Avena es un producto nuevo: solo los métodos 7 y 8 pueden puntuarla.",
             fontsize=9, color=TINTA2, ha="left", va="top")
    return _guardar(fig, "01_resumen_beto.png")


# ------------------------------------------------------------------ 2. Apriori y vecindario
def apriori_vecindario(r):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})
    elegidas = ["Pañales → Toallitas", "Pan → Mantequilla", "Pan → Cereal", "Cereal → Leche", "Pañales → Leche", "Pan → Leche"]
    reglas = [x for e in elegidas for x in r["1_apriori"] if x["regla"] == e]
    nombres = [x["regla"] for x in reglas][::-1]; lifts = [x["lift"] for x in reglas][::-1]
    a1.barh(nombres, lifts, color=[AZUL if l >= 1 else ROJO for l in lifts], height=.55)
    a1.axvline(1, color=TINTA2, lw=1, ls="--"); a1.text(1.03, len(lifts) - .45, "lift = 1: igual que el azar", color=TINTA2, fontsize=8.5)
    for y, l in enumerate(lifts): a1.text(l + .04, y, f"{l:.2f}", va="center", fontsize=9)
    a1.set_xlim(0, 2.9); a1.set_title("1 · Apriori: lift de las reglas", loc="left", fontweight="bold")
    for s in ("top", "right"): a1.spines[s].set_visible(False)
    sim = r["2_vecindario"]["similitud"]; pun = r["2_vecindario"]["puntaje"]
    M = [[sim[c] if c != "Beto" else np.nan for c in CLIENTES]]
    mapa(a2, M, ["coseno con Beto"], CLIENTES, escala=1, titulo="2 · Vecindario: ¿quién se parece a Beto?")
    a2.set_aspect(.9)
    fig.text(0.56, 0.06, "Puntaje para Beto: " + ", ".join(f"{p} {v:.2f}" for p, v in pun.items() if p not in ("Pan", "Mantequilla")),
             fontsize=9, color=TINTA2)
    return _guardar(fig, "02_apriori_vecindario.png")


# ------------------------------------------------------------------ 3. factorizaciones
def factorizaciones(r):
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.4))
    casos = [("3_svd", "3 · SVD (k = 2): los huecos se llenan"), ("4_funk", "4 · SVD de Funk: colapsa, todo ≈ 1"),
             ("5_als", "5 · ALS explícito: mismo colapso"), ("6_als_implicit", "6 · ALS implicit: acierta y ordena")]
    real = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 1, 1]], bool)
    for ax, (k, t) in zip(axs.flat, casos):
        mapa(ax, r[k]["reconstruida"], CLIENTES, PROD6, escala=1.0, resaltar=1, comprados=real.tolist(), titulo=t)
    fig.text(0.01, -0.01, "Cada matriz es la tabla de compras reconstruida por el modelo. Gris en cursiva: compra real. Celdas de color: predicción para un producto no comprado.\n"
             "La fila de Beto está enmarcada: lo deseable es azul en Leche y Cereal y rojo en Pañales y Toallitas.", fontsize=9, color=TINTA2, va="top")
    fig.tight_layout()
    return _guardar(fig, "03_factorizaciones.png")


# ------------------------------------------------------------------ 4. cold-start
def cold_start(r):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 2.9))
    for ax, k, t in ((a1, "7_contenido", "7 · Contenido (coseno)"), (a2, "8_two_tower", "8 · Two-Tower (probabilidad)")):
        M = [[r[k][c][p] for p in PROD7] for c in ("beto", "fede")]
        comp = [[p in ("Pan", "Mantequilla") for p in PROD7], [p == "Pañales" for p in PROD7]]
        mapa(ax, M, ["Beto", "Fede (nuevo)"], PROD7, escala=1.0, comprados=comp, titulo=t)
    fig.text(0.01, -0.06, "Avena nunca se vendió y Fede solo compró Pañales. El contenido usa solo atributos; Two-Tower además aprende del comportamiento\n"
             "de 200 clientes simulados: por eso a Fede le sube la Leche (0.41 → 0.63).", fontsize=9, color=TINTA2, va="top")
    fig.tight_layout()
    return _guardar(fig, "04_cold_start.png")


# ------------------------------------------------------------------ 5. secuencias
def secuencias(r):
    fig = plt.figure(figsize=(12, 5.2)); g = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=.55)
    carritos = list(r["9_gru4rec"]["siguiente"].keys())
    for i, (k, t) in enumerate((("9_gru4rec", "9 · GRU4Rec: siguiente producto"), ("10_sasrec", "10 · SASRec: siguiente producto"))):
        ax = fig.add_subplot(g[i, 0])
        M = [[r[k]["siguiente"][c][p] for p in PROD6] for c in carritos]
        mapa(ax, M, carritos, PROD6, escala=1.0, titulo=t)
    ax = fig.add_subplot(g[:, 1])
    filas, y = [], 0
    for c in carritos[::-1]:
        for prod, w in list(r["10_sasrec"]["atencion"][c].items())[::-1]:
            ax.barh(y, w, color=AZUL, height=.6); ax.text(w + .02, y, f"{w:.0%}", va="center", fontsize=9)
            filas.append(f"{prod}"); y += 1
        ax.text(-0.02, y - .3, c, fontsize=9, fontweight="bold", ha="left", transform=ax.get_yaxis_transform())
        y += .8
    ax.set_yticks([i for i in range(int(y)) if i < len(filas) + 3][:0]); ax.set_xlim(0, 1); ax.set_xticks([])
    pos = []; yy = 0
    for c in carritos[::-1]:
        n = len(r["10_sasrec"]["atencion"][c]); pos += [yy + i for i in range(n)]; yy += n + .8
    ax.set_yticks(pos, filas); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("10 · SASRec: ¿a qué miró?", loc="left", fontweight="bold", pad=22)
    fig.text(0.01, 0.0, "Dos carritos que terminan en Leche reciben respuestas distintas: el modelo recuerda lo que vino antes.\n"
             "Entrenados con 600 carritos simulados.", fontsize=9, color=TINTA2, va="top")
    return _guardar(fig, "05_secuencias.png")


def todas():
    r = cargar()
    return [f(r) for f in (resumen_beto, apriori_vecindario, factorizaciones, cold_start, secuencias)]


if __name__ == "__main__":
    for f in todas(): plt.close(f)
    print("\n".join(sorted(p.name for p in FIG.glob("*.png"))))
