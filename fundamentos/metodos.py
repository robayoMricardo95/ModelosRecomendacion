"""Los 10 métodos de recomendación aplicados a la misma pregunta: ¿qué le recomendamos a Beto?

Ejecutar:  python generar_datos.py && python metodos.py
Escribe resultados/resultados.json y resultados/resultados.md.
Dependencias: numpy, pandas, autograd (redes pequeñas sin necesidad de PyTorch).
"""
from itertools import permutations
from pathlib import Path
import json

import numpy as onp
import pandas as pd
import autograd.numpy as np
from autograd import grad

BASE = Path(__file__).parent
DATOS, SALIDA = BASE / "datos", BASE / "resultados"
CLIENTES = ["Ana", "Beto", "Caro", "Dani", "Eva"]
PROD6 = ["Pan", "Mantequilla", "Leche", "Cereal", "Pañales", "Toallitas"]
PROD7 = PROD6 + ["Avena"]
BETO = 1


# ---------------------------------------------------------------- utilidades
def cargar():
    compras = pd.read_csv(DATOS / "compras.csv")
    C = (compras.pivot_table(index="cliente", columns="producto", values="veces", fill_value=0)
         .reindex(index=CLIENTES, columns=PROD6, fill_value=0).to_numpy(float))
    attrs = pd.read_csv(DATOS / "productos.csv").set_index("producto").loc[PROD7]
    A = attrs[["Desayuno", "Lacteo", "Granos", "Bebe", "Higiene"]].to_numpy(float)
    return C, (C > 0).astype(float), A


def coseno(a, b):
    return float(a @ b / (onp.linalg.norm(a) * onp.linalg.norm(b) + 1e-12))


def adam(f_perdida, params, pasos, lr, extra=lambda: (), marcas=()):
    """Optimizador Adam genérico. Devuelve parámetros y la pérdida en las vueltas marcadas."""
    g = grad(f_perdida)
    m = [0 * p for p in params]; v = [0 * p for p in params]; historia = {}
    for t in range(1, pasos + 1):
        gr = g(params, *extra())
        for k in range(len(params)):
            m[k] = .9 * m[k] + .1 * gr[k]; v[k] = .999 * v[k] + .001 * gr[k] ** 2
            params[k] = params[k] - lr * (m[k] / (1 - .9 ** t)) / (onp.sqrt(v[k] / (1 - .999 ** t)) + 1e-8)
        if t in marcas:
            historia[t] = round(float(f_perdida(params, *extra())), 3)
    return params, historia


def softmax(z, eje=-1):
    z = z - np.max(z, axis=eje, keepdims=True); e = np.exp(z)
    return e / np.sum(e, axis=eje, keepdims=True)


def fila(v, productos=PROD6):
    return {p: round(float(x), 2) for p, x in zip(productos, v)}


# ---------------------------------------------------------------- 1. Apriori
def apriori(B, soporte_min=0.4):
    """Reglas A → B con soporte, confianza y lift sobre las canastas (una por cliente)."""
    sop = B.mean(0); reglas = []
    for a, b in permutations(range(6), 2):
        s = (B[:, a] * B[:, b]).mean()
        if s >= soporte_min:
            conf = s / sop[a]
            reglas.append({"regla": f"{PROD6[a]} → {PROD6[b]}", "soporte": round(s, 2),
                           "confianza": round(conf, 2), "lift": round(conf / sop[b], 2)})
    return sorted(reglas, key=lambda r: -r["lift"])


# ---------------------------------------------------------------- 2. Vecindario
def vecindario(B, u=BETO):
    """Similitud coseno entre clientes y promedio ponderado de lo que compraron los vecinos."""
    sim = onp.array([coseno(B[u], B[j]) for j in range(len(B))])
    w = sim.copy(); w[u] = 0
    return {"similitud": dict(zip(CLIENTES, sim.round(2).tolist())), "puntaje": fila(w @ B / w.sum())}


# ---------------------------------------------------------------- 3. SVD
def svd(B, k=2):
    """Descomposición R = U Σ Vᵀ, conservando los k patrones más fuertes."""
    U, s, Vt = onp.linalg.svd(B, full_matrices=False)
    R = U[:, :k] @ onp.diag(s[:k]) @ Vt[:k]
    return {"valores_singulares": s.round(2).tolist(), "reconstruida": R, "beto": fila(R[BETO])}


# ---------------------------------------------------------------- 4. SVD de Funk
def funk(B, rng, k=2, epocas=2000, paso=.05, lam=.02):
    """Descenso de gradiente solo sobre las celdas conocidas (aquí: solo los unos)."""
    X = rng.normal(0, .1, (5, k)); Y = rng.normal(0, .1, (6, k))
    conocidas = [(u, i) for u in range(5) for i in range(6) if B[u, i] > 0]
    for _ in range(epocas):
        for u, i in conocidas:
            e = 1 - X[u] @ Y[i]; xu = X[u].copy()
            X[u] += paso * (e * Y[i] - lam * X[u]); Y[i] += paso * (e * xu - lam * Y[i])
    R = X @ Y.T
    return {"reconstruida": R, "beto": fila(R[BETO])}


# ---------------------------------------------------------------- 5. ALS (explícito)
def als(B, rng, k=2, lam=.1, vueltas=15):
    """Mismo objetivo que Funk, resuelto por turnos con regresión exacta (solo celdas conocidas)."""
    X = rng.normal(0, .1, (5, k)); Y = rng.normal(0, .1, (6, k)); M = B > 0
    for _ in range(vueltas):
        for u in range(5):
            Yu = Y[M[u]]; X[u] = onp.linalg.solve(Yu.T @ Yu + lam * onp.eye(k), Yu.T @ B[u, M[u]])
        for i in range(6):
            Xi = X[M[:, i]]; Y[i] = onp.linalg.solve(Xi.T @ Xi + lam * onp.eye(k), Xi.T @ B[M[:, i], i])
    R = X @ Y.T
    return {"reconstruida": R, "beto": fila(R[BETO])}


# ---------------------------------------------------------------- 6. ALS implicit
def als_implicit(C, B, rng, k=2, alfa=10, lam=.1, vueltas=15):
    """Hu, Koren y Volinsky (2008): todas las celdas cuentan, ponderadas por confianza c = 1 + α·veces."""
    Cf = 1 + alfa * C
    X = rng.normal(0, .1, (5, k)); Y = rng.normal(0, .1, (6, k)); perdida = {}
    for t in range(vueltas):
        for u in range(5):
            Cu = onp.diag(Cf[u]); X[u] = onp.linalg.solve(Y.T @ Cu @ Y + lam * onp.eye(k), Y.T @ Cu @ B[u])
        for i in range(6):
            Ci = onp.diag(Cf[:, i]); Y[i] = onp.linalg.solve(X.T @ Ci @ X + lam * onp.eye(k), X.T @ Ci @ B[:, i])
        if t + 1 in (1, 2, 3, 5, 15):
            perdida[t + 1] = round(float((Cf * (B - X @ Y.T) ** 2).sum() + lam * ((X ** 2).sum() + (Y ** 2).sum())), 1)
    R = X @ Y.T
    return {"reconstruida": R, "beto": fila(R[BETO]), "perdida": perdida, "X": X, "Y": Y}


# ---------------------------------------------------------------- 7. Basado en contenido
def contenido(B, A):
    """Perfil del cliente = promedio de atributos de lo que compró; puntaje = coseno con cada producto."""
    def recomendar(historial):
        perfil = historial @ A[:6] / historial.sum()
        return fila([coseno(perfil, A[i]) for i in range(7)], PROD7), perfil
    beto, perfil = recomendar(B[BETO])
    fede, _ = recomendar(onp.eye(6)[4])  # Fede: cliente nuevo que solo compró Pañales
    return {"perfil_beto": perfil.round(2).tolist(), "beto": beto, "fede": fede}


# ---------------------------------------------------------------- 8. Two-Tower
def two_tower(B, A, semilla=7, oculta=16, dim=3, pasos=2000):
    """Torre del cliente (perfil + productos comprados) y torre del producto (atributos + ID).
    Se entrena con los 5 clientes del ejemplo + 200 simulados; en cada paso se oculta la mitad
    del historial de entrada para que aprenda a recomendar con poca historia."""
    rng = onp.random.default_rng(semilla)
    sim = pd.read_csv(DATOS / "clientes_simulados.csv")
    Bsim = (pd.crosstab(sim["cliente"], sim["producto"]).reindex(columns=PROD6, fill_value=0) > 0)
    Bm = onp.vstack([B, Bsim.to_numpy(float)])

    def entrada_cliente(r):
        return onp.concatenate([r @ A[:6] / max(r.sum(), 1), r / max(r.sum(), 1)])

    def entrada_producto(i):
        ident = onp.zeros(6)
        if i < 6: ident[i] = 1  # la Avena no tiene ID entrenado: solo atributos
        return onp.concatenate([A[i], ident])

    Xu = onp.stack([entrada_cliente(r) for r in Bm]); Xi = onp.stack([entrada_producto(i) for i in range(7)])
    torre = lambda x, W1, b1, W2: np.dot(np.maximum(0, np.dot(x, W1) + b1), W2)

    def historial_parcial():
        M = (rng.random(Bm.shape) < .5) * Bm
        for i in range(len(M)):
            if M[i].sum() == 0: M[i, rng.choice(onp.flatnonzero(Bm[i]))] = 1
        return (onp.stack([entrada_cliente(r) for r in M]),)

    def perdida(p, Xin):
        s = np.dot(torre(Xin, *p[:3]), torre(Xi[:6], *p[3:6]).T) + p[6]
        bce = (np.logaddexp(0, -s) * Bm + np.logaddexp(0, s) * (1 - Bm)).mean()
        return bce + 1e-3 * sum((q ** 2).sum() for q in p[:6])

    p = [rng.normal(0, .3, (11, oculta)), onp.zeros(oculta) + .1, rng.normal(0, .3, (oculta, dim)),
         rng.normal(0, .3, (11, oculta)), onp.zeros(oculta) + .1, rng.normal(0, .3, (oculta, dim)), onp.zeros(1)]
    p, hist = adam(perdida, p, pasos, .02, historial_parcial, marcas=(1, 100, 500, pasos))
    v = torre(Xi, *p[3:6]); sig = lambda s: 1 / (1 + onp.exp(-s))
    prob = lambda r: sig(torre(entrada_cliente(r)[None], *p[:3]) @ v.T + p[6])[0]
    return {"perdida": hist, "beto": fila(prob(B[BETO]), PROD7), "fede": fila(prob(onp.eye(6)[4]), PROD7)}


# ---------------------------------------------------------------- 9 y 10. Secuencias
def cargar_carritos(L=4):
    df = pd.read_csv(DATOS / "carritos_simulados.csv").sort_values(["carrito", "paso"])
    seqs = [[PROD6.index(x) for x in g["producto"]] for _, g in df.groupby("carrito")]
    n = len(seqs); X = onp.zeros((n, L, 6)); Y = onp.zeros((n, L, 6)); M = onp.zeros((n, L))
    for s, sec in enumerate(seqs):
        for k, it in enumerate(sec):
            X[s, k, it] = 1
            if k + 1 < len(sec): Y[s, k, sec[k + 1]] = 1; M[s, k] = 1
    return X, Y, M


def one_hot(carrito):
    x = onp.zeros((1, len(carrito), 6))
    for k, c in enumerate(carrito): x[0, k, PROD6.index(c)] = 1
    return x


CARRITOS = [["Pan", "Cereal"], ["Pan", "Leche"], ["Pañales", "Leche"]]


def gru4rec(semilla=3, d=8, H=12, pasos=400):
    """Red recurrente GRU: lee el carrito paso a paso y predice el siguiente producto."""
    X, Y, M = cargar_carritos(); rng = onp.random.default_rng(semilla + 100)
    sg = lambda z: 1 / (1 + np.exp(-z))

    def estados(p, Xo):
        E, Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh, Wo, bo = p
        h = np.zeros((Xo.shape[0], H)); hs = []
        for k in range(Xo.shape[1]):
            x = np.dot(Xo[:, k], E)
            z = sg(np.dot(x, Wz) + np.dot(h, Uz) + bz)          # cuánto actualizar
            r = sg(np.dot(x, Wr) + np.dot(h, Ur) + br)          # cuánto olvidar
            hc = np.tanh(np.dot(x, Wh) + np.dot(r * h, Uh) + bh)  # memoria candidata
            h = (1 - z) * h + z * hc; hs.append(h)
        return hs

    def perdida(p):
        hs = estados(p, X); l = 0
        for k in range(X.shape[1]):
            pr = softmax(np.dot(hs[k], p[10]) + p[11])
            l = l - np.sum(M[:, k] * np.sum(Y[:, k] * np.log(pr + 1e-9), 1))
        return l / M.sum() + 1e-4 * sum((q ** 2).sum() for q in p)

    p = ([rng.normal(0, .3, (6, d))] + [rng.normal(0, .3, (d, H)) for _ in range(3)]
         + [rng.normal(0, .3, (H, H)) for _ in range(3)] + [onp.zeros(H) for _ in range(3)]
         + [rng.normal(0, .3, (H, 6)), onp.zeros(6)])
    p, hist = adam(perdida, p, pasos, .03, marcas=(1, 50, 200, pasos))
    pred = {" → ".join(c): fila(softmax(np.dot(estados(p, one_hot(c))[-1], p[10]) + p[11])[0]) for c in CARRITOS}
    return {"perdida": hist, "siguiente": pred}


def sasrec(semilla=3, D=12, L=4, pasos=400):
    """Self-attention con máscara causal (una capa, una cabeza) y embeddings de posición."""
    X, Y, M = cargar_carritos(L); rng = onp.random.default_rng(semilla + 200)
    causal = onp.triu(onp.ones((L, L)), 1) * -1e9

    def modelo(p, Xo):
        E, Pos, Wq, Wk, Wv, W1, b1, W2, b2 = p; n = Xo.shape[1]
        x = np.dot(Xo, E) + Pos[:n]
        q, k, v = np.dot(x, Wq), np.dot(x, Wk), np.dot(x, Wv)
        att = softmax(np.einsum("bid,bjd->bij", q, k) / onp.sqrt(D) + causal[:n, :n])
        h = x + np.einsum("bij,bjd->bid", att, v)
        h = h + np.dot(np.maximum(0, np.dot(h, W1) + b1), W2) + b2
        return np.dot(h, E.T), att

    def perdida(p):
        pr = softmax(modelo(p, X)[0])
        return -np.sum(M[:, :, None] * Y * np.log(pr + 1e-9)) / M.sum() + 1e-4 * sum((q ** 2).sum() for q in p)

    p = [rng.normal(0, .3, (6, D)), rng.normal(0, .3, (L, D))] + [rng.normal(0, .3, (D, D)) for _ in range(4)] \
        + [onp.zeros(D), rng.normal(0, .3, (D, D)), onp.zeros(D)]
    p, hist = adam(perdida, p, pasos, .03, marcas=(1, 50, 200, pasos))
    pred, aten = {}, {}
    for c in CARRITOS:
        lg, att = modelo(p, one_hot(c)); nombre = " → ".join(c)
        pred[nombre] = fila(softmax(lg[0, -1])); aten[nombre] = dict(zip(c, att[0, -1].round(2).tolist()))
    return {"perdida": hist, "siguiente": pred, "atencion": aten}


# ---------------------------------------------------------------- ejecución
def main():
    C, B, A = cargar()
    rng = onp.random.default_rng(0)  # Funk y ALS implicit comparten este generador, en este orden
    r = {"1_apriori": apriori(B), "2_vecindario": vecindario(B), "3_svd": svd(B),
         "4_funk": funk(B, rng)}
    r["5_als"] = als(B, onp.random.default_rng(1))
    r["6_als_implicit"] = als_implicit(C, B, rng)
    r["7_contenido"] = contenido(B, A)
    r["8_two_tower"] = two_tower(B, A)
    r["9_gru4rec"] = gru4rec()
    r["10_sasrec"] = sasrec()

    SALIDA.mkdir(exist_ok=True)
    limpio = json.loads(json.dumps(r, default=lambda o: onp.round(o, 2).tolist()))
    (SALIDA / "resultados.json").write_text(json.dumps(limpio, ensure_ascii=False, indent=1))
    for k, v in limpio.items():
        print(f"\n== {k}")
        for kk in ("beto", "fede", "siguiente", "atencion", "perdida", "puntaje", "similitud", "valores_singulares", "perfil_beto"):
            if kk in v: print(f"  {kk}: {v[kk]}")
        if k == "1_apriori":
            for regla in v: print("  ", regla)


if __name__ == "__main__":
    main()
