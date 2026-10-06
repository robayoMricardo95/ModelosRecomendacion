"""Genera la base de datos de juguete del supermercado.

Archivos que crea en datos/:
  productos.csv            7 productos con 5 atributos (Avena es nueva: nadie la compró)
  compras.csv              5 clientes reales del ejemplo: cuántas veces compró cada producto
  clientes_simulados.csv   200 clientes simulados (120 de desayuno, 80 de bebé) para Two-Tower
  carritos_simulados.csv   600 carritos ordenados (con 30% de ruido) para GRU4Rec y SASRec

Todo usa semillas fijas, así que los resultados son reproducibles.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATOS = Path(__file__).parent / "datos"
PRODUCTOS = ["Pan", "Mantequilla", "Leche", "Cereal", "Pañales", "Toallitas", "Avena"]
ATRIBUTOS = ["Desayuno", "Lacteo", "Granos", "Bebe", "Higiene"]


def productos() -> pd.DataFrame:
    filas = [[1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 1, 0], [1, 0, 1, 0, 0],
             [0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [1, 0, 1, 0, 0]]
    df = pd.DataFrame(filas, columns=ATRIBUTOS)
    df.insert(0, "producto", PRODUCTOS)
    df["es_nuevo"] = df["producto"].eq("Avena").astype(int)
    return df


def compras() -> pd.DataFrame:
    conteos = {"Ana": [5, 3, 4, 2, 0, 0], "Beto": [4, 2, 0, 0, 0, 0], "Caro": [0, 0, 3, 0, 6, 4],
               "Dani": [2, 0, 3, 3, 0, 0], "Eva": [0, 0, 1, 0, 2, 3]}
    filas = [(c, p, v) for c, vs in conteos.items() for p, v in zip(PRODUCTOS[:6], vs) if v > 0]
    return pd.DataFrame(filas, columns=["cliente", "producto", "veces"])


def clientes_simulados(n_desayuno=120, n_bebe=80, semilla=7) -> pd.DataFrame:
    """Cada cliente compra cada producto con una probabilidad que depende de su segmento."""
    rng = np.random.default_rng(semilla)
    prob = {"desayuno": [.9, .6, .7, .6, .05, .05], "bebe": [.2, .1, .8, .15, .9, .85]}
    filas = []
    for i in range(n_desayuno + n_bebe):
        seg = "desayuno" if i < n_desayuno else "bebe"
        compro = rng.random(6) < prob[seg]
        if not compro.any():
            compro[2] = True  # nadie queda vacío: al menos Leche
        filas += [(f"sim_{i:03d}", seg, p) for p, c in zip(PRODUCTOS[:6], compro) if c]
    return pd.DataFrame(filas, columns=["cliente", "segmento", "producto"])


def carritos_simulados(n=600, ruido=0.3, semilla=3) -> pd.DataFrame:
    """Carritos en el orden en que se agregan los productos, a partir de plantillas de compra."""
    rng = np.random.default_rng(semilla)
    plantillas = [(["Pan", "Mantequilla", "Cereal", "Leche"], .24), (["Pan", "Cereal", "Leche"], .18),
                  (["Cereal", "Leche", "Pan"], .09), (["Pan", "Leche", "Cereal"], .09),
                  (["Pañales", "Leche", "Toallitas"], .20), (["Leche", "Pañales", "Toallitas"], .10),
                  (["Pañales", "Toallitas", "Leche"], .10)]
    p = np.array([w for _, w in plantillas]); p /= p.sum()
    filas = []
    for s in range(n):
        carrito = list(plantillas[rng.choice(len(plantillas), p=p)][0])
        if rng.random() < ruido:  # un producto al azar reemplaza a uno (nunca al primero)
            carrito[rng.integers(1, len(carrito))] = PRODUCTOS[rng.integers(6)]
        filas += [(s, k + 1, prod) for k, prod in enumerate(carrito)]
    return pd.DataFrame(filas, columns=["carrito", "paso", "producto"])


if __name__ == "__main__":
    DATOS.mkdir(exist_ok=True)
    for nombre, df in [("productos", productos()), ("compras", compras()),
                       ("clientes_simulados", clientes_simulados()), ("carritos_simulados", carritos_simulados())]:
        df.to_csv(DATOS / f"{nombre}.csv", index=False)
        print(f"{nombre}.csv: {len(df)} filas")
