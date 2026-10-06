"""Construye y ejecuta recorrido.ipynb (para que GitHub lo muestre con sus salidas)."""
import nbformat as nbf
from nbclient import NotebookClient

M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def metodo(num, nombre, que, mejora, limite, codigo):
    return [M(f"## {num}. {nombre}\n\n| | |\n|---|---|\n| **Qué hace** | {que} |\n| **Cómo mejora** | {mejora} |\n| **Su limitación** | {limite} |"),
            C(codigo)]


celdas = [
    M("# ¿Qué le recomendamos a Beto?\n\nUn solo supermercado de juguete recorrido por 10 métodos de recomendación. Cada método resuelve "
      "una falla del anterior. Todos los números salen de `metodos.py`; los datos, de `generar_datos.py`.\n\n"
      "**Respuesta esperada para Beto** (compra Pan y Mantequilla, como Ana y Dani): Leche y Cereal, nunca Pañales."),
    C("import json, numpy as np, pandas as pd\nimport generar_datos, metodos, figuras\n"
      "pd.set_option('display.precision', 2)\n\n"
      "C, B, A = metodos.cargar()\n"
      "pd.DataFrame(C.astype(int), index=metodos.CLIENTES, columns=metodos.PROD6)  # veces que compró cada producto"),
    C("pd.read_csv('datos/productos.csv').set_index('producto')  # atributos; Avena es nueva (nadie la compró)"),
    M("### El resultado completo en una imagen"),
    C("import contextlib, io\nwith contextlib.redirect_stdout(io.StringIO()):\n    metodos.main()  # corre los 10 métodos y guarda resultados/resultados.json\n"
      "res = figuras.cargar()\nfiguras.resumen_beto(res);"),
]
celdas += metodo(1, "Apriori", "Cuenta qué productos aparecen juntos en las canastas y arma reglas A → B con soporte, confianza y lift.",
                 "Punto de partida: describe la tienda sin ningún modelo.", "No personaliza: a todos los que llevan pan les dice lo mismo.",
                 "pd.DataFrame(metodos.apriori(B))")
celdas += metodo(2, "Vecindario (filtrado colaborativo)", "Compara la canasta de Beto con la de cada cliente (coseno) y promedia lo que compraron sus vecinos.",
                 "Personaliza con el historial de cada cliente.", "Similitud 0 entre clientes sin compras en común; costo al cuadrado.",
                 "v = metodos.vecindario(B)\ndisplay(pd.Series(v['similitud'], name='coseno con Beto').to_frame().T)\npd.Series(v['puntaje'], name='puntaje Beto').to_frame().T")
celdas += [C("figuras.apriori_vecindario(res);")]
celdas += metodo(3, "SVD", "Descompone la matriz en patrones ocultos y la reconstruye con los k más fuertes.",
                 "Conecta clientes de forma indirecta; nacen los embeddings.", "Trata lo no comprado como 'no le gusta'.",
                 "s = metodos.svd(B)\nprint('valores singulares:', s['valores_singulares'])\npd.DataFrame(s['reconstruida'], index=metodos.CLIENTES, columns=metodos.PROD6)")
celdas += metodo(4, "SVD de Funk", "Aprende los embeddings con descenso de gradiente solo sobre las celdas conocidas.",
                 "Ya no castiga los huecos.", "Con compras solo hay unos: colapsa y predice ≈ 1 en todo.",
                 "f = metodos.funk(B, np.random.default_rng(0))\npd.DataFrame(f['reconstruida'], index=metodos.CLIENTES, columns=metodos.PROD6)")
celdas += metodo(5, "ALS", "Mismo objetivo que Funk, resuelto por turnos con regresión exacta.",
                 "Rápido y paralelizable.", "Mismo colapso: cambia el cómo, no el qué.",
                 "a = metodos.als(B, np.random.default_rng(1))\npd.DataFrame(a['reconstruida'], index=metodos.CLIENTES, columns=metodos.PROD6)")
celdas += metodo(6, "ALS implicit", "Usa todas las celdas con una confianza c = 1 + α·veces: los huecos son un 'no' débil.",
                 "Primer método que acierta y ordena para Beto.", "No conoce productos ni clientes nuevos; ignora atributos y orden.",
                 "rng = np.random.default_rng(0); metodos.funk(B, rng)  # mismo orden de semillas que metodos.main\n"
                 "ai = metodos.als_implicit(C, B, rng)\nprint('pérdida por vuelta:', ai['perdida'])\n"
                 "pd.DataFrame(ai['reconstruida'], index=metodos.CLIENTES, columns=metodos.PROD6)")
celdas += [C("figuras.factorizaciones(res);")]
celdas += metodo(7, "Basado en contenido", "Perfil del cliente = promedio de atributos de lo que compró; recomienda los productos más parecidos.",
                 "Recomienda la Avena sin una sola venta.", "Solo 'más de lo mismo'; no aprende del comportamiento de otros.",
                 "c = metodos.contenido(B, A)\npd.DataFrame({'Beto': c['beto'], 'Fede (nuevo)': c['fede']}).T")
celdas += metodo(8, "Two-Tower", "Dos redes (cliente y producto) producen embeddings; P(compra) = σ(u·v). Entrenada con 200 clientes simulados.",
                 "Une comportamiento y atributos: resuelve producto y cliente nuevos.", "No ve el orden ni el momento de la compra.",
                 "t = metodos.two_tower(B, A)\nprint('pérdida:', t['perdida'])\npd.DataFrame({'Beto': t['beto'], 'Fede (nuevo)': t['fede']}).T")
celdas += [C("figuras.cold_start(res);")]
celdas += metodo(9, "GRU4Rec", "Lee el carrito en orden con una memoria (GRU) y predice el siguiente producto. 600 carritos simulados.",
                 "Responde '¿qué agrega ahora?'.", "Memoria que se diluye en carritos largos; entrenamiento secuencial.",
                 "g = metodos.gru4rec()\npd.DataFrame(g['siguiente']).T")
celdas += metodo(10, "SASRec", "Atención: en cada paso mira directamente a todos los productos anteriores y los pondera.",
                 "Conecta pasos lejanos, entrena en paralelo, pesos inspeccionables.", "Mucho dato; la versión base solo usa IDs.",
                 "sa = metodos.sasrec()\ndisplay(pd.DataFrame(sa['siguiente']).T)\npd.DataFrame(sa['atencion']).T")
celdas += [C("figuras.secuencias(res);"),
           M("## Conclusión\n\nLos métodos 1 a 6 responden *¿qué le gusta a Beto?*; el 7 y el 8 agregan *¿y lo nuevo?*; el 9 y el 10 responden "
             "*¿qué hará ahora?*. En producción se combinan: uno genera candidatos (ALS o Two-Tower) y otro los ordena según la sesión (GRU4Rec o SASRec).")]

nb = nbf.v4.new_notebook(cells=celdas, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
nbf.write(nb, "recorrido.ipynb")
print("recorrido.ipynb listo")
