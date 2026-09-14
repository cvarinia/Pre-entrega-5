"""Contrato de herramientas (Fase 1 de la consigna).

Cada funcion decorada con `@tool` es una herramienta que el LLM puede decidir
usar por su cuenta. El modelo NO ve este codigo: solo ve el nombre de la
funcion, la firma con sus type hints y el docstring. Por eso los docstrings
estan escritos como si fueran documentacion para otra persona: dicen cuando
usar la herramienta, que devuelve y con que encadenarla despues.

Dos decisiones de diseno importantes:

1. Las herramientas NUNCA lanzan excepciones hacia el grafo. Cuando algo sale
   mal devuelven un diccionario con las claves "error" y "sugerencia". Eso es lo
   que habilita el ciclo de retorno: el modelo lee el error como si fuera un
   resultado mas, y decide reintentar de otra forma o pedir aclaraciones.
2. `buscar_recetas_por_ingrediente` devuelve solo identificadores y nombres, sin
   calorias ni etiquetas. Esa informacion incompleta a proposito obliga al
   agente a encadenar una segunda llamada (razonamiento multi-paso real, no
   simulado con if/else).
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from agente.repositorio import obtener_repositorio


@tool
async def buscar_recetas_por_ingrediente(
    ingrediente: str, max_resultados: int = 5
) -> dict[str, Any]:
    """Busca recetas en la base de datos que contengan un ingrediente determinado.

    Usa esta herramienta cuando la persona menciona un alimento, una proteina o
    un producto que tiene en la heladera y quiere saber que puede cocinar con
    eso. Por ejemplo: "algo con pollo", "tengo lentejas", "recetas con papa".

    IMPORTANTE: esta herramienta devuelve unicamente el identificador, el nombre,
    el tiempo de preparacion y las porciones base de cada receta. NO devuelve
    ingredientes completos, ni calorias, ni informacion sobre si tiene lacteos o
    gluten. Para cualquiera de esos datos tenes que llamar despues a
    `obtener_receta` con el id de la receta que te interese.

    Args:
        ingrediente: Nombre del ingrediente a buscar, en singular y sin acentos
            necesarios (por ejemplo "pollo", "papa", "queso"). Un solo
            ingrediente por llamada.
        max_resultados: Cantidad maxima de recetas a devolver. Por defecto 5.

    Returns:
        Un diccionario con "cantidad" y "recetas" (lista de {id, nombre,
        tiempo_minutos, porciones_base}). Si no hay coincidencias devuelve
        "error" y "sugerencia" con la lista de ingredientes que si existen en la
        base, para que puedas reintentar con otro termino.
    """
    repo = obtener_repositorio()
    encontradas = await repo.buscar_por_ingrediente(ingrediente, limite=max_resultados)

    if not encontradas:
        disponibles = await repo.ingredientes_disponibles()
        return {
            "error": f"No hay ninguna receta con el ingrediente '{ingrediente}'.",
            "sugerencia": (
                "Reintenta con uno de estos ingredientes que si existen en la base, "
                "o preguntale a la persona por una alternativa."
            ),
            "ingredientes_disponibles": disponibles,
        }

    return {
        "cantidad": len(encontradas),
        "recetas": [
            {
                "id": receta["id"],
                "nombre": receta["nombre"],
                "tiempo_minutos": receta["tiempo_minutos"],
                "porciones_base": receta["porciones_base"],
            }
            for receta in encontradas
        ],
        "siguiente_paso": (
            "Para conocer ingredientes, calorias o restricciones alimentarias de "
            "alguna de estas recetas, llama a obtener_receta con su id."
        ),
    }


@tool
async def obtener_receta(receta_id: str) -> dict[str, Any]:
    """Devuelve la ficha completa de una receta a partir de su identificador.

    Usa esta herramienta cuando ya tenes el id de una receta (normalmente porque
    lo obtuviste de `buscar_recetas_por_ingrediente`) y necesitas conocer sus
    ingredientes, sus calorias por porcion, cuantas porciones rinde o sus
    etiquetas alimentarias.

    Las etiquetas son la unica fuente confiable para saber si una receta sirve
    para alguien con restricciones. Los valores posibles son: "contiene_lacteos",
    "sin_lacteos", "contiene_gluten", "sin_gluten", "vegetariana" y "vegana".
    Nunca deduzcas si una receta tiene lacteos o gluten leyendo el nombre:
    consulta siempre esta herramienta y mira las etiquetas.

    Podes llamar a esta herramienta varias veces seguidas, una por cada receta
    candidata, cuando necesites comparar opciones entre si.

    Args:
        receta_id: Identificador exacto de la receta, con el formato "r001".

    Returns:
        Un diccionario con id, nombre, ingredientes, porciones_base,
        calorias_por_porcion, tiempo_minutos y etiquetas. Si el id no existe
        devuelve "error", "sugerencia" y algunos ids validos de ejemplo, para que
        puedas corregir la llamada o buscar de nuevo por ingrediente.
    """
    repo = obtener_repositorio()
    receta = await repo.obtener(receta_id)

    if receta is None:
        ids = await repo.ids_disponibles()
        return {
            "error": f"No existe ninguna receta con el id '{receta_id}'.",
            "sugerencia": (
                "Los ids validos tienen el formato 'r001'. Si no conoces el id, usa "
                "primero buscar_recetas_por_ingrediente para obtenerlo."
            ),
            "ids_de_ejemplo": ids[:5],
        }

    return dict(receta)


@tool
async def calcular_nutricion_menu(
    receta_ids: list[str], comensales: int = 4
) -> dict[str, Any]:
    """Calcula el aporte calorico total de un menu compuesto por una o varias recetas.

    Usa esta herramienta como paso final, cuando ya elegiste que recetas
    componen el menu y necesitas dar un numero agregado: cuantas calorias suma
    el menu completo y cuantas le corresponden a cada comensal.

    Ajusta automaticamente las porciones: si una receta rinde 4 porciones y el
    menu es para 6 comensales, escala las cantidades proporcionalmente.

    Args:
        receta_ids: Lista de identificadores de recetas ya verificados con
            `obtener_receta`, por ejemplo ["r004", "r011"].
        comensales: Cantidad de personas que van a comer. Por defecto 4.

    Returns:
        Un diccionario con el detalle por receta, "calorias_totales_menu",
        "calorias_por_comensal" y una lista de "advertencias" con los ids que no
        se pudieron encontrar. Si ningun id es valido devuelve "error".
    """
    repo = obtener_repositorio()

    if comensales < 1:
        return {
            "error": "La cantidad de comensales tiene que ser al menos 1.",
            "sugerencia": "Preguntale a la persona para cuantas personas cocina.",
        }

    detalle: list[dict[str, Any]] = []
    advertencias: list[str] = []
    total = 0.0

    for receta_id in receta_ids:
        receta = await repo.obtener(receta_id)
        if receta is None:
            advertencias.append(f"El id '{receta_id}' no existe y fue ignorado.")
            continue

        factor = comensales / receta["porciones_base"]
        calorias_receta = receta["calorias_por_porcion"] * comensales
        total += calorias_receta
        detalle.append(
            {
                "id": receta["id"],
                "nombre": receta["nombre"],
                "calorias_por_porcion": receta["calorias_por_porcion"],
                "factor_de_escala": round(factor, 2),
                "calorias_para_el_grupo": calorias_receta,
            }
        )

    if not detalle:
        return {
            "error": "Ninguno de los ids recibidos existe en la base de recetas.",
            "sugerencia": "Verifica los ids con obtener_receta antes de calcular el menu.",
            "advertencias": advertencias,
        }

    return {
        "comensales": comensales,
        "detalle": detalle,
        "calorias_totales_menu": int(total),
        "calorias_por_comensal": int(total / comensales),
        "advertencias": advertencias,
    }


# Lista unica que consumen tanto el binding del LLM como el ToolNode del grafo.
# Se define aca para que agregar una herramienta sea cambiar un solo lugar.
HERRAMIENTAS: list[Any] = [
    buscar_recetas_por_ingrediente,
    obtener_receta,
    calcular_nutricion_menu,
]
