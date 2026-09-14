"""Definicion del grafo de razonamiento ciclico (Fase 2 de la consigna).

Topologia:

        START
          |
          v
    +-----------+   tools_condition (arista condicional)
    |  modelo   |-------------------------------+
    +-----------+                               |
          ^                                     v
          |                              +--------------+
          +------------------------------|  herramientas |
                 (siempre vuelve)        +--------------+
          |
          v (cuando el modelo responde sin pedir herramientas)
         END

El ciclo `modelo -> herramientas -> modelo` es todo el razonamiento ReAct. No hay
un solo `if` que decida que herramienta usar: la eleccion la toma el modelo y el
ruteo lo resuelve `tools_condition`, que simplemente mira si el ultimo mensaje
del modelo trae `tool_calls`.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AnyMessage, SystemMessage, trim_messages
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agente.checkpointer import AsyncSqliteSaver
from agente.config import Configuracion
from agente.llm import PROMPT_SISTEMA, construir_llm_con_herramientas
from agente.reintentos import con_reintentos
from agente.state import EstadoAgente
from agente.tools import HERRAMIENTAS


def recortar_historial(
    mensajes: list[AnyMessage], max_mensajes: int
) -> list[AnyMessage]:
    """Recorta el historial para que el contexto no crezca sin limite ("estado sucio").

    El estado se acumula: cada vuelta del ciclo agrega el pedido del modelo y la
    respuesta de la herramienta. Sin recorte, una conversacion larga termina
    mandando decenas de mensajes en cada llamada.

    `start_on="human"` es la parte critica: garantiza que el recorte nunca deje
    un ToolMessage huerfano (una respuesta de herramienta sin el pedido que la
    origino), algo que los proveedores rechazan con un error de validacion.
    """
    if len(mensajes) <= max_mensajes:
        return mensajes

    return trim_messages(
        mensajes,
        max_tokens=max_mensajes,
        token_counter=len,  # contamos mensajes, no tokens: simple y predecible
        strategy="last",  # nos quedamos con lo mas reciente
        start_on="human",  # nunca cortar en medio de un ciclo herramienta/respuesta
        include_system=False,
        allow_partial=False,
    )


def construir_grafo(
    config: Configuracion,
    checkpointer: AsyncSqliteSaver,
    llm_con_herramientas: Runnable | None = None,
) -> CompiledStateGraph:
    """Arma y compila el StateGraph del agente.

    Args:
        config: Configuracion de la corrida.
        checkpointer: Persistencia del estado entre invocaciones.
        llm_con_herramientas: Modelo ya vinculado a las tools. Se puede inyectar
            uno falso para testear el grafo sin gastar llamadas a la API.

    Returns:
        El grafo compilado, listo para `ainvoke` / `astream`.
    """
    modelo = llm_con_herramientas or construir_llm_con_herramientas(config)

    async def nodo_modelo(estado: EstadoAgente) -> dict[str, list]:
        """Nodo de razonamiento: le pasa el historial al LLM y devuelve su respuesta.

        La respuesta puede ser texto final o un pedido de herramientas; este nodo
        no distingue entre los dos casos. Esa decision es de `tools_condition`.
        """
        historial = recortar_historial(
            list(estado["messages"]), config.max_mensajes_contexto
        )
        print(f"  [modelo] consultando a {config.modelo}...", flush=True)

        async def consultar_modelo() -> AnyMessage:
            """Una llamada al modelo, con techo de tiempo.

            Sin `wait_for`, una conexion que se queda colgada deja el proceso
            esperando indefinidamente y sin mensaje: parece que el agente
            "no avanza". Con el techo, el intento falla y el reintento actua.
            """
            return await asyncio.wait_for(
                modelo.ainvoke([SystemMessage(PROMPT_SISTEMA), *historial]),
                timeout=config.timeout_modelo,
            )

        # El reintento cubre los errores temporales del proveedor (cuota por
        # minuto, servicio saturado, timeouts) sin cortar el ciclo por la mitad.
        respuesta = await con_reintentos(consultar_modelo)

        # Los nombres de las herramientas pedidas se acumulan en el estado via
        # el reducer operator.add definido en EstadoAgente.
        pedidas = [llamada["name"] for llamada in getattr(respuesta, "tool_calls", [])]
        return {"messages": [respuesta], "herramientas_invocadas": pedidas}

    constructor = StateGraph(EstadoAgente)

    constructor.add_node("modelo", nodo_modelo)
    # ToolNode ejecuta las herramientas que el modelo pidio y agrega un
    # ToolMessage por cada una. Si una tool lanza una excepcion, la captura y la
    # devuelve como mensaje, para que el modelo pueda reaccionar en vez de cortar.
    constructor.add_node("herramientas", ToolNode(HERRAMIENTAS, handle_tool_errors=True))

    constructor.add_edge(START, "modelo")
    constructor.add_conditional_edges(
        "modelo",
        tools_condition,  # devuelve "tools" si hay tool_calls, "__end__" si no
        {"tools": "herramientas", END: END},
    )
    # La arista de vuelta: lo que cierra el ciclo y permite el multi-paso.
    constructor.add_edge("herramientas", "modelo")

    return constructor.compile(checkpointer=checkpointer)
