"""Esquema del estado del grafo (Fase 2 de la consigna).

En LangGraph el estado no se modifica: cada nodo devuelve un diccionario con
"lo nuevo", y LangGraph lo fusiona con el estado anterior usando un *reducer*
por cada campo.

- `messages` viene heredado de `MessagesState` y usa el reducer `add_messages`,
  que agrega los mensajes nuevos al final del historial en vez de pisarlo.
- `herramientas_invocadas` usa `operator.add`, el reducer clasico para listas:
  si el estado tenia ["buscar"] y el nodo devuelve ["obtener"], el estado
  resultante es ["buscar", "obtener"]. Lo usamos para poder mostrar, al final de
  la corrida, el recorrido completo del razonamiento.
"""

from __future__ import annotations

import operator
from typing import Annotated

from langgraph.graph import MessagesState


class EstadoAgente(MessagesState):
    """Estado del agente: el historial de mensajes mas la traza de herramientas.

    Hereda de `MessagesState`, que ya define:
        messages: Annotated[list[AnyMessage], add_messages]
    """

    herramientas_invocadas: Annotated[list[str], operator.add]
