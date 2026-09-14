"""Sesion de conversacion: envuelve el grafo y le da memoria de hilo.

Este modulo es la unica puerta de entrada al agente. Tanto la CLI interactiva
como el script de demo lo usan, asi que la logica de invocacion (thread_id,
limite de recursion, extraccion de los mensajes nuevos) esta escrita una sola vez.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import Runnable
from langgraph.graph.state import CompiledStateGraph

from agente.checkpointer import crear_checkpointer
from agente.config import Configuracion
from agente.graph import construir_grafo
from agente.mensajes import texto_de_mensaje


@dataclass
class ResultadoTurno:
    """Lo que produjo una pregunta: la respuesta y el rastro del razonamiento."""

    respuesta: str
    mensajes_nuevos: list[AnyMessage]
    herramientas_invocadas: list[str]
    error: str | None = None
    """Mensaje de error si el ciclo no pudo completarse (None si salio bien)."""

    @property
    def cantidad_de_llamadas(self) -> int:
        """Cuantas veces se invoco una herramienta en este turno."""
        return len(self.herramientas_invocadas)


class SesionAgente:
    """Conversacion persistente identificada por un `thread_id`.

    Dos preguntas hechas con el mismo `thread_id` comparten historial aunque el
    proceso de Python se haya cerrado entre medio: el estado vive en SQLite, no
    en memoria.
    """

    def __init__(
        self, grafo: CompiledStateGraph, config: Configuracion, thread_id: str
    ) -> None:
        self._grafo = grafo
        self._config = config
        self.thread_id = thread_id

    @property
    def _configuracion_de_corrida(self) -> dict[str, Any]:
        """Config que recibe el grafo en cada invocacion.

        `thread_id` selecciona el hilo de memoria; `recursion_limit` pone el techo
        de pasos para que un ciclo mal cerrado no gaste llamadas a la API sin fin.
        """
        return {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": self._config.limite_recursion,
        }

    async def preguntar(self, texto: str) -> ResultadoTurno:
        """Hace una pregunta al agente y espera a que termine su ciclo de razonamiento."""
        estado_previo = await self._grafo.aget_state(self._configuracion_de_corrida)
        valores_previos = estado_previo.values if estado_previo else {}
        mensajes_previos = len(valores_previos.get("messages", []))
        tools_previas = len(valores_previos.get("herramientas_invocadas", []))

        try:
            final = await self._grafo.ainvoke(
                {"messages": [HumanMessage(texto)]},
                config=self._configuracion_de_corrida,
            )
        except Exception as error:  # noqa: BLE001 - incluye GraphRecursionError
            detalle = f"{type(error).__name__}: {error}"
            return ResultadoTurno(
                respuesta=f"El agente no pudo completar el razonamiento. {detalle}",
                mensajes_nuevos=[HumanMessage(texto)],
                herramientas_invocadas=[],
                error=detalle,
            )

        mensajes_nuevos = list(final["messages"])[mensajes_previos:]
        herramientas = list(final.get("herramientas_invocadas", []))[tools_previas:]
        respuesta = texto_de_mensaje(final["messages"][-1])

        return ResultadoTurno(
            respuesta=respuesta,
            mensajes_nuevos=mensajes_nuevos,
            herramientas_invocadas=herramientas,
        )

    async def historial(self) -> list[AnyMessage]:
        """Devuelve el historial completo guardado para este thread_id."""
        estado = await self._grafo.aget_state(self._configuracion_de_corrida)
        return list(estado.values.get("messages", [])) if estado else []


@asynccontextmanager
async def abrir_sesion(
    config: Configuracion,
    thread_id: str,
    llm_con_herramientas: Runnable | None = None,
) -> AsyncIterator[SesionAgente]:
    """Abre checkpointer + grafo + sesion, y cierra todo prolijamente al salir.

        async with abrir_sesion(config, "mi-hilo") as sesion:
            resultado = await sesion.preguntar("...")
    """
    async with crear_checkpointer(config.ruta_checkpoints) as saver:
        grafo = construir_grafo(config, saver, llm_con_herramientas)
        yield SesionAgente(grafo, config, thread_id)
