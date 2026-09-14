"""Persistencia del estado (Fase 3 de la consigna).

Un *checkpointer* guarda una foto del estado del grafo despues de cada paso. Con
el mismo `thread_id`, el grafo levanta esa foto y sigue la conversacion donde
habia quedado: eso es lo que convierte al agente de efimero en persistente.

La consigna nombra `SqliteSaver`. Como todo el proyecto es asincronico
(`asyncio`, `ainvoke`, tools `async def`), usamos su version asincronica
`AsyncSqliteSaver`, del mismo paquete `langgraph-checkpoint-sqlite`. Es el mismo
archivo .sqlite y el mismo modelo de datos; la version sincronica bloquearia el
event loop en cada escritura.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@asynccontextmanager
async def crear_checkpointer(ruta: Path) -> AsyncIterator[AsyncSqliteSaver]:
    """Abre la base SQLite de checkpoints y la cierra al salir del bloque.

    Se usa como context manager para garantizar que la conexion se cierre aunque
    la corrida termine con error:

        async with crear_checkpointer(ruta) as saver:
            grafo = construir_grafo(config, saver)

    Args:
        ruta: Archivo .sqlite donde se guardan los checkpoints. Se crea solo si
            no existe. Esta en .gitignore: es estado local, no codigo.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(ruta)) as saver:
        yield saver
