"""Interfaz de linea de comandos para conversar con el agente.

Uso tipico (desde la raiz del proyecto):

    python -m agente.cli --thread-id cocina-lunes
    python -m agente.cli --thread-id cocina-lunes --pregunta "que puedo hacer con papa?"

Volviendo a usar el mismo --thread-id mas tarde, el agente recuerda la
conversacion anterior aunque hayas cerrado la terminal.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace

from agente.config import Configuracion
from agente.sesion import abrir_sesion
from agente.tracing import RegistroDeTraza, imprimir_pasos, mensajes_a_pasos


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agente de recetas con razonamiento ciclico y memoria persistente."
    )
    parser.add_argument(
        "--thread-id",
        default="sesion-default",
        help="Identificador del hilo de conversacion (la memoria se guarda por hilo).",
    )
    parser.add_argument(
        "--pregunta",
        default=None,
        help="Pregunta unica. Si se omite, se abre el modo conversacion.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=None,
        help="Techo de pasos del grafo (por defecto el de .env, 10).",
    )
    parser.add_argument(
        "--guardar-traza",
        default=None,
        help="Nombre de archivo dentro de logs/ para guardar la traza de la sesion.",
    )
    return parser.parse_args()


async def _ejecutar() -> None:
    argumentos = _parsear_argumentos()
    config = Configuracion.desde_entorno()

    if argumentos.recursion_limit is not None:
        # `replace` crea una copia con un campo cambiado: la config es inmutable.
        config = replace(config, limite_recursion=argumentos.recursion_limit)

    registro = RegistroDeTraza(
        thread_id=argumentos.thread_id,
        modelo=config.modelo,
        limite_recursion=config.limite_recursion,
    )

    async with abrir_sesion(config, argumentos.thread_id) as sesion:
        preguntas: list[str] = []
        if argumentos.pregunta:
            preguntas = [argumentos.pregunta]

        print(f"\nAgente de recetas | thread_id: {sesion.thread_id}")
        if not preguntas:
            print("Escribi tu consulta. 'salir' para terminar.\n")

        while True:
            if preguntas:
                consulta = preguntas.pop(0)
                print(f"\n> {consulta}")
            else:
                try:
                    consulta = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if consulta.lower() in {"salir", "exit", "quit"}:
                    break
                if not consulta:
                    continue

            resultado = await sesion.preguntar(consulta)
            pasos = mensajes_a_pasos(resultado.mensajes_nuevos)
            imprimir_pasos([p for p in pasos if p["tipo"] != "usuario"])
            if resultado.error:
                print(f"  [ERROR] {resultado.error}")
            registro.agregar_turno(
                titulo="consulta interactiva",
                pregunta=consulta,
                mensajes=resultado.mensajes_nuevos,
                herramientas_invocadas=resultado.herramientas_invocadas,
                respuesta=resultado.respuesta,
                error=resultado.error,
            )

            if argumentos.pregunta and not preguntas:
                break

    if argumentos.guardar_traza:
        ruta = registro.guardar(config.ruta_logs / argumentos.guardar_traza)
        print(f"\nTraza guardada en {ruta}")


def main() -> None:
    """Punto de entrada sincronico que arranca el event loop."""
    asyncio.run(_ejecutar())


if __name__ == "__main__":
    main()
