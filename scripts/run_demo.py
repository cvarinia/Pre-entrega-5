"""Prueba de ejecucion con razonamiento multi-paso, memoria y ciclo de error.

Corre tres escenarios seguidos SOBRE EL MISMO thread_id y guarda todo en
logs/traza_ejemplo.json. Cada escenario demuestra uno de los criterios de
aceptacion de la consigna:

  1. Multi-paso  -> el agente encadena 3 herramientas para una sola pregunta.
  2. Memoria     -> la segunda pregunta no repite el contexto y el agente igual
                    sabe de que cena estamos hablando (resiliencia de estado).
  3. Ciclo de error -> se pide una receta con un id inexistente; la herramienta
                    devuelve un error estructurado y el agente reintenta solo.

Uso:  python -m scripts.run_demo
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Permite ejecutar el archivo directamente (boton "Run" de VS Code) ademas de
# con `python -m scripts.run_demo`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agente.config import Configuracion  # noqa: E402
from agente.sesion import abrir_sesion  # noqa: E402
from agente.tracing import RegistroDeTraza, imprimir_pasos, mensajes_a_pasos  # noqa: E402

ESCENARIOS: list[tuple[str, str]] = [
    (
        "Razonamiento multi-paso",
        "Quiero hacer una cena con pollo para 4 personas, algo que no lleve mas "
        "de una hora. Deci cual me conviene y cuantas calorias tiene en total el menu.",
    ),
    (
        "Memoria del hilo (thread_id)",
        "Y si la quiero sin lacteos? Cambia algo de lo que me dijiste?",
    ),
    (
        "Ciclo de retorno ante error",
        "Contame los ingredientes de la receta r099.",
    ),
    (
        "Reintento por otro camino",
        "Tengo tofu en la heladera, que puedo cocinar con eso?",
    ),
]


async def main() -> None:
    config = Configuracion.desde_entorno()
    # Un thread_id nuevo por corrida: asi la demo siempre arranca de cero y la
    # memoria que se ve en el escenario 2 es indiscutiblemente de esta sesion.
    thread_id = f"demo-{datetime.now():%Y%m%d-%H%M%S}"

    registro = RegistroDeTraza(
        thread_id=thread_id,
        modelo=config.modelo,
        limite_recursion=config.limite_recursion,
    )

    print("=" * 78)
    print(f"DEMO DEL AGENTE  |  modelo: {config.modelo}  |  thread_id: {thread_id}")
    print(f"recursion_limit: {config.limite_recursion}")
    print("=" * 78)

    async with abrir_sesion(config, thread_id) as sesion:
        for titulo, pregunta in ESCENARIOS:
            print(f"\n--- {titulo} ---")
            print(f"  [USUARIO] {pregunta}")

            resultado = await sesion.preguntar(pregunta)

            pasos = mensajes_a_pasos(resultado.mensajes_nuevos)
            # El primer paso es la pregunta, que ya imprimimos arriba.
            imprimir_pasos([p for p in pasos if p["tipo"] != "usuario"])

            if resultado.error:
                print(f"  [ERROR] {resultado.error}")
            else:
                print(f"  [RESPUESTA] {resultado.respuesta}")
            print(f"  ({resultado.cantidad_de_llamadas} llamadas a herramientas)")

            registro.agregar_turno(
                titulo=titulo,
                pregunta=pregunta,
                mensajes=resultado.mensajes_nuevos,
                herramientas_invocadas=resultado.herramientas_invocadas,
                respuesta=resultado.respuesta,
                error=resultado.error,
            )

        historial = await sesion.historial()

    ruta = registro.guardar(config.ruta_logs / "traza_ejemplo.json")

    total_llamadas = sum(t["cantidad_de_llamadas_a_herramientas"] for t in registro.turnos)
    print("\n" + "=" * 78)
    print(f"Mensajes acumulados en el estado del hilo: {len(historial)}")
    print(f"Llamadas a herramientas en toda la sesion: {total_llamadas}")
    print(f"Traza guardada en: {ruta}")
    if total_llamadas == 0:
        print(
            "\nATENCION: el agente no llamo a ninguna herramienta. Revisa el detalle "
            "de [ERROR] mas arriba, o corre 'python -m scripts.diagnostico' para "
            "verificar la conexion con el modelo."
        )
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
