"""Diagnostico de conexion: revisa la configuracion antes de correr el agente.

Contesta cuatro preguntas, en orden, y se detiene en la primera que falla:

  1. Se puede leer el .env y la clave tiene pinta de clave real?
  2. La base de recetas se lee bien?
  3. El modelo responde a una pregunta simple? (aca se ve si la clave sirve)
  4. El modelo entiende las herramientas y decide usarlas?

Uso:  python -m scripts.diagnostico
      python -m scripts.diagnostico --modelos   (solo lista modelos, no gasta cuota)
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402

from agente.config import Configuracion  # noqa: E402
from agente.llm import construir_llm, construir_llm_con_herramientas  # noqa: E402
from agente.repositorio import obtener_repositorio  # noqa: E402


def _enmascarar(clave: str) -> str:
    """Muestra solo el principio y el final de la clave: nunca la clave entera."""
    if len(clave) <= 14:
        return "***"
    return f"{clave[:8]}...{clave[-4:]} ({len(clave)} caracteres)"


async def _modelos_disponibles(config: Configuracion) -> list[str]:
    """Pregunta al proveedor que modelos puede usar esta clave.

    Sirve cuando el modelo configurado devuelve 404 o se agoto su cuota: en vez
    de adivinar nombres, se consulta la lista real. Listar no consume cuota de
    generacion. Si la consulta falla, devuelve una lista vacia.
    """
    try:
        if config.proveedor == "anthropic":
            import anthropic

            cliente = anthropic.Anthropic(api_key=config.api_key)
            modelos = await asyncio.to_thread(lambda: list(cliente.models.list()))
            return sorted(str(m.id) for m in modelos)

        from google import genai

        cliente_google = genai.Client(api_key=config.api_key)
        modelos = await asyncio.to_thread(lambda: list(cliente_google.models.list()))
    except Exception:
        return []

    nombres: list[str] = []
    for modelo in modelos:
        acciones = getattr(modelo, "supported_actions", None) or []
        if not acciones or "generateContent" in acciones:
            nombres.append(str(modelo.name).replace("models/", ""))
    return sorted(nombres)


async def main() -> int:
    solo_listar = "--modelos" in sys.argv

    print("\n1) Configuracion (.env)")
    try:
        config = Configuracion.desde_entorno()
    except RuntimeError as error:
        print(f"   FALLA: {error}")
        return 1
    print(f"   OK  proveedor: {config.proveedor}")
    print(f"   OK  clave: {_enmascarar(config.api_key)}")
    print(f"   OK  modelo: {config.modelo}")

    if solo_listar:
        print("\nModelos habilitados para tu clave:")
        for nombre in await _modelos_disponibles(config):
            print(f"  - {nombre}")
        print("\nElegi uno y ponelo en .env como MODELO_LLM=<nombre>\n")
        return 0

    print("\n2) Base de recetas")
    try:
        recetas = await obtener_repositorio(config.ruta_recetas).buscar_por_ingrediente(
            "pollo"
        )
        print(f"   OK  {len(recetas)} recetas con pollo (primera: {recetas[0]['nombre']})")
    except Exception:
        print("   FALLA al leer data/recetas.json:")
        traceback.print_exc()
        return 1

    print("\n3) Conexion con el modelo")
    try:
        respuesta = await construir_llm(config).ainvoke(
            [HumanMessage("Responde unicamente con la palabra: listo")]
        )
        print(f"   OK  el modelo respondio: {str(respuesta.content).strip()[:60]}")
    except Exception as error:
        print(f"   FALLA: {type(error).__name__}: {error}")
        print(
            "\n   Causas mas frecuentes:\n"
            "   - El nombre del modelo ya no existe o no esta habilitado para tu clave\n"
            "   - La clave no es valida, o la cuenta no tiene credito/cuota\n"
            "   - No hay conexion a internet o un proxy/firewall bloquea la salida"
        )
        modelos = await _modelos_disponibles(config)
        if modelos:
            print("\n   Modelos habilitados para tu clave (elegi uno para MODELO_LLM):")
            for nombre in modelos:
                print(f"     - {nombre}")
        return 1

    print("\n4) El modelo decide usar herramientas")
    try:
        con_tools = construir_llm_con_herramientas(config)
        salida = await con_tools.ainvoke(
            [HumanMessage("Que recetas tenes con pollo? Usa las herramientas.")]
        )
        llamadas = getattr(salida, "tool_calls", [])
        if llamadas:
            print(f"   OK  el modelo pidio: {[l['name'] for l in llamadas]}")
        else:
            print(
                "   ATENCION: el modelo respondio sin pedir herramientas.\n"
                "   El cableado esta bien, pero el modelo no se convencio de usarlas.\n"
                f"   Respondio: {str(salida.content)[:120]}"
            )
    except Exception as error:
        print(f"   FALLA: {type(error).__name__}: {error}")
        return 1

    print("\nTodo en orden. Ya podes correr: python -m scripts.run_demo\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
