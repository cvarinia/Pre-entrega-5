"""Prueba del grafo SIN consumir la API (modelo simulado).

Sirve para verificar que el cableado esta bien -- herramientas, arista
condicional, ciclo de vuelta, checkpointer y traza -- sin gastar una sola
llamada a Gemini. Si esta prueba pasa, cualquier problema que aparezca despues
esta en el prompt o en el modelo, no en la arquitectura.

Uso:  python -m tests.test_grafo_offline
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage  # noqa: E402

from agente.config import RAIZ_PROYECTO, Configuracion  # noqa: E402
from agente.sesion import abrir_sesion  # noqa: E402
from agente.tracing import mensajes_a_pasos  # noqa: E402


class ModeloSimulado:
    """Reemplaza al LLM devolviendo una secuencia fija de decisiones.

    Imita lo unico que el grafo le pide al modelo: un metodo `ainvoke` que
    recibe mensajes y devuelve un AIMessage (con o sin tool_calls).
    """

    def __init__(self, respuestas: list[AIMessage]) -> None:
        self._respuestas = respuestas
        self.llamadas: list[list[AnyMessage]] = []

    async def ainvoke(self, mensajes: list[AnyMessage], **_: object) -> AIMessage:
        self.llamadas.append(list(mensajes))
        if not self._respuestas:
            return AIMessage(content="Sin mas respuestas simuladas.")
        return self._respuestas.pop(0)


def _pedido(nombre: str, args: dict, identificador: str) -> AIMessage:
    """Arma un AIMessage que pide una herramienta, como haria el modelo real."""
    return AIMessage(
        content="",
        tool_calls=[{"name": nombre, "args": args, "id": identificador}],
    )


def _config_de_prueba(carpeta: Path) -> Configuracion:
    """Configuracion aislada: base de checkpoints temporal, sin API key real."""
    return Configuracion(
        proveedor="gemini",
        api_key="clave-de-prueba-suficientemente-larga",
        modelo="modelo-simulado",
        temperatura=0.0,
        limite_recursion=10,
        max_mensajes_contexto=20,
        timeout_modelo=90.0,
        ruta_recetas=RAIZ_PROYECTO / "data" / "recetas.json",
        ruta_checkpoints=carpeta / "checkpoints_test.sqlite",
        ruta_logs=carpeta / "logs",
    )


async def prueba_multipaso_y_memoria(config: Configuracion) -> None:
    """El agente encadena 3 herramientas y despues recuerda el hilo."""
    modelo = ModeloSimulado(
        [
            _pedido("buscar_recetas_por_ingrediente", {"ingrediente": "pollo"}, "a1"),
            _pedido("obtener_receta", {"receta_id": "r004"}, "a2"),
            _pedido(
                "calcular_nutricion_menu",
                {"receta_ids": ["r004"], "comensales": 4},
                "a3",
            ),
            AIMessage(content="Te recomiendo el pollo al limon: 1400 kcal en total."),
            AIMessage(content="Si, el pollo al limon ya es sin lacteos."),
        ]
    )

    async with abrir_sesion(config, "hilo-de-prueba", modelo) as sesion:
        turno1 = await sesion.preguntar("Cena con pollo para 4, cuantas calorias?")
        turno2 = await sesion.preguntar("Y si la quiero sin lacteos?")
        historial = await sesion.historial()

    assert turno1.cantidad_de_llamadas == 3, turno1.herramientas_invocadas
    assert turno1.herramientas_invocadas == [
        "buscar_recetas_por_ingrediente",
        "obtener_receta",
        "calcular_nutricion_menu",
    ]
    assert "1400" in turno1.respuesta

    # Resiliencia de estado: en la ultima llamada al modelo, el historial que
    # recibio incluye la conversacion anterior (no arranco de cero).
    ultimos_mensajes = modelo.llamadas[-1]
    assert any(
        "cuantas calorias" in str(m.content).lower() for m in ultimos_mensajes
    ), "El agente no recibio el historial previo: la persistencia no esta funcionando."
    assert len(historial) > len(turno2.mensajes_nuevos)

    # La traza tiene los 3 resultados de herramienta con datos reales.
    pasos = mensajes_a_pasos(turno1.mensajes_nuevos)
    resultados = [p for p in pasos if p["tipo"] == "resultado_herramienta"]
    assert len(resultados) == 3
    assert resultados[1]["resultado"]["nombre"].startswith("Pollo al limon")
    assert resultados[2]["resultado"]["calorias_totales_menu"] == 1400
    print("OK  multi-paso, memoria del thread_id y traza")


async def prueba_ciclo_de_error(config: Configuracion) -> None:
    """Ante un id inexistente la tool devuelve error y el agente reintenta."""
    modelo = ModeloSimulado(
        [
            _pedido("obtener_receta", {"receta_id": "r099"}, "b1"),
            _pedido("buscar_recetas_por_ingrediente", {"ingrediente": "pollo"}, "b2"),
            AIMessage(content="Ese id no existe; encontre estas recetas con pollo."),
        ]
    )

    async with abrir_sesion(config, "hilo-de-error", modelo) as sesion:
        turno = await sesion.preguntar("Ingredientes de la receta r099")

    mensajes_tool = [m for m in turno.mensajes_nuevos if isinstance(m, ToolMessage)]
    assert "error" in mensajes_tool[0].content
    assert "sugerencia" in mensajes_tool[0].content
    # Lo importante: despues del error el grafo volvio al nodo modelo y hubo un
    # segundo intento con otra herramienta.
    assert turno.cantidad_de_llamadas == 2
    print("OK  ciclo de retorno ante error de herramienta")


async def prueba_limite_de_recursion(config: Configuracion) -> None:
    """Un modelo que pide herramientas para siempre choca contra el techo."""
    config_corta = Configuracion(**{**config.__dict__, "limite_recursion": 4})
    modelo = ModeloSimulado(
        [_pedido("obtener_receta", {"receta_id": "r001"}, f"c{i}") for i in range(50)]
    )

    async with abrir_sesion(config_corta, "hilo-infinito", modelo) as sesion:
        turno = await sesion.preguntar("Entra en loop")

    assert "no pudo completar" in turno.respuesta.lower(), turno.respuesta
    print("OK  recursion_limit corta el bucle infinito")


async def prueba_reintento_ante_error_temporal() -> None:
    """Un 429 con demora corta se reintenta solo; un error definitivo no."""
    from agente.reintentos import con_reintentos

    fallas = {"n": 0}

    async def falla_una_vez() -> str:
        if fallas["n"] == 0:
            fallas["n"] += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED ... retryDelay': '0s'")
        return "listo"

    assert await con_reintentos(falla_una_vez) == "listo"
    assert fallas["n"] == 1

    async def clave_invalida() -> str:
        raise RuntimeError("401 API key not valid")

    try:
        await con_reintentos(clave_invalida)
    except RuntimeError as error:
        assert "401" in str(error)
    else:  # pragma: no cover
        raise AssertionError("un error definitivo no deberia reintentarse")

    print("OK  reintento ante errores temporales del proveedor")


async def main() -> None:
    with tempfile.TemporaryDirectory() as carpeta:
        config = _config_de_prueba(Path(carpeta))
        await prueba_multipaso_y_memoria(config)
        await prueba_ciclo_de_error(config)
        await prueba_limite_de_recursion(config)
        await prueba_reintento_ante_error_temporal()
    print("\nTodas las pruebas pasaron.")


if __name__ == "__main__":
    asyncio.run(main())
