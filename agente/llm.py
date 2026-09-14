"""Construccion del modelo de lenguaje y su vinculacion con las herramientas.

Este es el UNICO archivo que sabe que proveedor de LLM se esta usando. El grafo,
el estado, las herramientas y el checkpointer son identicos con Gemini, Claude o
cualquier otro modelo de LangChain, porque todos exponen la misma interfaz
`bind_tools()` / `ainvoke()`.

El proveedor se elige con la variable PROVEEDOR del .env:

    PROVEEDOR=anthropic   -> Claude (lo que sugiere la consigna)
    PROVEEDOR=gemini      -> Gemini (API gratuita, usada durante el desarrollo)

Las importaciones de cada cliente son perezosas (adentro de la funcion) para que
el proyecto arranque aunque este instalado solo uno de los dos paquetes.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from agente.config import Configuracion
from agente.tools import HERRAMIENTAS

# Instruccion de sistema: define el rol y, sobre todo, la POLITICA de uso de
# herramientas. No enumera pasos fijos (eso seria volver a los if/else): describe
# criterios para que el modelo decida solo.
PROMPT_SISTEMA = """Sos un asistente de cocina que trabaja sobre una base de datos de recetas.

Como trabajas:
- Nunca inventes recetas, calorias ni ingredientes. Todo dato que afirmes tiene
  que venir de una herramienta. Si no lo consultaste, no lo sabes.
- Encadena las herramientas que hagan falta antes de responder. Es normal y
  esperable llamar a dos o tres herramientas seguidas para una sola pregunta.
- Si una herramienta devuelve un campo "error", no te rindas ni le traslades el
  error crudo a la persona: leelo, mira la "sugerencia" y reintenta de otra
  forma. Solo si el reintento tampoco funciona, pedile una aclaracion concreta.
- Para saber si una receta tiene lacteos o gluten, mira siempre las etiquetas
  que devuelve obtener_receta. No lo deduzcas del nombre.
- Cuando tengas todos los datos, responde en espanol rioplatense, en pocas
  lineas, con los numeros concretos que obtuviste.
"""

# Techo de tokens de la respuesta. Claude lo exige explicitamente; para este
# agente, que contesta en pocas lineas, sobra de lejos.
MAX_TOKENS_RESPUESTA = 2048


def construir_llm(config: Configuracion) -> BaseChatModel:
    """Instancia el modelo de chat del proveedor configurado, sin herramientas."""
    if config.proveedor == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.modelo,
            temperature=config.temperatura,
            api_key=config.api_key,
            timeout=config.timeout_modelo,
            max_tokens=MAX_TOKENS_RESPUESTA,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.modelo,
        temperature=config.temperatura,
        google_api_key=config.api_key,
    )


def construir_llm_con_herramientas(config: Configuracion) -> Runnable:
    """Instancia el modelo y le vincula las herramientas con `bind_tools()`.

    `bind_tools` es lo que convierte a las funciones de `tools.py` en opciones
    que el modelo puede elegir por si mismo: le manda al proveedor el nombre, la
    firma y el docstring de cada una. La decision de cual usar es del modelo,
    no del codigo. Es la misma llamada para cualquier proveedor.
    """
    return construir_llm(config).bind_tools(HERRAMIENTAS)
