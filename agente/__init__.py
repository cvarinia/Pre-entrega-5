"""Agente de recetas con razonamiento ciclico (LangGraph) y memoria persistente.

Modulos:
    config       Configuracion central leida del .env
    repositorio  Capa de datos (base de recetas simulada)
    tools        Herramientas @tool que el modelo puede elegir
    state        Esquema del estado del grafo y sus reducers
    llm          Modelo de lenguaje + bind_tools + prompt de sistema
    checkpointer Persistencia del estado en SQLite
    graph        Definicion y compilacion del StateGraph
    sesion       Invocacion del grafo por thread_id
    tracing      Traza ReAct exportable a JSON
    cli          Interfaz de linea de comandos
"""

__version__ = "1.0.0"
