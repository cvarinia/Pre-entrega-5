# Pre-entrega 5 — Agente de razonamiento cíclico con memoria persistente

Agente conversacional construido con **LangGraph** que responde consultas sobre una base de recetas. Decide por su cuenta qué herramientas usar, encadena varias llamadas para una sola pregunta, se recupera solo cuando una herramienta devuelve un error, y recuerda la conversación entre invocaciones gracias a un checkpointer en SQLite.

Todo el proyecto es asincrónico (`asyncio`), está tipado (type hints) y modularizado: cada archivo tiene una única responsabilidad.

---

## 1. El grafo en una imagen

```
                        START
                          │
                          ▼
                   ┌─────────────┐
                   │   modelo    │  ← el LLM decide: ¿respondo o pido una tool?
                   └─────────────┘
                     │         ▲
   tools_condition   │         │  arista de vuelta: cierra el ciclo
   (arista condicional)        │
                     │         │
        ┌────────────┴───┐     │
        │                │     │
        ▼                ▼     │
      END        ┌──────────────┴─┐
                 │  herramientas  │  ← ToolNode ejecuta lo que el modelo pidió
                 └────────────────┘
```

El ciclo `modelo → herramientas → modelo` **es** el razonamiento ReAct. No hay ningún `if/else` que elija herramienta: la decisión la toma el modelo a partir de los docstrings, y `tools_condition` solo mira si el último mensaje trae `tool_calls`.

---

## 2. Estructura del proyecto

```
pre-entrega_5/
├── agente/
│   ├── config.py         Configuración central (.env, rutas, límites)
│   ├── repositorio.py    Capa de datos: la "base de datos" simulada
│   ├── tools.py          Las 3 herramientas @tool con sus docstrings
│   ├── state.py          EstadoAgente(MessagesState) + reducers
│   ├── llm.py            Modelo (Claude o Gemini) + bind_tools() + prompt
│   ├── checkpointer.py   Persistencia con AsyncSqliteSaver
│   ├── graph.py          StateGraph, nodos y arista condicional
│   ├── sesion.py         Invocación por thread_id (usada por CLI y demo)
│   ├── reintentos.py     Reintento ante errores temporales del proveedor
│   ├── mensajes.py       Extracción de texto legible de los mensajes
│   ├── tracing.py        Traducción de mensajes a traza .json
│   └── cli.py            Interfaz de línea de comandos
├── data/recetas.json     18 recetas simuladas
├── logs/                 Trazas de ejecución
├── scripts/run_demo.py   Prueba de los 3 escenarios
├── scripts/diagnostico.py  Chequeo de .env, datos y conexión con el modelo
├── tests/test_grafo_offline.py   Verificación sin gastar API
├── requirements.txt
└── .env.example
```

La separación clave: **`tools.py` no sabe de dónde salen los datos** (eso es `repositorio.py`) y **`graph.py` no sabe qué hacen las herramientas** (solo las cablea). Cambiar el JSON por una base de datos real es reescribir un solo archivo.

---

## 3. Cómo levantar el entorno

Requisito: **Python 3.12 o superior**. Verificá con `python --version`.

Desde PowerShell, parado en la carpeta del proyecto:

```powershell
# 1. Crear el entorno virtual
python -m venv .venv

# 2. Activarlo (si PowerShell lo bloquea, ver la nota de abajo)
.\.venv\Scripts\Activate.ps1

# 3. Instalar las dependencias
pip install -r requirements.txt

# 4. Configurar la clave
copy .env.example .env
#    …y abrí .env para elegir PROVEEDOR y pegar la clave correspondiente
```

> Si al activar el entorno PowerShell dice *"la ejecución de scripts está deshabilitada"*, corré una vez:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

El proyecto funciona con dos proveedores, elegidos con `PROVEEDOR` en el `.env`:

| `PROVEEDOR` | Clave | Dónde se saca | Modelo por defecto |
|---|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | [platform.claude.com/settings/keys](https://platform.claude.com/settings/keys) (requiere crédito) | `claude-haiku-4-5-20251001` |
| `gemini` | `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) (gratis, con cuota diaria) | `gemini-3.6-flash` |

Cambiar de uno a otro es cambiar esa línea del `.env`: no se toca una sola línea del grafo. **La clave nunca se sube al repo**: `.env` está en `.gitignore`.

---

## 4. Cómo ejecutarlo

### La prueba completa (genera la traza de la entrega)

```powershell
python -m scripts.run_demo
```

Corre tres escenarios seguidos sobre el mismo `thread_id`, imprime el ciclo paso a paso en consola y escribe **`logs/traza_ejemplo.json`**.

### Conversación interactiva

```powershell
python -m agente.cli --thread-id cocina-lunes
```

Si más tarde volvés a correrlo con el mismo `--thread-id`, el agente sigue la conversación donde había quedado, aunque hayas cerrado la terminal.

Otras opciones:

```powershell
python -m agente.cli --thread-id prueba --pregunta "que puedo hacer con papa?"
python -m agente.cli --thread-id prueba --recursion-limit 6 --guardar-traza mi_traza.json
```

### Diagnóstico (si algo no funciona)

```powershell
python -m scripts.diagnostico
```

Chequea en orden la clave del `.env`, la lectura de la base de recetas, la conexión con el modelo y si el modelo entiende las herramientas. Se detiene en el primer punto que falla y dice qué revisar.

```powershell
python -m scripts.diagnostico --modelos
```

Lista solo los modelos habilitados para tu clave, sin consumir cuota de generación.

> **Cuotas y límites:** con `PROVEEDOR=gemini`, el nivel gratuito limita las peticiones diarias por modelo (unas 20). Una corrida completa del demo consume alrededor de una docena. Ante un `429 RESOURCE_EXHAUSTED`, se puede esperar al día siguiente o cambiar `MODELO_LLM`: la cuota se cuenta por modelo. El agente reintenta solo ante límites por minuto y timeouts (`agente/reintentos.py`), pero la cuota diaria no se resuelve esperando. Con `PROVEEDOR=anthropic` no hay cuota diaria, se consume el crédito de la cuenta.

### Verificación sin consumir API

```powershell
python -m tests.test_grafo_offline
```

Corre el grafo con un modelo simulado y comprueba las tres cosas que pide la consigna: el multi-paso, la memoria del `thread_id` y el corte por `recursion_limit`. Útil para confirmar que la instalación está bien antes de gastar llamadas.

---

## 5. Las herramientas

| Herramienta | Qué hace | Por qué está |
|---|---|---|
| `buscar_recetas_por_ingrediente(ingrediente, max_resultados)` | Devuelve **solo** id, nombre, tiempo y porciones | La información incompleta *a propósito* obliga al agente a encadenar una segunda llamada |
| `obtener_receta(receta_id)` | Ficha completa: ingredientes, calorías, etiquetas | Es la única fuente confiable para saber si algo tiene lácteos o gluten |
| `calcular_nutricion_menu(receta_ids, comensales)` | Agrega calorías y escala porciones | Paso final del razonamiento: necesita los ids ya verificados |

Ninguna herramienta lanza excepciones hacia el grafo. Cuando algo falla devuelven `{"error": ..., "sugerencia": ...}`, y **ese error es lo que habilita el segundo intento**: el modelo lo lee como un resultado más y corrige el rumbo.

---

## 6. Persistencia y memoria

```python
config = {"configurable": {"thread_id": "cocina-lunes"}, "recursion_limit": 10}
```

- El **`thread_id`** identifica el hilo de conversación. Dos preguntas con el mismo `thread_id` comparten historial.
- El **checkpointer** (`AsyncSqliteSaver`) guarda una foto del estado después de cada paso en `checkpoints.sqlite`. Ese archivo está en `.gitignore`: es estado local, no código.
- El **`recursion_limit`** es el techo de pasos del grafo. Sin él, un ciclo mal cerrado gastaría llamadas a la API sin fin. Está fijado en 10 y es configurable desde `.env`.
- El **recorte de historial** (`recortar_historial` en `graph.py`) evita el "estado sucio": el contexto se acumula en cada vuelta del ciclo, así que solo se le mandan al modelo los últimos N mensajes, cortando siempre en un mensaje humano para no dejar respuestas de herramienta huérfanas.

---

## 7. Traza de ejecución

- **`logs/traza_ejemplo.json`** — la corrida real con Gemini. Se genera con `python -m scripts.run_demo`.
- **`logs/traza_offline_simulada.json`** — la misma estructura generada con el modelo simulado, sin llamadas a la API. Sirve para ver el formato.

Formato de cada paso:

```json
{ "paso": 2, "tipo": "decision_del_agente", "accion": "llamar_herramienta",
  "herramienta": "buscar_recetas_por_ingrediente", "argumentos": {"ingrediente": "pollo"} },
{ "paso": 3, "tipo": "resultado_herramienta",
  "herramienta": "buscar_recetas_por_ingrediente", "resultado": { "cantidad": 5, "recetas": [...] } }
```

Los tipos posibles son `usuario`, `decision_del_agente`, `resultado_herramienta` y `respuesta_final`.

---

## 8. Dónde está cada criterio de la consigna

| Criterio | Archivo | Qué mirar |
|---|---|---|
| Autonomía (sin `if/else`) | `graph.py`, `tools.py` | `bind_tools()` + `tools_condition`; la decisión vive en los docstrings |
| Ciclo de retorno | `tools.py`, `graph.py` | Errores estructurados + arista `herramientas → modelo` |
| Resiliencia de estado | `checkpointer.py`, `sesion.py` | `AsyncSqliteSaver` + `thread_id` |
| Código limpio | todo el proyecto | Python 3.12, type hints, `async`/`await` en todo el camino |
| `StateGraph` hereda de `MessagesState` | `state.py` | `EstadoAgente(MessagesState)` + reducer `operator.add` |
| Al menos 1 tool con docstring | `tools.py` | Son 3, con docstrings escritos para que los lea el modelo |
| `recursion_limit` | `sesion.py`, `.env` | Se pasa en cada invocación |
| Multi-paso (≥2 llamadas) | `scripts/run_demo.py` | Escenario 1: encadena las 3 herramientas |
| Traza en `.json` | `tracing.py`, `logs/` | Generada automáticamente al correr el demo |

---

## 9. Dos aclaraciones para la corrección

**Sobre el proveedor del LLM.** El agente corre con **Claude** (`claude-haiku-4-5`), como sugiere la consigna, y también con **Gemini**, que es lo que se usó durante buena parte del desarrollo por ser gratuito. La elección se hace con una variable de entorno, no tocando código: `agente/llm.py` es el único archivo que sabe qué proveedor hay detrás. Eso es posible porque `bind_tools()` pertenece a la interfaz común de LangChain, así que el grafo, el estado, las herramientas y el checkpointer son idénticos en los dos casos.

**Sobre el checkpointer.** La consigna nombra `SqliteSaver`; acá se usa **`AsyncSqliteSaver`**, la versión asincrónica del mismo paquete (`langgraph-checkpoint-sqlite`), porque la consigna también pide gestión asincrónica con `asyncio`. Es el mismo archivo `.sqlite` y el mismo modelo de datos; la versión sincrónica bloquearía el event loop en cada escritura.
