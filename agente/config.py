"""Configuracion central del proyecto.

Un unico lugar donde se leen las variables de entorno y se resuelven las rutas.
Ningun otro modulo llama a os.getenv(): asi, si manana cambia el modelo, el
proveedor o la ubicacion de la base, se toca un solo archivo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Raiz del repositorio (config.py esta en <raiz>/agente/config.py)
RAIZ_PROYECTO: Path = Path(__file__).resolve().parent.parent

PROVEEDORES_VALIDOS: tuple[str, ...] = ("gemini", "anthropic")

# Que variable de entorno guarda la clave de cada proveedor.
VARIABLE_DE_CLAVE: dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Modelo por defecto de cada proveedor, con la version fijada a proposito: los
# alias tipo "latest" cambian solos y volverian irreproducible la entrega.
MODELO_POR_DEFECTO: dict[str, str] = {
    "gemini": "gemini-3.6-flash",
    "anthropic": "claude-haiku-4-5-20251001",
}

# Textos de ejemplo del .env: si siguen ahi, la clave no fue completada.
MARCADORES_DE_EJEMPLO: tuple[str, ...] = ("pega_aca_tu_clave", "tu_clave", "sk-ant-xxx")


@dataclass(frozen=True)
class Configuracion:
    """Parametros de ejecucion del agente.

    Es `frozen` (inmutable) a proposito: la configuracion se arma una vez al
    arrancar y despues nadie la modifica desde adentro del grafo.
    """

    proveedor: str
    api_key: str
    modelo: str
    temperatura: float
    limite_recursion: int
    max_mensajes_contexto: int
    timeout_modelo: float
    ruta_recetas: Path
    ruta_checkpoints: Path
    ruta_logs: Path

    @classmethod
    def desde_entorno(cls) -> "Configuracion":
        """Construye la configuracion leyendo el archivo .env / las variables de entorno.

        Lanza RuntimeError si el proveedor no es valido o si falta la clave, para
        fallar temprano y con un mensaje claro en vez de romper recien cuando el
        modelo intenta responder.
        """
        load_dotenv(RAIZ_PROYECTO / ".env")

        proveedor = os.getenv("PROVEEDOR", "gemini").strip().lower()
        if proveedor not in PROVEEDORES_VALIDOS:
            raise RuntimeError(
                f"PROVEEDOR='{proveedor}' no es valido. Opciones: "
                f"{', '.join(PROVEEDORES_VALIDOS)}."
            )

        variable = VARIABLE_DE_CLAVE[proveedor]
        api_key = os.getenv(variable, "").strip().strip("\"'")

        if not api_key:
            raise RuntimeError(
                f"Falta {variable} para el proveedor '{proveedor}'. Copia "
                ".env.example a .env y completa esa variable."
            )
        if api_key in MARCADORES_DE_EJEMPLO or len(api_key) < 20:
            raise RuntimeError(
                f"{variable} parece no ser una clave real (sigue el texto de ejemplo "
                "o es demasiado corta). Abri el archivo .env y reemplaza el valor por "
                "tu clave, sin comillas ni espacios."
            )
        if proveedor == "anthropic" and not api_key.startswith("sk-ant-"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY deberia empezar con 'sk-ant-'. Revisa que sea la "
                "clave de la consola de API (platform.claude.com) y no otra cosa."
            )

        return cls(
            proveedor=proveedor,
            api_key=api_key,
            modelo=os.getenv("MODELO_LLM", "").strip() or MODELO_POR_DEFECTO[proveedor],
            temperatura=float(os.getenv("TEMPERATURA", "0")),
            # Techo de pasos del grafo: evita bucles infinitos y costos inesperados.
            limite_recursion=int(os.getenv("LIMITE_RECURSION", "10")),
            # Cuantos mensajes del historial se le mandan al modelo (estado sucio).
            max_mensajes_contexto=int(os.getenv("MAX_MENSAJES_CONTEXTO", "20")),
            # Techo de espera por cada llamada al modelo. Sin esto, una conexion
            # colgada deja el proceso esperando para siempre, sin decir nada.
            timeout_modelo=float(os.getenv("TIMEOUT_MODELO", "90")),
            ruta_recetas=RAIZ_PROYECTO / "data" / "recetas.json",
            ruta_checkpoints=RAIZ_PROYECTO / "checkpoints.sqlite",
            ruta_logs=RAIZ_PROYECTO / "logs",
        )
