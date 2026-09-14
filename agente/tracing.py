"""Captura de la traza ReAct para entregarla como .json (requisito de la consigna).

El grafo ya guarda todo en el checkpointer, pero ese formato es interno. Este
modulo traduce la lista de mensajes a una traza legible: quien hablo, que
herramienta se decidio usar, con que argumentos, que devolvio y cual fue la
conclusion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from agente.mensajes import texto_de_mensaje


def _parsear(contenido: str) -> Any:
    """Convierte el contenido de un ToolMessage a objeto si es JSON valido."""
    try:
        return json.loads(contenido)
    except (json.JSONDecodeError, TypeError):
        return contenido


def mensajes_a_pasos(mensajes: list[AnyMessage]) -> list[dict[str, Any]]:
    """Traduce una lista de mensajes de LangChain a pasos numerados de la traza."""
    pasos: list[dict[str, Any]] = []

    for mensaje in mensajes:
        numero = len(pasos) + 1

        if isinstance(mensaje, HumanMessage):
            pasos.append(
                {
                    "paso": numero,
                    "tipo": "usuario",
                    "contenido": texto_de_mensaje(mensaje),
                }
            )

        elif isinstance(mensaje, AIMessage):
            llamadas = mensaje.tool_calls or []
            if llamadas:
                for llamada in llamadas:
                    pasos.append(
                        {
                            "paso": len(pasos) + 1,
                            "tipo": "decision_del_agente",
                            "accion": "llamar_herramienta",
                            "herramienta": llamada["name"],
                            "argumentos": llamada["args"],
                        }
                    )
            else:
                pasos.append(
                    {
                        "paso": numero,
                        "tipo": "respuesta_final",
                        "contenido": texto_de_mensaje(mensaje),
                    }
                )

        elif isinstance(mensaje, ToolMessage):
            pasos.append(
                {
                    "paso": numero,
                    "tipo": "resultado_herramienta",
                    "herramienta": mensaje.name,
                    "resultado": _parsear(mensaje.content),
                }
            )

    return pasos


@dataclass
class RegistroDeTraza:
    """Acumula los turnos de una sesion y los guarda como JSON."""

    thread_id: str
    modelo: str
    limite_recursion: int
    turnos: list[dict[str, Any]] = field(default_factory=list)

    def agregar_turno(
        self,
        titulo: str,
        pregunta: str,
        mensajes: list[AnyMessage],
        herramientas_invocadas: list[str],
        respuesta: str = "",
        error: str | None = None,
    ) -> dict[str, Any]:
        """Registra un turno completo (pregunta + ciclo de razonamiento + respuesta)."""
        turno: dict[str, Any] = {
            "turno": len(self.turnos) + 1,
            "titulo": titulo,
            "pregunta": pregunta,
            "herramientas_invocadas": herramientas_invocadas,
            "cantidad_de_llamadas_a_herramientas": len(herramientas_invocadas),
            "pasos": mensajes_a_pasos(mensajes),
            "respuesta": respuesta,
        }
        if error:
            # Si el ciclo fallo, el motivo queda en la traza: sin esto, un turno
            # fallido se veria identico a un turno vacio.
            turno["error"] = error
        self.turnos.append(turno)
        return turno

    def como_diccionario(self) -> dict[str, Any]:
        """Devuelve la traza completa lista para serializar."""
        return {
            "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "thread_id": self.thread_id,
            "modelo": self.modelo,
            "limite_recursion": self.limite_recursion,
            "total_de_turnos": len(self.turnos),
            "turnos": self.turnos,
        }

    def guardar(self, ruta: Path) -> Path:
        """Escribe la traza en disco como JSON con acentos legibles."""
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(
            json.dumps(self.como_diccionario(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return ruta


def imprimir_pasos(pasos: list[dict[str, Any]]) -> None:
    """Muestra la traza por consola mientras corre, para poder seguir el ciclo."""
    iconos = {
        "usuario": "[USUARIO]",
        "decision_del_agente": "[AGENTE ->]",
        "resultado_herramienta": "[<- TOOL ]",
        "respuesta_final": "[RESPUESTA]",
    }
    for paso in pasos:
        etiqueta = iconos.get(paso["tipo"], paso["tipo"])
        if paso["tipo"] == "decision_del_agente":
            print(f"  {etiqueta} {paso['herramienta']}({paso['argumentos']})")
        elif paso["tipo"] == "resultado_herramienta":
            resumen = json.dumps(paso["resultado"], ensure_ascii=False)
            if len(resumen) > 220:
                resumen = resumen[:220] + "..."
            print(f"  {etiqueta} {paso['herramienta']} -> {resumen}")
        else:
            print(f"  {etiqueta} {paso['contenido']}")
