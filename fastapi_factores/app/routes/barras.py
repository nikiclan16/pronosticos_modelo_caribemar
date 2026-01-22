from typing import List

from fastapi import APIRouter, Query

from app.schemas.factores import BarraIn
from app.services import factores_service as service

router = APIRouter(prefix="/factores/barras", tags=["barras"])


@router.get("")
def listar_barras():
    return {"ok": True, "data": service.consultar_barras_index()}


@router.get("/por-mc/{mc}")
def listar_barras_por_mc(mc: str):
    return {"ok": True, "data": service.consultar_barras_index_xmc(mc)}


@router.get("/{barra_id}")
def obtener_barra(barra_id: str):
    return {"ok": True, "data": service.consultar_barra(barra_id)}


@router.get("/mercado/{mc}")
def barras_mercado(mc: str):
    return {"ok": True, "data": service.consultar_barras_mercado(mc)}


@router.get("/por-codigo-rpm/{codigo_rpm}")
def barra_por_codigo_rpm(codigo_rpm: str):
    return {"ok": True, "data": service.consultar_barra_x_codigo_rpm(codigo_rpm)}


@router.get("/codigos-rpm")
def barra_codigos_rpm(barra: str):
    return {"ok": True, "data": service.consultar_barra_nombre(barra)}


@router.get("/flujo")
def barra_flujo(barra: str, tipo: str, codigo_rpm: List[str] = Query(...)):
    return {"ok": True, "data": service.consultar_barra_flujo_nombre(barra, tipo, codigo_rpm)}


@router.get("/factor")
def barra_factor(barra: str, tipo: str, codigo_rpm: List[str] = Query(...)):
    return {"ok": True, "data": service.consultar_barra_factor_nombre(barra, tipo, codigo_rpm)}


@router.get("/flujo-inicial")
def barra_flujo_inicial(barra: str, tipo: str):
    return {"ok": True, "data": service.consultar_barra_flujo_nombre_inicial(barra, tipo)}


@router.post("")
def crear_barra(payload: BarraIn):
    ok = service.guardar_barra(None, payload.barra, payload.descripcion, payload.nivel_tension, payload.observaciones, payload.habilitar, payload.mc)
    return {"ok": ok}


@router.put("/{barra_id}")
def actualizar_barra(barra_id: str, payload: BarraIn):
    ok = service.guardar_barra(barra_id, payload.barra, payload.descripcion, payload.nivel_tension, payload.observaciones, payload.habilitar, payload.mc)
    return {"ok": ok}


@router.delete("/{barra_id}")
def eliminar_barra(barra_id: str):
    ok = service.eliminar_barra(barra_id)
    return {"ok": ok}


@router.delete("/{barra_id}/con-agrupaciones")
def eliminar_barra_y_agrupaciones(barra_id: str):
    ok = service.eliminar_barra_agrupaciones(barra_id)
    return {"ok": ok}
