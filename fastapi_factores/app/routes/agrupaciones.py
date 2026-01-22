from fastapi import APIRouter

from app.schemas.factores import AgrupacionIn
from app.services import factores_service as service

router = APIRouter(prefix="/factores/agrupaciones", tags=["agrupaciones"])


@router.get("")
def listar_agrupaciones():
    return {"ok": True, "data": service.consultar_agrupaciones_index()}


@router.get("/por-barra/{barra_id}")
def listar_agrupaciones_por_barra(barra_id: str):
    return {"ok": True, "data": service.consultar_agrupaciones_index_x_barra_id(barra_id)}


@router.get("/por-barra-rpm/{barra_id}")
def listar_agrupaciones_por_barra_rpm(barra_id: str):
    return {"ok": True, "data": service.consultar_agrupaciones_index_x_barra_id_order_rpm(barra_id)}


@router.get("/por-barra-nombre/{barra}")
def listar_agrupaciones_por_barra_nombre(barra: str):
    return {"ok": True, "data": service.consultar_agrupaciones_index_x_barra_order_rpm(barra)}


@router.get("/{agrupacion_id}")
def obtener_agrupacion(agrupacion_id: str):
    return {"ok": True, "data": service.consultar_agrupacion(agrupacion_id)}


@router.get("/factor")
def obtener_factor(codigo_rpm: str, flujo: str):
    return {"ok": True, "data": service.consultar_factor(codigo_rpm, flujo)}


@router.post("")
def crear_agrupacion(payload: AgrupacionIn):
    ok = service.guardar_agrupacion(None, payload.barra_id, payload.codigo_rpm, payload.flujo, payload.habilitar, payload.revision, payload.factor)
    return {"ok": ok}


@router.put("/{agrupacion_id}")
def actualizar_agrupacion(agrupacion_id: str, payload: AgrupacionIn):
    ok = service.guardar_agrupacion(agrupacion_id, payload.barra_id, payload.codigo_rpm, payload.flujo, payload.habilitar, payload.revision, payload.factor)
    return {"ok": ok}


@router.delete("/{agrupacion_id}")
def eliminar_agrupacion(agrupacion_id: str):
    ok = service.eliminar_agrupacion(agrupacion_id)
    return {"ok": ok}


@router.delete("/por-barra/{barra_id}")
def eliminar_agrupaciones_por_barra(barra_id: str):
    ok = service.eliminar_agrupaciones(barra_id)
    return {"ok": ok}
