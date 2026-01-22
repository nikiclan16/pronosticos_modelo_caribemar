from fastapi import APIRouter

from app.schemas.factores import RangoFechasIn
from app.services import factores_service as service

router = APIRouter(prefix="/factores/rangos", tags=["rangos"])


@router.post("")
def crear_rango(payload: RangoFechasIn):
    ok = service.guardar_rango_fecha(payload.fechainicio, payload.fechafin, payload.ucp, payload.barra, payload.tipo_dia, payload.nro_dias)
    return {"ok": ok}


@router.get("")
def obtener_rango(ucp: str):
    return {"ok": True, "data": service.buscar_ucp_rango_fechas(ucp)}


@router.put("")
def actualizar_rango(payload: RangoFechasIn):
    ok = service.actualizar_rango_fechas(payload.fechainicio, payload.fechafin, payload.ucp, payload.barra, payload.tipo_dia, payload.nro_dias)
    return {"ok": ok}


@router.delete("")
def eliminar_rango(ucp: str, barra: str, tipo_dia: str, nro_dias: str):
    ok = service.eliminar_fechas_ingresadas(ucp, barra, tipo_dia, nro_dias)
    return {"ok": ok}


@router.delete("/ucp/{ucp}")
def eliminar_rangos_ucp(ucp: str):
    ok = service.eliminar_fechas_ingresadas_todos(ucp)
    return {"ok": ok}
