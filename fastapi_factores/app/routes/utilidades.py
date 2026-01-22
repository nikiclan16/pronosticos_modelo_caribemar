from fastapi import APIRouter

from app.services import factores_service as service

router = APIRouter(prefix="/factores", tags=["utilidades"])


@router.get("/mac/{mac}")
def buscar_mac(mac: str):
    return {"ok": True, "data": service.buscar_mac(mac)}


@router.get("/tipo-dia/{nombre}")
def buscar_tipo_dia(nombre: str):
    return {"ok": True, "data": service.buscar_tipo_dia(nombre)}


@router.get("/batch/estado")
def batch_estado():
    return {"ok": True, "data": service.buscar_batch_corriendo()}


@router.post("/batch/iniciar")
def batch_iniciar():
    return {"ok": service.iniciar_proceso_batch()}


@router.get("/batch/ruta")
def batch_ruta():
    return {"ok": True, "data": service.buscar_ruta_batch()}


@router.get("/festivos")
def buscar_festivo(fecha: str, ucp: str):
    return {"ok": True, "data": service.buscar_festivo(fecha, ucp)}
