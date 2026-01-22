from typing import List

from fastapi import APIRouter, Query

from app.schemas.factores import MedidaIn, MedidasBulkIn, ModalInicioIn, MarcarMedidaIn
from app.services import factores_service as service

router = APIRouter(prefix="/factores/medidas", tags=["medidas"])


@router.get("/index")
def listar_medidas_index():
    return {"ok": True, "data": service.consultar_medidas_index()}


@router.get("")
def consultar_medidas(
    fecha_inicial: str,
    fecha_final: str,
    barra: str,
    e_ar: str,
):
    data = service.consultar_medidas(fecha_inicial, fecha_final, barra, e_ar)
    return {"ok": True, "data": data}


@router.get("/calcular")
def consultar_medidas_calcular(
    fecha_inicial: str,
    fecha_final: str,
    e_ar: str,
    mc: str,
):
    data = service.consultar_medidas_calcular(fecha_inicial, fecha_final, e_ar, mc)
    return {"ok": True, "data": data}


@router.get("/e-ar")
def consultar_medidas_e_ar(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
):
    data = service.consultar_medidas_e_ar(fecha_inicial, fecha_final, mc)
    return {"ok": True, "data": data}


@router.get("/completo")
def consultar_medidas_completo(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    e_ar: str = "",
    tipo_dia: str = "",
):
    data = service.consultar_medidas_completo(fecha_inicial, fecha_final, mc, e_ar, tipo_dia)
    return {"ok": True, "data": data}


@router.get("/calcular-completo")
def consultar_medidas_calcular_completo(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    barra: str,
    flujo: List[str] = Query(...),
    tipo_dia: str = "",
    codigo_rpm: List[str] = Query(...),
    marcado: bool = False,
):
    data = service.consultar_medidas_calcular_completo(
        fecha_inicial,
        fecha_final,
        mc,
        flujo,
        tipo_dia,
        codigo_rpm,
        barra,
        marcado,
    )
    return {"ok": True, "data": data}


@router.get("/calcular-completo-sin-barra")
def consultar_medidas_calcular_completo_sin_barra(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    barra: str,
    flujo: List[str] = Query(...),
    tipo_dia: str = "",
    codigo_rpm: List[str] = Query(...),
    marcado: bool = False,
):
    data = service.consultar_medidas_calcular_completo_sin_barra(
        fecha_inicial,
        fecha_final,
        mc,
        flujo,
        tipo_dia,
        codigo_rpm,
        barra,
        marcado,
    )
    return {"ok": True, "data": data}


@router.get("/mc")
def consultar_medidas_mc(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
):
    data = service.consultar_medidas_mc(fecha_inicial, fecha_final, mc)
    return {"ok": True, "data": data}


@router.get("/{codigo}")
def obtener_medida(codigo: str):
    return {"ok": True, "data": service.consultar_medida(codigo)}


@router.get("/fecha-inicial")
def fecha_inicial_medidas(mc: str):
    return {"ok": True, "data": service.consultar_fecha_inicial_medidas(mc)}


@router.get("/fecha-final")
def fecha_final_medidas(mc: str):
    return {"ok": True, "data": service.consultar_fecha_final_medidas(mc)}


@router.get("/existe")
def medida_existe(flujo: str, fecha: str, codigo_rpm: str):
    return {"ok": True, "data": service.buscar_medida(flujo, fecha, codigo_rpm)}


@router.post("")
def crear_medida(payload: MedidaIn):
    ok = service.ingresar_medida(payload.flujo, payload.fecha, payload.codigo_rpm, payload.periodos)
    return {"ok": ok}


@router.put("")
def actualizar_medida(payload: MedidaIn):
    ok = service.actualizar_medida(payload.flujo, payload.fecha, payload.codigo_rpm, payload.periodos)
    return {"ok": ok}


@router.post("/bulk")
def bulk_medidas(payload: MedidasBulkIn):
    items = [item.dict() for item in payload.items]
    count = service.bulk_medidas(items, payload.mode)
    return {"ok": True, "count": count}


@router.post("/reiniciar")
def reiniciar_medidas():
    ok = service.reiniciar_medidas()
    return {"ok": ok}


@router.post("/marcar")
def marcar_medidas(payload: MarcarMedidaIn):
    codigos = service.consultar_barra_nombre(payload.barra)
    codigo_rpm = [row["codigo_rpm"] for row in codigos]
    ok = service.marcar_desmarcar_medida(payload.fecha, payload.marcado, codigo_rpm)
    return {"ok": ok}


@router.post("/faltantes")
def medidas_faltantes(payload: ModalInicioIn):
    data = service.modal_inicio(payload.fechainicio, payload.fechafin, payload.mc, payload.tipodia, payload.barra, payload.e_ar)
    return data
