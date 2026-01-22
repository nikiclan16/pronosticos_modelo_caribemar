from typing import List, Optional
from pydantic import BaseModel, Field


class BarraIn(BaseModel):
    barra: str
    descripcion: Optional[str] = None
    nivel_tension: Optional[str] = None
    observaciones: Optional[str] = None
    habilitar: Optional[str] = None
    mc: Optional[str] = None


class AgrupacionIn(BaseModel):
    barra_id: str
    codigo_rpm: str
    flujo: str
    habilitar: Optional[str] = None
    revision: Optional[str] = None
    factor: Optional[str] = None


class RangoFechasIn(BaseModel):
    fechainicio: str
    fechafin: str
    ucp: str
    barra: Optional[str] = None
    tipo_dia: Optional[str] = None
    nro_dias: Optional[str] = None


class MedidaIn(BaseModel):
    flujo: str
    fecha: str
    codigo_rpm: str
    periodos: List[float] = Field(min_items=24, max_items=24)
    marcado: Optional[int] = None


class MedidasBulkIn(BaseModel):
    items: List[MedidaIn]
    mode: str = Field("upsert", description="insert|update|upsert|delete")


class ModalInicioIn(BaseModel):
    fechainicio: str
    fechafin: str
    mc: str
    tipodia: str
    barra: str
    e_ar: str
    barra_id: Optional[str] = None


class MarcarMedidaIn(BaseModel):
    fecha: str
    marcado: bool
    mc: Optional[str] = None
    e_ar: Optional[str] = None
    barra: str
