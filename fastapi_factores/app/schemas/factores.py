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


# --- Cálculos: curvas típicas, FDA, FDP ---


class CurvaTipicaRef(BaseModel):
    """Referencia a una curva (barra + fecha) para FDA/FDP."""
    barra: str
    fecha: str


class ClusteringRequest(BaseModel):
    """Request para clustering por barra (agregar factores y agrupar)."""
    fecha_inicial: str
    fecha_final: str
    mc: str
    barra: str
    flujo_tipo: str  # 'A' | 'R'
    tipo_dia: Optional[str] = None
    database_url: Optional[str] = Field(None, description="URL de conexión a BD alternativa (ej: postgresql://user:pass@host:5432/db)")


class CurvasTipicasRequest(BaseModel):
    """Request para obtener las N curvas más típicas del histórico (forma y nivel)."""
    fecha_inicial: str
    fecha_final: str
    mc: str
    tipo_dia: str  # ORDINARIO, SABADO, FESTIVO
    flujo_tipo: str  # 'A' | 'R'
    n_max: int = Field(8, ge=1, le=100, description="Máximo de curvas típicas a devolver")
    barra: Optional[str] = None  # si se da, solo curvas de esa barra; si no, todas las barras del MC
    database_url: Optional[str] = Field(None, description="URL de conexión a BD alternativa (ej: postgresql://user:pass@host:5432/db)")


class CalculoFDARequest(BaseModel):
    """Request para FDA: se calcula solo sobre las curvas típicas indicadas."""
    fecha_inicial: str
    fecha_final: str
    mc: str
    tipo_dia: str  # ORDINARIO, SABADO, FESTIVO
    curvas_tipicas: List[CurvaTipicaRef] = Field(
        ...,
        description="Curvas seleccionadas (salida de curvas-tipicas). FDA se calcula solo sobre estas."
    )
    database_url: Optional[str] = Field(None, description="URL de conexión a BD alternativa (ej: postgresql://user:pass@host:5432/db)")


class CalculoFDPRequest(BaseModel):
    """Request para FDP: se calcula solo sobre las curvas típicas indicadas."""
    fecha_inicial: str
    fecha_final: str
    mc: str
    tipo_dia: str  # ORDINARIO, SABADO, FESTIVO
    curvas_tipicas: List[CurvaTipicaRef] = Field(
        ...,
        description="Curvas seleccionadas (salida de curvas-tipicas). FDP se calcula solo sobre estas."
    )
    database_url: Optional[str] = Field(None, description="URL de conexión a BD alternativa (ej: postgresql://user:pass@host:5432/db)")
