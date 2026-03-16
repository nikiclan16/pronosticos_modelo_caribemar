"""
API Gateway para Sistema de Pronóstico Automatizado de Demanda Energética - EPM
=================================================================================

Endpoints:
    POST /predict - Genera predicción de 30 días con granularidad horaria
    GET /health - Estado del sistema
    GET /models - Información de modelos disponibles

Autor: Sistema EPM
Fecha: Noviembre 2024
Versión: 1.0.0
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import logging
import traceback
import pandas as pd
from fastapi.concurrency import run_in_threadpool
import os
from dotenv import load_dotenv
from openai import OpenAI
# Importar componentes del sistema
from src.pipeline.orchestrator import run_automated_pipeline
from src.models.trainer import ModelTrainer
from src.prediction.forecaster import ForecastPipeline
from src.prediction.hourly import HourlyDisaggregationEngine
from src.prediction.hourly.adjustment_validator import HourlyAdjustmentValidator
from src.pipeline.update_csv import full_update_csv
from fastapi.concurrency import run_in_threadpool

# Cargar variables de entorno
load_dotenv()
# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="EPM Energy Demand Forecasting API",
    description="API Gateway para pronóstico automatizado de demanda energética",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# SCHEMAS DE REQUEST/RESPONSE
# ============================================================================
class EventsRequest(BaseModel):
    """Schema para solicitud de eventos futuros"""
    ucp: str = Field(
        ...,
        description="Selección de UCP para búsqueda de eventos"
    )
    fecha_inicio: str = Field(
        ...,
        description="Fecha inicio del periodo a predecir (formato: YYYY-MM-DD)"
    )
    fecha_fin: str = Field(
        ...,
        description="Fecha fin del periodo a predecir (formato: YYYY-MM-DD)"
    )

class EventsResponse(BaseModel):
    """Schema para respuesta de eventos futuros"""

    events: Dict[str, str] = Field(..., description="Eventos futuros que podrían afectar la demanda energética (formato: {'YYYY-MM-DD': 'Nombre del evento'})")


class DeviationItem(BaseModel):
    """Schema para un desvío individual"""
    fecha: str = Field(..., description="Fecha del desvío (formato: YYYY-MM-DD)")
    mape: float = Field(..., description="MAPE del desvío (positivo o negativo, en porcentaje)")

    @field_validator('fecha')
    @classmethod
    def validate_fecha_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Formato de fecha inválido. Usar YYYY-MM-DD')
        return v


class AnalyzeDeviationRequest(BaseModel):
    """Schema para solicitud de análisis de desvíos"""
    ucp: str = Field(
        ...,
        description="UCP a analizar (ej: 'Atlantico', 'Oriente', 'Antioquia')"
    )
    desvios: List[DeviationItem] = Field(
        ...,
        description="Lista de desvíos a analizar (fecha y MAPE)",
        min_length=1
    )


class DeviationAnalysisItem(BaseModel):
    """Schema para el análisis de un desvío individual"""
    fecha: str = Field(..., description="Fecha del desvío")
    mape: float = Field(..., description="MAPE reportado")
    analisis: str = Field(..., description="Análisis de posibles causas del desvío")


class AnalyzeDeviationResponse(BaseModel):
    """Schema para respuesta de análisis de desvíos"""
    ucp: str = Field(..., description="UCP analizado")
    total_desvios: int = Field(..., description="Número total de desvíos analizados")
    resultados: List[DeviationAnalysisItem] = Field(..., description="Análisis detallado por cada desvío")
    resumen_general: str = Field(..., description="Resumen general de los desvíos y patrones identificados")


class BaseCurveRequest(BaseModel):
    """Schema para solicitud de curva base"""
    ucp: str = Field(
        ...,
        description="UCP para filtrar datos históricos (ej: 'Atlantico', 'Oriente')"
    )
    fecha_inicio: str = Field(
        ...,
        description="Fecha inicio del periodo (formato: YYYY-MM-DD)"
    )
    fecha_fin: str = Field(
        ...,
        description="Fecha fin del periodo (formato: YYYY-MM-DD)"
    )

    @field_validator('fecha_inicio', 'fecha_fin')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Valida formato de fechas"""
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Formato de fecha inválido. Usar YYYY-MM-DD')
        return v

class BaseCurveResponse(BaseModel):
    """Schema para respuesta de curva base"""
    curves: Dict[str, List[float]] = Field(
        ...,
        description="Curvas base por fecha (formato: {'YYYY-MM-DD': [P1, P2, ..., P24]})"
    )


class ValidateAdjustmentsRequest(BaseModel):
    """Schema para solicitud de validación de ajustes horarios"""
    fecha: str = Field(..., description="Fecha de predicción (YYYY-MM-DD)")
    tipo_dia: str = Field(..., description="Tipo de día: 'laboral', 'festivo', 'weekend'")
    predicciones_actuales: List[float] = Field(
        ...,
        description="24 valores actuales de predicción (P1-P24) en MWh",
        min_length=24,
        max_length=24
    )
    ajustes_solicitados: List[float] = Field(
        ...,
        description="24 ajustes solicitados (positivo=aumentar, negativo=disminuir, 0=sin cambio)",
        min_length=24,
        max_length=24
    )
    ucp: str = Field(
        default="Antioquia",
        description="UCP para cargar modelos (default: Antioquia)"
    )

    @field_validator('fecha')
    @classmethod
    def validate_fecha_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Formato de fecha inválido. Usar YYYY-MM-DD')
        return v

    @field_validator('tipo_dia')
    @classmethod
    def validate_tipo_dia(cls, v: str) -> str:
        valid_types = ['laboral', 'festivo', 'weekend']
        if v.lower() not in valid_types:
            raise ValueError(f'tipo_dia debe ser uno de: {valid_types}')
        return v.lower()

    @field_validator('predicciones_actuales', 'ajustes_solicitados')
    @classmethod
    def validate_list_length(cls, v: List[float]) -> List[float]:
        if len(v) != 24:
            raise ValueError('Debe proporcionar exactamente 24 valores (uno por hora)')
        return v

    class Config:
        schema_extra = {
            "example": {
                "fecha": "2024-12-15",
                "tipo_dia": "laboral",
                "predicciones_actuales": [1197.0, 1134.0, 1071.0, 1008.0, 945.0, 882.0,
                                          819.0, 1260.0, 1386.0, 1449.0, 1512.0, 1575.0,
                                          1638.0, 1701.0, 1764.0, 1827.0, 1890.0, 1953.0,
                                          2016.0, 2079.0, 2142.0, 2205.0, 2268.0, 2331.0],
                "ajustes_solicitados": [0, 0, 0, 200.0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "ucp": "Antioquia"
            }
        }


class PeriodComparison(BaseModel):
    """Schema para comparación de un periodo individual"""
    periodo: str = Field(..., description="Período (P1-P24)")
    valor_original: float = Field(..., description="Valor original de predicción (MWh)")
    ajuste_solicitado: float = Field(..., description="Ajuste solicitado por usuario")
    ajuste_aplicado: float = Field(..., description="Ajuste realmente aplicado")
    valor_final: float = Field(..., description="Valor final después de ajuste")
    razon_modificacion: Optional[str] = Field(None, description="Razón si el ajuste fue modificado")

    class Config:
        schema_extra = {
            "example": {
                "periodo": "P4",
                "valor_original": 1008.0,
                "ajuste_solicitado": 200.0,
                "ajuste_aplicado": 100.0,
                "valor_final": 1108.0,
                "razon_modificacion": "Ajuste reducido de +200.0 MWh a +100.0 MWh para preservar valle nocturno (desviación 18.5% > 15%)"
            }
        }


class ValidateAdjustmentsResponse(BaseModel):
    """Schema para respuesta de validación de ajustes horarios"""
    valores_ajustados: List[float] = Field(
        ...,
        description="24 valores ajustados optimizados (P1-P24) en MWh"
    )
    comparacion: List[PeriodComparison] = Field(
        ...,
        description="Comparación período por período"
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description="Metadatos del proceso de validación"
    )

    class Config:
        schema_extra = {
            "example": {
                "valores_ajustados": [1197.5, 1134.2, 1071.0, 1108.0, 945.0, 882.0,
                                      819.0, 1260.0, 1386.0, 1449.0, 1512.0, 1575.0,
                                      1638.0, 1701.0, 1764.0, 1827.0, 1890.0, 1953.0,
                                      2016.0, 2079.0, 2142.0, 2205.0, 2268.0, 2331.0],
                "comparacion": [
                    {
                        "periodo": "P1",
                        "valor_original": 1197.0,
                        "ajuste_solicitado": 0.0,
                        "ajuste_aplicado": 0.5,
                        "valor_final": 1197.5,
                        "razon_modificacion": None
                    }
                ],
                "metadata": {
                    "cluster_id": 12,
                    "metodo": "normal",
                    "total_original": 29800.5,
                    "total_ajustado": 30100.2,
                    "cambio_total": 299.7,
                    "desviacion_promedio_forma": 8.5,
                    "iteraciones_convergencia": 2,
                    "periodos_modificados": 3,
                    "ajustes_extremos_detectados": 1,
                    "tolerancia_aplicada": 15.0
                }
            }
        }


class ForecastTypeItem(BaseModel):
    """Entrada para mapear el tipo de pronóstico por día"""
    codigo: Optional[str] = Field(None, description="Código de referencia")
    ucp: Optional[str] = Field(None, description="UCP asociada")
    fecha: str = Field(..., description="Fecha del día (YYYY-MM-DD)")
    tipopronostico: str = Field(..., description="Tipo de pronóstico a devolver para ese día")

    @field_validator('fecha')
    @classmethod
    def validate_item_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Formato de fecha inválido en data[].fecha. Usar YYYY-MM-DD')
        return v


class PredictRequest(BaseModel):
    """Schema para solicitud de predicción"""
    # power_data_path: str = Field(
    #     ...,
    #     description="Ruta al archivo CSV con datos históricos de demanda hasta el día anterior"
    # )
    # weather_data_path: Optional[str] = Field(
    #     'data/raw/clima_new.csv',
    #     description="Ruta al archivo CSV con datos meteorológicos API EPM (se usa por defecto data/raw/clima_new.csv si no se especifica)"
    # )
    # start_date: Optional[str] = Field(
    #     None,
    #     description="Fecha inicial para filtrar datos históricos (formato: YYYY-MM-DD)"
    # )
    ucp: str = Field(
        None,
        description="Selección de UCP para calculos"
    )
    end_date: Optional[str] = Field(
        None,
        description="Fecha final de datos históricos (formato: YYYY-MM-DD)"
    )
    n_days: int = Field(
        30,
        description="Número de días a predecir",
        ge=1,
        le=90
    )
    force_retrain: bool = Field(
        False,
        description="Forzar reentrenamiento del modelo aunque exista uno. Si es True, entrena los 3 modelos y selecciona automáticamente el mejor basado en rMAPE"
    )
    offset_scalar: Optional[float] = Field(
        None,
        description="Escalar opcional para ajustar todas las predicciones (ej: 1.2 para aumentar 20%, 0.8 para disminuir 20%). Si no se proporciona, no se aplica ningún ajuste.",
        gt=0.0
    )

    @field_validator( 'end_date')
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        """Valida formato de fechas"""
        if v is not None:
            try:
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError('Formato de fecha inválido. Usar YYYY-MM-DD')
        return v

    class Config:
        schema_extra = {
            "example": {
                "power_data_path": "data/raw/demanda_historica.csv",
                "weather_data_path": "data/raw/clima_historico.csv",
                "start_date": "2023-01-01",
                "end_date": "2024-11-27",
                "n_days": 30,
                "force_retrain": False,
                "offset_scalar": 1.2
            }
        }


class HourlyPrediction(BaseModel):
    """Schema para predicción horaria de un día"""
    fecha: str = Field(..., description="Fecha de la predicción (YYYY-MM-DD)")
    dia_semana: str = Field(..., description="Día de la semana")
    demanda_total: float = Field(..., description="Demanda total del día (MWh)")
    is_festivo: bool = Field(..., description="Si es día festivo")
    is_weekend: bool = Field(..., description="Si es fin de semana")
    metodo_desagregacion: str = Field(..., description="Método usado (normal/special)")
    cluster_id: Optional[int] = Field(None, description="ID del cluster usado para desagregación")
    P1: float = Field(..., description="Período 1 (00:00-01:00) en MWh")
    P2: float = Field(..., description="Período 2 (01:00-02:00) en MWh")
    P3: float = Field(..., description="Período 3 (02:00-03:00) en MWh")
    P4: float = Field(..., description="Período 4 (03:00-04:00) en MWh")
    P5: float = Field(..., description="Período 5 (04:00-05:00) en MWh")
    P6: float = Field(..., description="Período 6 (05:00-06:00) en MWh")
    P7: float = Field(..., description="Período 7 (06:00-07:00) en MWh")
    P8: float = Field(..., description="Período 8 (07:00-08:00) en MWh")
    P9: float = Field(..., description="Período 9 (08:00-09:00) en MWh")
    P10: float = Field(..., description="Período 10 (09:00-10:00) en MWh")
    P11: float = Field(..., description="Período 11 (10:00-11:00) en MWh")
    P12: float = Field(..., description="Período 12 (11:00-12:00) en MWh")
    P13: float = Field(..., description="Período 13 (12:00-13:00) en MWh")
    P14: float = Field(..., description="Período 14 (13:00-14:00) en MWh")
    P15: float = Field(..., description="Período 15 (14:00-15:00) en MWh")
    P16: float = Field(..., description="Período 16 (15:00-16:00) en MWh")
    P17: float = Field(..., description="Período 17 (16:00-17:00) en MWh")
    P18: float = Field(..., description="Período 18 (17:00-18:00) en MWh")
    P19: float = Field(..., description="Período 19 (18:00-19:00) en MWh")
    P20: float = Field(..., description="Período 20 (19:00-20:00) en MWh")
    P21: float = Field(..., description="Período 21 (20:00-21:00) en MWh")
    P22: float = Field(..., description="Período 22 (21:00-22:00) en MWh")
    P23: float = Field(..., description="Período 23 (22:00-23:00) en MWh")
    P24: float = Field(..., description="Período 24 (23:00-00:00) en MWh")

    class Config:
        schema_extra = {
            "example": {
                "fecha": "2024-12-01",
                "dia_semana": "Domingo",
                "demanda_total": 31500.0,
                "is_festivo": False,
                "is_weekend": True,
                "metodo_desagregacion": "normal",
                "P1": 1197.0, "P2": 1134.0, "P3": 1071.0,
                # ... (resto de períodos)
            }
        }


class PredictResponse(BaseModel):
    """Schema para respuesta de predicción"""
    status: str = Field(..., description="Estado de la operación")
    message: str = Field(..., description="Mensaje descriptivo")
    metadata: Dict[str, Any] = Field(..., description="Metadata de la predicción")
    predictions: List[HourlyPrediction] = Field(..., description="Array de predicciones diarias con desagregación horaria")
    should_retrain: bool = Field(..., description="Indica si se recomienda reentrenar el modelo (true/false)")
    reason: str = Field(..., description="Razón por la cual se recomienda o no reentrenar")
    #events: Dict[str, str] = Field(..., description="Eventos futuros que podrían afectar la demanda energética (formato: {'YYYY-MM-DD': 'Nombre del evento'})")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Predicción generada exitosamente para 30 días",
                "metadata": {
                    "fecha_generacion": "2024-11-28T10:30:00",
                    "modelo_usado": "xgboost_20241120_161937",
                    "dias_predichos": 30,
                    "fecha_inicio": "2024-11-28",
                    "fecha_fin": "2024-12-27",
                    "demanda_promedio": 31500.0,
                    "demanda_min": 28000.0,
                    "demanda_max": 35000.0,
                    "modelo_entrenado": False,
                    "metricas_modelo": {
                        "mape": 0.45,
                        "rmape": 3.2,
                        "r2": 0.946
                    }
                },
                "predictions": [],  # Array de HourlyPrediction
                "should_retrain": False,
                "reason": "Error dentro de límites aceptables (MAPE: 2.35%)",
                "events": {
                    "2024-12-25": "Navidad",
                    "2024-12-31": "Fin de Año",
                    "2025-01-01": "Año Nuevo"
                }
            }
        }


class PredictWithBaseCurveRequest(PredictRequest):
    """Schema para solicitud combinada de predicción + curva base"""
    fecha_inicio: Optional[str] = Field(
        None,
        description="Fecha inicio del rango para curva base (YYYY-MM-DD). Si no se envía, se usa fecha_inicio de la predicción."
    )
    fecha_fin: Optional[str] = Field(
        None,
        description="Fecha fin del rango para curva base (YYYY-MM-DD). Si no se envía, se usa fecha_fin de la predicción."
    )

    @field_validator('fecha_inicio', 'fecha_fin')
    @classmethod
    def validate_optional_date_format(cls, v: Optional[str]) -> Optional[str]:
        """Valida formato de fechas cuando se envían opcionalmente."""
        if v is None:
            return v
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Formato de fecha inválido. Usar YYYY-MM-DD')
        return v
    
    data: Optional[List[ForecastTypeItem]] = Field(
        None,
        description="Lista opcional de fechas y tipopronostico a devolver junto con la respuesta."
    )


class PredictWithBaseCurveResponse(BaseModel):
    """Schema para respuesta combinada de predicción + curva base"""
    # prediction: PredictResponse = Field(..., description="Resultado completo de /predict")
    # base_curve: BaseCurveResponse = Field(..., description="Resultado de /base-curve para el mismo rango")
    resultado: Dict[str, Dict[str, float]]
    # forecast_types: List[ForecastTypeItem] = Field(
    #     default_factory=list,
    #     description="Listado de tipopronostico por fecha (filtrado a las fechas calculadas)"
    # )


class HistoricalDay(BaseModel):
    """Perfil histórico de un día (P1-P24 y opcional TOTAL)"""
    fecha: str = Field(..., description="Fecha del día histórico")
    valores: Dict[str, float] = Field(..., description="Valores por período (P1..P24) y TOTAL si está disponible")


class HourlyDiff(BaseModel):
    """Comparación horaria contra un histórico"""
    delta: float = Field(..., description="Predicción - histórico")
    delta_pct: Optional[float] = Field(None, description="(delta / histórico) * 100. None si histórico es 0.")
    relacion: str = Field(..., description="superior | inferior | igual")


class SingleDayComparisons(BaseModel):
    """Comparaciones por hora contra el día de referencia solicitado"""
    referencia: Optional[Dict[str, HourlyDiff]] = None


class AdjustedComparison(BaseModel):
    """Perfil ajustado y diffs resultantes (manteniendo el total original)"""
    profile: Dict[str, float] = Field(..., description="Perfil ajustado P1..P24")
    adjusted_deltas: Dict[str, HourlyDiff] = Field(..., description="Deltas vs histórico después del ajuste")
    scale_factor: float = Field(..., description="Factor de re-escalado para conservar el total original")


class SingleDayAdjustments(BaseModel):
    """Ajustes aplicados contra el día de referencia"""
    referencia: Optional[AdjustedComparison] = None


class ScaledHistoricalProfile(BaseModel):
    """Perfil histórico escalado para igualar el total de la predicción"""
    profile: Dict[str, float] = Field(..., description="Perfil escalado P1..P24")
    scale_factor: float = Field(..., description="Factor aplicado al histórico")
    total_pred: float = Field(..., description="Total de la predicción")
    total_hist: float = Field(..., description="Total original del histórico")


class SingleDayScaledResponse(BaseModel):
    """Respuesta para perfil histórico escalado al total de la predicción"""
    prediction: PredictResponse = Field(..., description="Resultado de la predicción para el día solicitado")
    history: Dict[str, HistoricalDay] = Field(
        default_factory=dict,
        description="Historial: día de referencia solicitado si existe"
    )
    scaled: Optional[ScaledHistoricalProfile] = Field(
        None,
        description="Perfil histórico escalado al total de la predicción"
    )


class SingleDayPredictResponse(BaseModel):
    """Respuesta para predicción de un solo día con historial relacionado"""
    prediction: PredictResponse = Field(..., description="Resultado de la predicción para el día solicitado")
    history: Dict[str, HistoricalDay] = Field(
        default_factory=dict,
        description="Historial: día de referencia solicitado si existe"
    )
    comparisons: SingleDayComparisons = Field(
        default_factory=SingleDayComparisons,
        description="Comparaciones horarias contra el día de referencia"
    )
    adjustments: SingleDayAdjustments = Field(
        default_factory=SingleDayAdjustments,
        description="Ajustes aplicados cuando |delta_pct| > 5% manteniendo el total de la predicción"
    )


class SingleDayPredictRequest(BaseModel):
    """Schema para predicción de un solo día específico"""
    ucp: str = Field(..., description="UCP para cálculos")
    fecha: str = Field(..., description="Fecha objetivo de la predicción (YYYY-MM-DD)")
    fecha_referencia: str = Field(..., description="Fecha histórica con la que se comparará (YYYY-MM-DD)")
    force_retrain: bool = Field(False, description="Forzar reentrenamiento del modelo")
    offset_scalar: Optional[float] = Field(
        None,
        description="Escalar opcional para ajustar la predicción (ej: 1.2 aumenta 20%)",
        gt=0.0
    )

    @field_validator('fecha')
    @classmethod
    def validate_fecha(cls, v: str) -> str:
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Formato de fecha inválido. Usar YYYY-MM-DD')
        return v

    @field_validator('fecha_referencia')
    @classmethod
    def validate_fecha_referencia(cls, v: str) -> str:
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Formato de fecha_referencia inválido. Usar YYYY-MM-DD')
        return v


class ReasonResponse(BaseModel):
    """Schema para respuesta de predicción"""
    reason: str = Field(..., description="Razón por la cual se recomienda o no reentrenar")
    #events: Dict[str, str] = Field(..., description="Eventos futuros que podrían afectar la demanda energética (formato: {'YYYY-MM-DD': 'Nombre del evento'})")

    class Config:
        schema_extra = {
            
                "reason": "Error dentro de límites aceptables (MAPE: 2.35%)",
             
        }
class ReasonRequest(BaseModel):
    """Schema para solicitud de predicción"""
    # power_data_path: str = Field(
    #     ...,
    #     description="Ruta al archivo CSV con datos históricos de demanda hasta el día anterior"
    # )
    # weather_data_path: Optional[str] = Field(
    #     'data/raw/clima_new.csv',
    #     description="Ruta al archivo CSV con datos meteorológicos API EPM (se usa por defecto data/raw/clima_new.csv si no se especifica)"
    # )
    # start_date: Optional[str] = Field(
    #     None,
    #     description="Fecha inicial para filtrar datos históricos (formato: YYYY-MM-DD)"
    # )
    ucp: str = Field(
        None,
        description="Selección de UCP para calculos"
    )
    end_date: Optional[str] = Field(
        None,
        description="Fecha final de datos históricos (formato: YYYY-MM-DD)"
    )
    force_retrain: bool = Field(
        False,
        description="Forzar reentrenamiento del modelo aunque exista uno. Si es True, entrena los 3 modelos y selecciona automáticamente el mejor basado en rMAPE"
    )


class HealthResponse(BaseModel):
    """Schema para respuesta de health check"""
    status: str
    timestamp: str
    version: str
    components: Dict[str, Dict[str, Any]]




async def get_future_events_from_openai(
    ucp: str,
    fecha_inicio: str,
    fecha_fin: str
) -> Dict[str, str]:
    """
    Obtiene eventos futuros que podrían afectar la demanda energética usando OpenAI

    Args:
        ucp: Nombre del UCP (ej: 'Atlantico', 'Oriente')
        fecha_inicio: Fecha inicio del periodo a predecir
        fecha_fin: Fecha fin del periodo a predecir

    Returns:
        Dict[str, str]: Diccionario con formato {"YYYY-MM-DD": "Nombre del evento"}
    """
    try:
        # Obtener API key desde variables de entorno
        api_key = os.getenv('API_KEY')
        if not api_key:
            logger.warning("⚠ API_KEY no encontrada en .env, saltando búsqueda de eventos")
            return {}

        # Inicializar cliente de OpenAI
        client = OpenAI(api_key=api_key)

        prompt = f"""Busca en internet todos los eventos que podrían afectar la demanda de energía eléctrica en {ucp}, Colombia entre {fecha_inicio} y {fecha_fin}.

Incluye:
- Festivos nacionales y locales
- Eventos masivos (conciertos, festivales, ferias)
- Partidos de fútbol importantes
- Eventos climáticos previstos
- Paros o manifestaciones programadas
- Elecciones o eventos políticos
- Cualquier otro evento relevante

IMPORTANTE: Responde ÚNICAMENTE en formato JSON así:
{{
  "YYYY-MM-DD": "Nombre breve del evento",
  "YYYY-MM-DD": "Otro evento"
}}

REGLAS:
- NO incluyas links, URLs ni referencias a fuentes
- Solo nombres cortos y concisos (máximo 50 caracteres)
- NO agregues explicaciones adicionales
- Solo el JSON puro"""

        logger.info(f"🔍 Buscando eventos futuros para {ucp} ({fecha_inicio} a {fecha_fin})...")

        response = await run_in_threadpool(
            lambda: client.responses.create(
                model="gpt-5-mini",
                input=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                tools=[
                    {
                        "type": "web_search"
                    }
                ]
            )
        )

        events_json = response.output_text.strip()

        # Limpiar respuesta (quitar markdown si existe)
        if events_json.startswith("```json"):
            events_json = events_json.replace("```json", "").replace("```", "").strip()
        elif events_json.startswith("```"):
            events_json = events_json.replace("```", "").strip()

        # Parsear JSON
        import json
        events = json.loads(events_json)

        logger.info(f"✓ Eventos encontrados: {len(events)}")
        return events

    except Exception as e:
        logger.error(f"Error obteniendo eventos futuros: {e}")
        logger.error(traceback.format_exc())
        return {}


async def analyze_error_with_openai(
    ucp: str,
    error_type: str,
    mape_total: float,
    fecha_inicio: str,
    fecha_fin: str,
    dias_consecutivos: Optional[List[str]] = None
) -> str:
    """
    Analiza las posibles causas del error de predicción usando OpenAI con búsqueda en internet

    Args:
        ucp: Nombre del UCP (ej: 'Atlantico', 'Oriente')
        error_type: Tipo de error ('mensual', 'consecutivo', 'ambos')
        mape_total: MAPE mensual calculado
        fecha_inicio: Fecha inicio del periodo analizado
        fecha_fin: Fecha fin del periodo analizado
        dias_consecutivos: Lista de fechas con errores consecutivos > 5%

    Returns:
        str: Análisis detallado de OpenAI sobre posibles causas
    """
    try:
        # Obtener API key desde variables de entorno
        api_key = os.getenv('API_KEY')
        if not api_key:
            logger.warning("⚠ API_KEY no encontrada en .env, saltando análisis de OpenAI")
            return "Análisis no disponible (API_KEY no configurada)"

        # Inicializar cliente de OpenAI
        client = OpenAI(api_key=api_key)

        # Construir prompt según tipo de error
        if error_type == 'consecutivo':
            dias_str = ', '.join(dias_consecutivos) if dias_consecutivos else 'últimos 2 días'
            prompt = f"""Eres un analista energético experto. Necesito que investigues en internet las posibles causas de una anomalía en la demanda energética.

**Contexto:**
- UCP: {ucp}, Colombia
- Fechas afectadas: {dias_str}
- Tipo de anomalía: Dos días consecutivos con error de predicción superior al 5%
- Periodo analizado: {fecha_inicio} a {fecha_fin}

**Tarea:**
Busca en internet eventos, acontecimientos o situaciones que pudieron haber ocurrido en {ucp}, Colombia en las fechas {dias_str} que pudieron causar variaciones significativas en la demanda de energía eléctrica.

Considera:
- Eventos climáticos extremos (tormentas, olas de calor/frío)
- Eventos públicos masivos (conciertos, partidos, festivales)
- Días festivos locales o nacionales
- Apagones o fallas en el suministro
- Eventos políticos o sociales
- Paros o manifestaciones
- cualquier otro acontecimiento relevante

Proporciona un análisis conciso (máximo 2-3 oraciones) con las causas más probables encontradas."""

        elif error_type == 'mensual':
            # Extraer mes y año de fecha_fin
            fecha_obj = datetime.strptime(fecha_fin, '%Y-%m-%d')
            mes_nombre = fecha_obj.strftime('%B %Y')

            prompt = f"""Eres un analista energético experto. Necesito que investigues en internet las posibles causas de una anomalía en la demanda energética.

**Contexto:**
- UCP: {ucp}, Colombia
- Periodo: {mes_nombre} (del {fecha_inicio} al {fecha_fin})
- Tipo de anomalía: Error mensual de predicción de {mape_total:.2f}% (superior al límite del 5%)

**Tarea:**
Busca en internet eventos, acontecimientos o condiciones que pudieron haber ocurrido en {ucp}, Colombia durante {mes_nombre} que pudieron causar variaciones significativas en la demanda de energía eléctrica durante todo el mes.

Considera:
- Condiciones climáticas atípicas del mes (sequías, lluvias intensas, temperaturas anormales)
- Cambios en la actividad industrial o comercial
- Eventos recurrentes durante el mes
- Temporadas vacacionales o escolares
- Restricciones energéticas o racionamientos
- Crecimiento poblacional o cambios demográficos
- cualquier otro acontecimiento relevante

Proporciona un análisis conciso (máximo 2-3 oraciones) con las causas más probables encontradas."""

        else:  # 'ambos'
            dias_str = ', '.join(dias_consecutivos) if dias_consecutivos else 'últimos 2 días'
            fecha_obj = datetime.strptime(fecha_fin, '%Y-%m-%d')
            mes_nombre = fecha_obj.strftime('%B %Y')

            prompt = f"""Eres un analista energético experto. Necesito que investigues en internet las posibles causas de una anomalía severa en la demanda energética.

**Contexto:**
- UCP: {ucp}, Colombia
- Periodo mensual: {mes_nombre} (del {fecha_inicio} al {fecha_fin})
- Error mensual: {mape_total:.2f}% (superior al límite del 5%)
- Días consecutivos afectados: {dias_str} (errores > 5%)

**Tarea:**
Busca en internet eventos o condiciones que pudieron causar tanto el error mensual sostenido como los picos específicos en las fechas {dias_str} en {ucp}, Colombia.

Proporciona un análisis conciso (máximo 3-4 oraciones) con las causas más probables encontradas, conectando los eventos puntuales con las tendencias mensuales."""

        logger.info(f"🤖 Consultando OpenAI (gpt-5-mini Responses API) para análisis de causalidad ({error_type})...")

        # Llamar a OpenAI con búsqueda en internet habilitada
        # Nota: GPT-5-mini usa la nueva Responses API con web_search nativo
        response = await run_in_threadpool(
            lambda: client.responses.create(
                model="gpt-5-mini",  # Modelo GPT-5-mini con capacidad de búsqueda web
                input=[  # NOTA: Responses API usa 'input' en lugar de 'messages'
                    {
                        "role": "user",
                        "content": f"Eres un analista experto en sistemas energéticos y demanda eléctrica en Colombia. Proporcionas análisis concisos basados en información factual encontrada en internet.\n\n{prompt}"
                    }
                ],
                tools=[
                    {
                        "type": "web_search"  # Herramienta nativa de búsqueda web
                    }
                ]
            )
        )

        # La respuesta viene directamente en output_text en la nueva Responses API
        analysis = response.output_text.strip()
        logger.info(f"✓ Análisis de OpenAI recibido: {len(analysis)} caracteres")

        return analysis

    except Exception as e:
        logger.error(f"Error en análisis de OpenAI: {e}")
        logger.error(traceback.format_exc())
        return f"Análisis automático no disponible (error: {str(e)})"


def check_model_exists(ucp: str) -> Tuple[bool, Optional[Path]]:
    """
    Verifica si existe un modelo entrenado en el registro para un UCP específico

    Args:
        ucp: Nombre del UCP (ej: 'Atlantico', 'Oriente')

    Returns:
        Tupla (existe: bool, path: Optional[Path])
    """
    models_dir = Path(f'models/{ucp}/trained')
    registry_path = Path(f'models/{ucp}/registry/champion_model.joblib')

    # Prioridad 1: Modelo campeón en registry
    if registry_path.exists():
        logger.info(f"✓ Modelo campeón encontrado para {ucp}: {registry_path}")
        return True, registry_path

    # Prioridad 2: Último modelo entrenado (por timestamp)
    if models_dir.exists():
        model_files = sorted(models_dir.glob('*.joblib'), key=lambda p: p.stat().st_mtime, reverse=True)
        if model_files:
            logger.info(f"✓ Último modelo entrenado encontrado para {ucp}: {model_files[0]}")
            return True, model_files[0]

    logger.warning(f"⚠ No se encontró ningún modelo entrenado para {ucp}")
    return False, None


def train_model_if_needed(df_with_features: pd.DataFrame,
                         ucp: str,
                         force_retrain: bool = False) -> Tuple[Path, Dict[str, Any]]:
    """
    Entrena los 3 modelos (XGBoost, LightGBM, RandomForest) y selecciona automáticamente el mejor
    basándose en rMAPE de validación

    Args:
        df_with_features: DataFrame con features procesados
        ucp: Nombre del UCP (ej: 'Atlantico', 'Oriente')
        force_retrain: Forzar reentrenamiento

    Returns:
        Tupla (model_path: Path, metrics: Dict con métricas del mejor modelo)
    """
    model_exists, model_path = check_model_exists(ucp)

    if model_exists and not force_retrain:
        logger.info("✓ Usando modelo existente (no se requiere entrenamiento)")
        # model_path ya está verificado que no es None
        assert model_path is not None
        return model_path, {}

    logger.info("="*80)
    logger.info("🔧 INICIANDO ENTRENAMIENTO AUTOMÁTICO DE MODELOS (SIN LAGS)")
    logger.info("="*80)

    # IMPORTANTE: Excluir features de lag para evitar train-test mismatch en predicción recursiva
    FEATURES_LAG_TO_EXCLUDE = [
        'total_lag_1d', 'total_lag_7d', 'total_lag_14d',
        'p8_lag_1d', 'p8_lag_7d',
        'p12_lag_1d', 'p12_lag_7d',
        'p18_lag_1d', 'p18_lag_7d',
        'p20_lag_1d', 'p20_lag_7d',
        'total_day_change', 'total_day_change_pct'
    ]

    # Preparar datos para entrenamiento
    exclude_cols = ['FECHA', 'fecha', 'TOTAL', 'demanda_total'] + [f'P{i}' for i in range(1, 25)]
    exclude_cols.extend(FEATURES_LAG_TO_EXCLUDE)  # ← AGREGAR LAGS A EXCLUSIÓN
    
    feature_cols = [col for col in df_with_features.columns if col not in exclude_cols]
    
    logger.info(f"  ⚠️  Excluyendo {len(FEATURES_LAG_TO_EXCLUDE)} features de lag para mejor predicción recursiva")

    # Normalizar nombres de columnas
    target_col = 'TOTAL' if 'TOTAL' in df_with_features.columns else 'demanda_total'

    X = df_with_features[feature_cols].fillna(0)
    y = df_with_features[target_col].copy()

    # Eliminar NaN en target
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]

    logger.info(f"  Total registros: {len(X)}")
    logger.info(f"  Features totales: {len(df_with_features.columns) - len(exclude_cols)}")
    logger.info(f"  Features usados: {len(feature_cols)} (sin lags)")
    logger.info(f"  Features excluidos: {len(FEATURES_LAG_TO_EXCLUDE)} lags")

    # Split temporal (80% train, 20% validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"  Train: {len(X_train)} registros")
    logger.info(f"  Validation: {len(X_val)} registros")

    # Entrenar TODOS los modelos
    logger.info(f"\n🚀 Entrenando los 3 modelos para {ucp} (XGBoost, LightGBM, RandomForest)...")

    trainer = ModelTrainer(
        optimize_hyperparams=False,  # Deshabilitado para velocidad
        cv_splits=3
    )

    trained_models = trainer.train_all_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        models=['xgboost', 'lightgbm', 'randomforest']
    )

    # Seleccionar automáticamente el MEJOR modelo basado en rMAPE
    logger.info("\n🏆 Seleccionando mejor modelo basado en rMAPE de validación...")

    best_name, _, best_results = trainer.select_best_model(
        criterion='rmape',  # Usar rMAPE como criterio (métrica más robusta)
        use_validation=True
    )

    logger.info(f"✓ Mejor modelo seleccionado para {ucp}: {best_name.upper()}")
    logger.info(f"  MAPE: {best_results['val_metrics']['mape']:.4f}%")
    logger.info(f"  rMAPE: {best_results['val_metrics']['rmape']:.4f}")
    logger.info(f"  R²: {best_results['val_metrics']['r2']:.4f}")
    logger.info(f"  MAE: {best_results['val_metrics']['mae']:.2f}")

    # Guardar TODOS los modelos en directorio específico del UCP
    models_dir = Path(f'models/{ucp}/trained')
    models_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = trainer.save_all_models(overwrite=True, output_dir=str(models_dir))

    # Path del mejor modelo
    best_model_path = saved_paths[best_name]
    # Asegurar que es un Path object
    best_model_path = Path(best_model_path) if isinstance(best_model_path, str) else best_model_path

    # Guardar el MEJOR modelo como campeón en registry del UCP
    registry_dir = Path(f'models/{ucp}/registry')
    registry_dir.mkdir(parents=True, exist_ok=True)
    champion_path = registry_dir / 'champion_model.joblib'

    import shutil
    shutil.copy(best_model_path, champion_path)

    logger.info(f"\n✓ Modelos guardados en: models/{ucp}/trained/")
    for name, path in saved_paths.items():
        status = "🏆 CAMPEÓN" if name == best_name else ""
        # Asegurar que path es un Path object
        path_obj = Path(path) if isinstance(path, str) else path
        logger.info(f"    {name}: {path_obj.name} {status}")

    logger.info(f"✓ Modelo campeón actualizado para {ucp}: {champion_path}")
    logger.info("="*80)

    # Métricas del mejor modelo
    metrics = {
        'modelo_seleccionado': best_name,
        'mape': best_results['val_metrics']['mape'],
        'rmape': best_results['val_metrics']['rmape'],
        'r2': best_results['val_metrics']['r2'],
        'mae': best_results['val_metrics']['mae'],
        'comparacion_modelos': {
            name: {
                'mape': results[1]['val_metrics']['mape'],
                'rmape': results[1]['val_metrics']['rmape'],
                'r2': results[1]['val_metrics']['r2']
            }
            for name, results in trained_models.items()
        }
    }

    return champion_path, metrics


def check_hourly_disaggregation_trained(ucp: str) -> bool:
    """
    Verifica si el sistema de desagregación horaria está entrenado para un UCP específico

    Args:
        ucp: Nombre del UCP (ej: 'Atlantico', 'Oriente')

    Returns:
        bool: True si está entrenado
    """
    normal_path = Path(f'models/{ucp}/hourly_disaggregator.pkl')
    special_path = Path(f'models/{ucp}/special_days_disaggregator.pkl')

    return normal_path.exists() and special_path.exists()


def train_hourly_disaggregation_if_needed(df_with_features: pd.DataFrame, ucp: str, force_retrain: bool = False):
    """
    Entrena sistema de desagregación horaria si no existe para un UCP específico

    Args:
        df_with_features: DataFrame con datos históricos y features
        ucp: Nombre del UCP (ej: 'Atlantico', 'Oriente')
        force_retrain: Si True, reentrena los clusters incluso si ya existen
    """
    if check_hourly_disaggregation_trained(ucp) and not force_retrain:
        logger.info(f"✓ Sistema de desagregación horaria ya está entrenado para {ucp}")
        return
    
    if force_retrain:
        logger.info(f"🔄 Reentrenando sistema de desagregación horaria para {ucp} (force_retrain=True)...")

    logger.info("="*80)
    logger.info(f"🔧 ENTRENANDO SISTEMA DE DESAGREGACIÓN HORARIA PARA {ucp}")
    logger.info("="*80)

    try:
        # Crear directorio para modelos del UCP
        models_ucp_dir = Path(f'models/{ucp}')
        models_ucp_dir.mkdir(parents=True, exist_ok=True)

        engine = HourlyDisaggregationEngine(auto_load=False, ucp=ucp)

        # Normalizar nombre de columna de fecha
        df_temp = df_with_features.copy()
        # if 'FECHA' in df_temp.columns:
        #     df_temp.rename(columns={'FECHA': 'fecha'}, inplace=True)

        # Guardar temporal para entrenamiento
        temp_path = Path(f'data/features/{ucp}/temp_for_training.csv')
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        df_temp.to_csv(temp_path, index=False)

        engine.train_all(
            data_path=temp_path,
            n_clusters_normal=35,
            n_clusters_special=15,
            save=True,
            output_dir=str(models_ucp_dir)
        )

        # Eliminar temporal
        if temp_path.exists():
            temp_path.unlink()

        logger.info(f"✓ Sistema de desagregación horaria entrenado para {ucp}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Error entrenando desagregación horaria para {ucp}: {e}")
        logger.warning("Se usarán placeholders para distribución horaria")


# ============================================================================
# ENDPOINTS
# ============================================================================


@app.post("/Error-feedback", response_model=ReasonResponse, status_code=status.HTTP_200_OK)
async def predict_demand(request: ReasonRequest):
    """
    Genera predicción de demanda energética para los próximos N días con granularidad horaria

    Flujo:
    1. Ejecuta pipeline de feature engineering con datos históricos hasta ayer
    2. Verifica si existe modelo entrenado (o entrena uno nuevo si se requiere)
    3. Genera predicción para los próximos N días
    4. Desagrega cada día en 24 períodos horarios (P1-P24) usando clustering K-Means
    5. Retorna array JSON con predicciones completas

    Args:
        request: PredictRequest con parámetros de la predicción

    Returns:
        PredictResponse con array de predicciones horarias

    Raises:
        HTTPException: Si hay error en algún paso del proceso
    """
    try:
        logger.info("="*80)
        logger.info("🚀 INICIANDO PREDICCIÓN DE DEMANDA")
        logger.info("="*80)

        # ====================================================================
        # PASO 1: EJECUTAR PIPELINE DE FEATURE ENGINEERING
        # ====================================================================
        logger.info(f"\n📊 PASO 1: Procesando datos históricos y creando features para {request.ucp}...")
        await run_in_threadpool(full_update_csv, request.ucp)
        try:
            # Paths dinámicos basados en UCP
            power_data_path = f'data/raw/{request.ucp}/datos.csv'
            weather_data_path = f'data/raw/{request.ucp}/clima_new.csv'
            output_dir = Path(f'data/features/{request.ucp}')

            df_with_features, _ = run_automated_pipeline(
                power_data_path=power_data_path,
                weather_data_path=weather_data_path,
                start_date='2015-01-01',
                end_date=request.end_date,
                output_dir=output_dir
            )

            logger.info(f"✓ Pipeline completado para {request.ucp}: {len(df_with_features)} registros con {len(df_with_features.columns)} columnas")

        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Archivo no encontrado: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error en pipeline de datos: {str(e)}"
            )

        # ====================================================================
        # PASO 2: VERIFICAR/ENTRENAR MODELO
        # ====================================================================
        logger.info(f"\n🤖 PASO 2: Verificando modelo de predicción para {request.ucp}...")

        try:
            model_path, train_metrics = train_model_if_needed(
                df_with_features=df_with_features,
                ucp=request.ucp,
                force_retrain=request.force_retrain
            )

            modelo_entrenado = len(train_metrics) > 0

            if modelo_entrenado:
                logger.info(f"✓ Modelo entrenado exitosamente")
                logger.info(f"  MAPE: {train_metrics['mape']:.4f}%")
                logger.info(f"  rMAPE: {train_metrics['rmape']:.4f}")
                logger.info(f"  R²: {train_metrics['r2']:.4f}")
            else:
                logger.info(f"✓ Usando modelo existente: {model_path.name}")

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error en entrenamiento de modelo: {str(e)}"
            )

        # ====================================================================
        # PASO 3: VERIFICAR/ENTRENAR SISTEMA DE DESAGREGACIÓN HORARIA
        # ====================================================================
        logger.info(f"\n⏰ PASO 3: Verificando sistema de desagregación horaria para {request.ucp}...")

        try:
            logger.info(f"   Verificando si desagregación horaria está entrenada...")
            train_hourly_disaggregation_if_needed(df_with_features, request.ucp, force_retrain=request.force_retrain)
            logger.info(f"Desagrecacion horaria se ejecuta")

        except Exception as e:
            logger.warning(f"⚠ Error en desagregación horaria: {e}")
            logger.warning("Se continuará con placeholders")

        # ====================================================================
        # PASO 4: GENERAR PREDICCIONES
        # ====================================================================
        
        try:












            climate_raw_path = f'data/raw/{request.ucp}/clima_new.csv'






            df_try_features = df_with_features.copy()
            max_date = df_with_features['FECHA'].max()
            cut_date = max_date - pd.Timedelta(days=30)

            df_try_features = df_try_features[df_try_features['FECHA'] <= cut_date]        
            temp_try_path = f'data/features/{request.ucp}/temp_api_features_try.csv'
            df_try_features.to_csv(temp_try_path, index=False)
            # Inicializar pipeline de predicción con datos RECIEN PROCESADOS
            pipeline = ForecastPipeline(
                model_path=str(model_path),
                historical_data_path=temp_try_path,
                festivos_path='config/festivos.json',
                enable_hourly_disaggregation=True,  # ← Habilitado con nuevo modelo
                raw_climate_path=climate_raw_path,
                ucp=request.ucp  # ← Pasar UCP al pipeline
            )
            check_date=max_date - pd.Timedelta(days=29)
            # Generar predicciones
            predictions_df = pipeline.predict_next_n_days(n_days=30)
            print('predictions_df'*40)
            print(predictions_df)
            
            mape_check_df=df_with_features[df_with_features['FECHA'] >= check_date] 
            print(mape_check_df)  
            
            import numpy as np
            

            # --- Alinear por fecha ---
            pred = predictions_df.copy()
            real = mape_check_df.copy()

            pred['fecha'] = pd.to_datetime(pred['fecha'])
            real['FECHA'] = pd.to_datetime(real['FECHA'])

            # Unimos por fecha
            df_merged = pred.merge(real, left_on='fecha', right_on='FECHA', how='inner')
            print(df_merged.columns)
            # ============================
            # 1️⃣ MAPE TOTAL (demanda_predicha vs TOTAL)
            # ============================
            df_merged['abs_pct_error'] = np.abs(
                (df_merged['demanda_predicha'] - df_merged['TOTAL']) *100/ df_merged['TOTAL']
            )

            print(df_merged[['abs_pct_error','demanda_predicha','TOTAL']])
            # Condición: error mayor al 5%
            cond = df_merged['abs_pct_error'] > 5
            print(cond)
            # Detectar si hay dos True consecutivos
            hay_dos_seguidos = (cond & cond.shift(1)).any()

            print("¿Hay dos errores seguidos > 5%?:", hay_dos_seguidos)
            
            mape_total = df_merged['abs_pct_error'].mean()
            print("MAPE TOTAL:", mape_total)
            print("MAPE TOTAL:", df_merged[['abs_pct_error','demanda_predicha','TOTAL']])

            print('df_try_features'*40)

            # ====================================================================
            # ANÁLISIS DE CAUSALIDAD CON OPENAI (si se requiere reentrenamiento)
            # ====================================================================

            # Determinar si se requiere reentrenamiento y tipo de error
            if mape_total > 5 and hay_dos_seguidos:
                should_retrain = True
                error_type = 'ambos'
                reason_base = f'Error mensual superior al 5% (MAPE: {mape_total:.2f}%) y dos días consecutivos con error superior al 5%'
                logger.info(f"⚠ MAPE TOTAL: {mape_total:.2f}%. Se requiere reentrenamiento.")
            elif hay_dos_seguidos:
                should_retrain = True
                error_type = 'consecutivo'
                reason_base = f'Dos días consecutivos con error superior al 5% (MAPE mensual: {mape_total:.2f}%)'
                logger.info(f"⚠ MAPE TOTAL: {mape_total:.2f}%. Se requiere reentrenamiento.")
            elif mape_total > 5:
                should_retrain = True
                error_type = 'mensual'
                reason_base = f'Error mensual superior al 5% (MAPE: {mape_total:.2f}%)'
                logger.info(f"⚠ MAPE TOTAL: {mape_total:.2f}%. Se requiere reentrenamiento.")
            else:
                should_retrain = False
                error_type = None
                reason = f'Error dentro de límites aceptables (MAPE: {mape_total:.2f}%)'
                logger.info(f"✓ MAPE TOTAL: {mape_total:.2f}%. No se requiere reentrenamiento.")

            # Si se requiere reentrenamiento, analizar causas con OpenAI
            if should_retrain:
                logger.info("="*80)
                logger.info("🔍 ANALIZANDO CAUSAS DEL ERROR CON OPENAI")
                logger.info("="*80)

                # Extraer fechas de días con errores consecutivos si aplica
                dias_consecutivos = None
                if error_type in ['consecutivo', 'ambos']:
                    # Encontrar los días consecutivos con error > 5%
                    mask_consecutivos = cond & cond.shift(1)
                    indices_consecutivos = df_merged.index[mask_consecutivos].tolist()

                    # Convertir a datetime si no lo está y formatear
                    if len(indices_consecutivos) > 0:
                        fechas_consecutivas = []
                        for idx in indices_consecutivos:
                            fecha_val = df_merged.loc[idx, 'FECHA']
                            if isinstance(fecha_val, pd.Timestamp):
                                fechas_consecutivas.append(fecha_val.strftime('%Y-%m-%d'))
                            else:
                                fechas_consecutivas.append(str(fecha_val))

                        # También incluir el día anterior al primero marcado
                        if fechas_consecutivas and indices_consecutivos:
                            primer_idx = indices_consecutivos[0]
                            if primer_idx > 0:
                                fecha_ant_val = df_merged.loc[primer_idx - 1, 'FECHA']
                                if isinstance(fecha_ant_val, pd.Timestamp):
                                    fecha_anterior = fecha_ant_val.strftime('%Y-%m-%d')
                                else:
                                    fecha_anterior = str(fecha_ant_val)
                                dias_consecutivos = [fecha_anterior] + fechas_consecutivas
                            else:
                                dias_consecutivos = fechas_consecutivas

                    logger.info(f"  Días consecutivos identificados: {dias_consecutivos}")

                # Obtener rango de fechas del análisis
                fecha_min = df_merged['FECHA'].min()
                fecha_max = df_merged['FECHA'].max()

                if isinstance(fecha_min, pd.Timestamp):
                    fecha_inicio_analisis = fecha_min.strftime('%Y-%m-%d')
                else:
                    fecha_inicio_analisis = str(fecha_min)

                if isinstance(fecha_max, pd.Timestamp):
                    fecha_fin_analisis = fecha_max.strftime('%Y-%m-%d')
                else:
                    fecha_fin_analisis = str(fecha_max)

                # Llamar a OpenAI para análisis de causalidad
                openai_analysis = await analyze_error_with_openai(
                    ucp=request.ucp,
                    error_type=error_type,
                    mape_total=mape_total,
                    fecha_inicio=fecha_inicio_analisis,
                    fecha_fin=fecha_fin_analisis,
                    dias_consecutivos=dias_consecutivos
                )

                # Combinar reason base con análisis de OpenAI
                reason = f"{reason_base}// Análisis de causas: {openai_analysis}"

                logger.info(f"✓ Análisis de causalidad agregado al reporte")
                logger.info("="*80)
                return ReasonResponse(
                    reason=reason
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error generando predicciones: {str(e)}"
            )


    except Exception as e:        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en proceso de predicción: {str(e)}"
        )





















async def run_predict_flow(request: PredictRequest) -> PredictResponse:
    """
    Genera predicción de demanda energética para los próximos N días con granularidad horaria

    Flujo:
    1. Ejecuta pipeline de feature engineering con datos históricos hasta ayer
    2. Verifica si existe modelo entrenado (o entrena uno nuevo si se requiere)
    3. Genera predicción para los próximos N días
    4. Desagrega cada día en 24 períodos horarios (P1-P24) usando clustering K-Means
    5. Retorna array JSON con predicciones completas

    Args:
        request: PredictRequest con parámetros de la predicción

    Returns:
        PredictResponse con array de predicciones horarias

    Raises:
        HTTPException: Si hay error en algún paso del proceso
    """
    try:
        logger.info("="*80)
        logger.info("🚀 INICIANDO PREDICCIÓN DE DEMANDA")
        logger.info("="*80)

        # ====================================================================
        # PASO 1: EJECUTAR PIPELINE DE FEATURE ENGINEERING
        # ====================================================================
        logger.info(f"\n📊 PASO 1: Procesando datos históricos y creando features para {request.ucp}...")
        await run_in_threadpool(full_update_csv, request.ucp)
        try:
            # Paths dinámicos basados en UCP
            power_data_path = f'data/raw/{request.ucp}/datos.csv'
            weather_data_path = f'data/raw/{request.ucp}/clima_new.csv'
            output_dir = Path(f'data/features/{request.ucp}')

            df_with_features, _ = run_automated_pipeline(
                power_data_path=power_data_path,
                weather_data_path=weather_data_path,
                start_date='2015-01-01',
                end_date=request.end_date,
                output_dir=output_dir
            )

            logger.info(f"✓ Pipeline completado para {request.ucp}: {len(df_with_features)} registros con {len(df_with_features.columns)} columnas")

        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Archivo no encontrado: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error en pipeline de datos: {str(e)}"
            )

        # ====================================================================
        # PASO 2: VERIFICAR/ENTRENAR MODELO
        # ====================================================================
        logger.info(f"\n🤖 PASO 2: Verificando modelo de predicción para {request.ucp}...")

        try:
            model_path, train_metrics = train_model_if_needed(
                df_with_features=df_with_features,
                ucp=request.ucp,
                force_retrain=request.force_retrain
            )

            modelo_entrenado = len(train_metrics) > 0

            if modelo_entrenado:
                logger.info(f"✓ Modelo entrenado exitosamente")
                logger.info(f"  MAPE: {train_metrics['mape']:.4f}%")
                logger.info(f"  rMAPE: {train_metrics['rmape']:.4f}")
                logger.info(f"  R²: {train_metrics['r2']:.4f}")
            else:
                logger.info(f"✓ Usando modelo existente: {model_path.name}")

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error en entrenamiento de modelo: {str(e)}"
            )

        # ====================================================================
        # PASO 3: VERIFICAR/ENTRENAR SISTEMA DE DESAGREGACIÓN HORARIA
        # ====================================================================
        logger.info(f"\n⏰ PASO 3: Verificando sistema de desagregación horaria para {request.ucp}...")

        try:
            logger.info(f"   Verificando si desagregación horaria está entrenada...")
            train_hourly_disaggregation_if_needed(df_with_features, request.ucp, force_retrain=request.force_retrain)
            logger.info(f"Desagrecacion horaria se ejecuta")

        except Exception as e:
            logger.warning(f"⚠ Error en desagregación horaria: {e}")
            logger.warning("Se continuará con placeholders")

        # ====================================================================
        # PASO 4: GENERAR PREDICCIONES
        # ====================================================================
        logger.info(f"\n🔮 PASO 4: Generando predicciones para {request.n_days} días...")

        try:












            climate_raw_path = f'data/raw/{request.ucp}/clima_new.csv'






            df_try_features = df_with_features.copy()
            max_date = df_with_features['FECHA'].max()
            cut_date = max_date - pd.Timedelta(days=30)

            df_try_features = df_try_features[df_try_features['FECHA'] <= cut_date]        
            temp_try_path = f'data/features/{request.ucp}/temp_api_features_try.csv'
            df_try_features.to_csv(temp_try_path, index=False)
            # Inicializar pipeline de predicción con datos RECIEN PROCESADOS
            pipeline = ForecastPipeline(
                model_path=str(model_path),
                historical_data_path=temp_try_path,
                festivos_path='config/festivos.json',
                enable_hourly_disaggregation=True,  # ← Habilitado con nuevo modelo
                raw_climate_path=climate_raw_path,
                ucp=request.ucp  # ← Pasar UCP al pipeline
            )
            check_date=max_date - pd.Timedelta(days=29)
            # Generar predicciones
            predictions_df = pipeline.predict_next_n_days(n_days=30)
            print('predictions_df'*40)
            print(predictions_df)
            
            mape_check_df=df_with_features[df_with_features['FECHA'] >= check_date] 
            print(mape_check_df)  
            
            import numpy as np
            

            # --- Alinear por fecha ---
            pred = predictions_df.copy()
            real = mape_check_df.copy()

            pred['fecha'] = pd.to_datetime(pred['fecha'])
            real['FECHA'] = pd.to_datetime(real['FECHA'])

            # Unimos por fecha
            df_merged = pred.merge(real, left_on='fecha', right_on='FECHA', how='inner')
            print(df_merged.columns)
            # ============================
            # 1️⃣ MAPE TOTAL (demanda_predicha vs TOTAL)
            # ============================
            df_merged['abs_pct_error'] = np.abs(
                (df_merged['demanda_predicha'] - df_merged['TOTAL']) *100/ df_merged['TOTAL']
            )
            # cols_xy = ["FECHA"]+[f'P{i}_x' for i in range(1, 25)] + [f'P{i}_y' for i in range(1, 25)]
        

            # def calcular_mape_por_dia(df):
            #     resultados = []

            #     for idx, row in df.iterrows():
            #         fecha = row["FECHA"]

            #         # Extraer columnas reales y predichas
            #         reales = [row[f"P{h}_x"] for h in range(1,25)]
            #         preds  = [row[f"P{h}_y"] for h in range(1,25)]

            #         # Calcular MAPE hora por hora
            #         errores = []
            #         for r, p in zip(reales, preds):
            #             if r == 0:
            #                 errores.append(0)
            #             else:
            #                 errores.append(abs((r - p) / r))

            #         mape_dia = np.mean(errores) * 100

            #         resultados.append({
            #             "FECHA": fecha,
            #             "MAPE": mape_dia
            #         })

            #     return pd.DataFrame(resultados)

            # df_mape = calcular_mape_por_dia(df_merged[cols_xy])
            #print(df_mape)
            print(df_merged[['abs_pct_error','demanda_predicha','TOTAL']])
            # Condición: error mayor al 5%
            cond = df_merged['abs_pct_error'] > 5
            print(cond)
            # Detectar si hay dos True consecutivos
            hay_dos_seguidos = (cond & cond.shift(1)).any()

            print("¿Hay dos errores seguidos > 5%?:", hay_dos_seguidos)
            
            mape_total = df_merged['abs_pct_error'].mean()
            print("MAPE TOTAL:", mape_total)
            print("MAPE TOTAL:", df_merged[['abs_pct_error','demanda_predicha','TOTAL']])

            print('df_try_features'*40)

            # ====================================================================
            # ANÁLISIS DE CAUSALIDAD CON OPENAI (si se requiere reentrenamiento)
            # ====================================================================

            # Determinar si se requiere reentrenamiento y tipo de error
            if mape_total > 5 and hay_dos_seguidos:
                should_retrain = True
                error_type = 'ambos'
                reason_base = f'Error mensual superior al 5% (MAPE: {mape_total:.2f}%) y dos días consecutivos con error superior al 5%'
                logger.info(f"⚠ MAPE TOTAL: {mape_total:.2f}%. Se requiere reentrenamiento.")
            elif hay_dos_seguidos:
                should_retrain = True
                error_type = 'consecutivo'
                reason_base = f'Dos días consecutivos con error superior al 5% (MAPE mensual: {mape_total:.2f}%)'
                logger.info(f"⚠ MAPE TOTAL: {mape_total:.2f}%. Se requiere reentrenamiento.")
            elif mape_total > 5:
                should_retrain = True
                error_type = 'mensual'
                reason_base = f'Error mensual superior al 5% (MAPE: {mape_total:.2f}%)'
                logger.info(f"⚠ MAPE TOTAL: {mape_total:.2f}%. Se requiere reentrenamiento.")
            else:
                should_retrain = False
                error_type = None
                reason = f'Error dentro de límites aceptables (MAPE: {mape_total:.2f}%)'
                logger.info(f"✓ MAPE TOTAL: {mape_total:.2f}%. No se requiere reentrenamiento.")

            # Si se requiere reentrenamiento, analizar causas con OpenAI
            if should_retrain:
                logger.info("="*80)
                logger.info("🔍 ANALIZANDO CAUSAS DEL ERROR CON OPENAI")
                logger.info("="*80)

                # Extraer fechas de días con errores consecutivos si aplica
                dias_consecutivos = None
                if error_type in ['consecutivo', 'ambos']:
                    # Encontrar los días consecutivos con error > 5%
                    mask_consecutivos = cond & cond.shift(1)
                    indices_consecutivos = df_merged.index[mask_consecutivos].tolist()

                    # Convertir a datetime si no lo está y formatear
                    if len(indices_consecutivos) > 0:
                        fechas_consecutivas = []
                        for idx in indices_consecutivos:
                            fecha_val = df_merged.loc[idx, 'FECHA']
                            if isinstance(fecha_val, pd.Timestamp):
                                fechas_consecutivas.append(fecha_val.strftime('%Y-%m-%d'))
                            else:
                                fechas_consecutivas.append(str(fecha_val))

                        # También incluir el día anterior al primero marcado
                        if fechas_consecutivas and indices_consecutivos:
                            primer_idx = indices_consecutivos[0]
                            if primer_idx > 0:
                                fecha_ant_val = df_merged.loc[primer_idx - 1, 'FECHA']
                                if isinstance(fecha_ant_val, pd.Timestamp):
                                    fecha_anterior = fecha_ant_val.strftime('%Y-%m-%d')
                                else:
                                    fecha_anterior = str(fecha_ant_val)
                                dias_consecutivos = [fecha_anterior] + fechas_consecutivas
                            else:
                                dias_consecutivos = fechas_consecutivas

                    logger.info(f"  Días consecutivos identificados: {dias_consecutivos}")

                # Obtener rango de fechas del análisis
                fecha_min = df_merged['FECHA'].min()
                fecha_max = df_merged['FECHA'].max()

                if isinstance(fecha_min, pd.Timestamp):
                    fecha_inicio_analisis = fecha_min.strftime('%Y-%m-%d')
                else:
                    fecha_inicio_analisis = str(fecha_min)

                if isinstance(fecha_max, pd.Timestamp):
                    fecha_fin_analisis = fecha_max.strftime('%Y-%m-%d')
                else:
                    fecha_fin_analisis = str(fecha_max)

                # Llamar a OpenAI para análisis de causalidad
                # openai_analysis = await analyze_error_with_openai(
                #     ucp=request.ucp,
                #     error_type=error_type,
                #     mape_total=mape_total,
                #     fecha_inicio=fecha_inicio_analisis,
                #     fecha_fin=fecha_fin_analisis,
                #     dias_consecutivos=dias_consecutivos
                # )

                # Combinar reason base con análisis de OpenAI
                reason = f"{reason_base}"

                logger.info(f"✓ Análisis de causalidad agregado al reporte")
                logger.info("="*80)

























            # Determinar ruta de datos climáticos RAW específicos del UCP
            climate_raw_path = f'data/raw/{request.ucp}/clima_new.csv'

            # CRITICO: Guardar datos procesados temporalmente en directorio del UCP
            temp_features_path = f'data/features/{request.ucp}/temp_api_features.csv'
            df_with_features.to_csv(temp_features_path, index=False)
            
            # Log datos guardados (detectar columna de fecha)
            if 'FECHA' in df_with_features.columns:
                logger.info(f"   Última fecha en temp: {df_with_features['FECHA'].max()}")
            elif 'fecha' in df_with_features.columns:
                print('fecha'*40)
                logger.info(f"   Última fecha en temp: {df_with_features['fecha'].max()}")
            else:
                logger.info(f"   Datos guardados en temp (sin columna fecha explícita)")
            logger.info(f"   Total filas: {len(df_with_features)}")
            
            # Inicializar pipeline de predicción con datos RECIEN PROCESADOS
            pipeline = ForecastPipeline(
                model_path=str(model_path),
                historical_data_path=temp_features_path,
                festivos_path='config/festivos.json',
                enable_hourly_disaggregation=True,  # ← Habilitado con nuevo modelo
                raw_climate_path=climate_raw_path,
                ucp=request.ucp  # ← Pasar UCP al pipeline
            )

            # Generar predicciones
            predictions_df = pipeline.predict_next_n_days(n_days=request.n_days)
            
            # Limpiar archivo temporal
            import os
            if os.path.exists(temp_features_path):
                os.remove(temp_features_path)

            logger.info(f"✓ Predicciones generadas: {len(predictions_df)} días")

            # ====================================================================
            # APLICAR OFFSET SCALAR (si se proporciona y es diferente de 1.0)
            # ====================================================================
            if request.offset_scalar is not None and request.offset_scalar > 0 and request.offset_scalar != 1.0:
                logger.info(f"\n🔧 Aplicando offset scalar: {request.offset_scalar}")
                
                # Aplicar escalar a demanda total
                predictions_df['demanda_predicha'] = predictions_df['demanda_predicha'] * request.offset_scalar
                
                # Aplicar escalar a todos los períodos horarios (P1-P24)
                for i in range(1, 25):
                    col_name = f'P{i}'
                    if col_name in predictions_df.columns:
                        predictions_df[col_name] = predictions_df[col_name] * request.offset_scalar
                
                logger.info(f"✓ Offset aplicado: todas las predicciones multiplicadas por {request.offset_scalar}")
            elif request.offset_scalar == 1.0:
                logger.info("✓ Offset scalar es 1.0, se omite (no se requiere ajuste)")
            else:
                logger.info("✓ No se aplicó offset scalar (no proporcionado o inválido)")

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error generando predicciones: {str(e)}\n{traceback.format_exc()}"
            )

        # ====================================================================
        # PASO 5: FORMATEAR RESPUESTA
        # ====================================================================
        logger.info("\n📋 PASO 5: Formateando respuesta...")

        try:
            # Convertir DataFrame a lista de diccionarios
            predictions_list = []

            # Mapeo de días de la semana
            dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

            for _, row in predictions_df.iterrows():
                fecha = pd.to_datetime(row['fecha'])

                # Determinar método de desagregación usado
                metodo = row.get('metodo_desagregacion', 'special' if row.get('is_festivo', False) else 'normal')

                prediction = {
                    'fecha': fecha.strftime('%Y-%m-%d'),
                    'dia_semana': dias_semana[row.get('dayofweek', fecha.dayofweek)],
                    'demanda_total': round(float(row['demanda_predicha']), 2),
                    'is_festivo': bool(row.get('is_festivo', False)),
                    'is_weekend': bool(row.get('is_weekend', False)),
                    'metodo_desagregacion': metodo,
                    'cluster_id': int(row['cluster_id']) if pd.notna(row.get('cluster_id')) else None,
                    **{f'P{i}': round(float(row.get(f'P{i}', 0)), 2) for i in range(1, 25)}
                }

                predictions_list.append(prediction)

            # Calcular estadísticas
            metadata = {
                'fecha_generacion': datetime.now().isoformat(),
                'modelo_usado': model_path.stem,
                'dias_predichos': len(predictions_df),
                'fecha_inicio': predictions_df['fecha'].min().strftime('%Y-%m-%d'),
                'fecha_fin': predictions_df['fecha'].max().strftime('%Y-%m-%d'),
                'demanda_promedio': round(float(predictions_df['demanda_predicha'].mean()), 2),
                'demanda_min': round(float(predictions_df['demanda_predicha'].min()), 2),
                'demanda_max': round(float(predictions_df['demanda_predicha'].max()), 2),
                'dias_laborables': int((predictions_df['is_weekend'] == False).sum()),
                'dias_fin_de_semana': int((predictions_df['is_weekend'] == True).sum()),
                'dias_festivos': int((predictions_df['is_festivo'] == True).sum()),
                'modelo_entrenado': modelo_entrenado,
                'metricas_modelo': train_metrics if modelo_entrenado else {}
            }

            logger.info("✓ Respuesta formateada correctamente")
            logger.info(f"  Demanda promedio: {metadata['demanda_promedio']:,.2f} MWh")
            logger.info(f"  Rango: {metadata['demanda_min']:,.2f} - {metadata['demanda_max']:,.2f} MWh")

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error formateando respuesta: {str(e)}"
            )

        # ====================================================================
        # PASO 6: OBTENER EVENTOS FUTUROS
        # ====================================================================
        

        # ====================================================================
        # RESPUESTA FINAL
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("✅ PREDICCIÓN COMPLETADA EXITOSAMENTE")
        logger.info("="*80 + "\n")

        return PredictResponse(
            status="success",
            message=f"Predicción generada exitosamente para {request.n_days} días con granularidad horaria",
            metadata=metadata,
            predictions=predictions_list,
            should_retrain=should_retrain,
            reason=reason,
           # events=events
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inesperado en el servidor: {str(e)}"
        )


@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_demand(request: PredictRequest):
    """Endpoint /predict que delega la lógica principal a run_predict_flow."""
    return await run_predict_flow(request)


def _get_historical_day_profile(ucp: str, date_str: str) -> Optional[HistoricalDay]:
    """
    Obtiene el perfil histórico de un día específico (P1-P24 y TOTAL si existe).
    Retorna None si no se encuentra o faltan columnas P1-P24.
    """
    candidate_paths = [
        Path(f"data/raw/{ucp}/datos.csv"),
        Path(f"data/features/{ucp}/data_with_features_latest.csv"),
        Path("data/raw/datos.csv"),
        Path("data/features/data_with_features_latest.csv"),
    ]

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if "FECHA" in df.columns:
                df = df.rename(columns={"FECHA": "fecha"})
            df["fecha"] = pd.to_datetime(df["fecha"])

            if "UCP" in df.columns:
                df = df[df["UCP"] == ucp]

            day_df = df[df["fecha"] == pd.to_datetime(date_str)]
            if day_df.empty:
                continue

            period_cols = [f"P{i}" for i in range(1, 25)]
            if not all(col in day_df.columns for col in period_cols):
                continue

            row = day_df.iloc[0]
            valores = {f"P{i}": float(row[f"P{i}"]) for i in range(1, 25)}
            if "TOTAL" in row:
                try:
                    valores["TOTAL"] = float(row["TOTAL"])
                except Exception:
                    pass

            return HistoricalDay(
                fecha=pd.to_datetime(row["fecha"]).strftime("%Y-%m-%d"),
                valores=valores
            )
        except Exception as e:
            logger.warning(f"⚠ No se pudo leer historial desde {path}: {e}")
            continue

    return None


@app.post("/predict-day", response_model=SingleDayPredictResponse, status_code=status.HTTP_200_OK)
async def predict_single_day(request: SingleDayPredictRequest):
    """
    Endpoint para generar predicción de un único día específico.
    Calcula n_days=1 y usa end_date = fecha objetivo - 1 día.
    Además retorna historial del día de referencia solicitado y comparaciones/ajustes.
    """
    try:
        target_dt = datetime.strptime(request.fecha, '%Y-%m-%d')
        derived_end_date = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')
        reference_dt = datetime.strptime(request.fecha_referencia, '%Y-%m-%d')

        predict_payload = PredictRequest(
            ucp=request.ucp,
            end_date=derived_end_date,
            n_days=1,
            force_retrain=request.force_retrain,
            offset_scalar=request.offset_scalar
        )

        logger.info(f"🎯 Predicción de un día: {request.fecha} (end_date derivado: {derived_end_date})")
        prediction_response = await run_predict_flow(predict_payload)

        # Historial: día de referencia solicitado
        reference_date = reference_dt.strftime('%Y-%m-%d')
        history: Dict[str, HistoricalDay] = {}

        ref_profile = _get_historical_day_profile(request.ucp, reference_date)
        if ref_profile:
            history["referencia"] = ref_profile
        else:
            logger.info(f"ℹ Sin historial para fecha_referencia ({reference_date})")

        # Comparaciones horarias contra históricos disponibles
        comparisons = SingleDayComparisons()
        adjustments = SingleDayAdjustments()
        if prediction_response.predictions:
            pred_day_dict = prediction_response.predictions[0].model_dump()
            # Total original para preservar luego en ajustes
            pred_total = sum(float(pred_day_dict.get(f"P{i}", 0.0)) for i in range(1, 25))

            def build_diffs(hist_profile: HistoricalDay) -> Dict[str, HourlyDiff]:
                diffs: Dict[str, HourlyDiff] = {}
                for i in range(1, 25):
                    key = f"P{i}"
                    if key not in hist_profile.valores or key not in pred_day_dict:
                        continue
                    pred_val = float(pred_day_dict[key])
                    hist_val = float(hist_profile.valores[key])
                    delta = pred_val - hist_val
                    if hist_val == 0:
                        delta_pct = None
                    else:
                        delta_pct = (delta / hist_val) * 100
                    if abs(delta) < 1e-9:
                        relacion = "igual"
                    elif delta > 0:
                        relacion = "superior"
                    else:
                        relacion = "inferior"
                    diffs[key] = HourlyDiff(
                        delta=delta,
                        delta_pct=delta_pct,
                        relacion=relacion
                    )
                return diffs

            def build_adjusted(hist_profile: HistoricalDay) -> AdjustedComparison:
                adjusted_values = {}
                period_cols = [f"P{i}" for i in range(1, 25)]

                # Paso 1: Ajustar solo horas con |delta_pct| > 5%
                for key in period_cols:
                    if key not in hist_profile.valores or key not in pred_day_dict:
                        continue
                    pred_val = float(pred_day_dict[key])
                    hist_val = float(hist_profile.valores[key])
                    if hist_val == 0:
                        adjusted_values[key] = pred_val
                        continue
                    delta_pct = (pred_val - hist_val) / hist_val * 100
                    if abs(delta_pct) > 5:
                        # Llevar el delta a exactamente ±5%
                        sign = 1 if delta_pct > 0 else -1
                        adjusted_values[key] = hist_val * (1 + sign * 0.05)
                    else:
                        adjusted_values[key] = pred_val

                # Si faltó alguna hora (por columnas faltantes), usar el valor original
                for key in period_cols:
                    if key not in adjusted_values and key in pred_day_dict:
                        adjusted_values[key] = float(pred_day_dict[key])

                # Paso 2: Re-escalar para conservar el total original
                adjusted_sum = sum(adjusted_values.values())
                if adjusted_sum > 0:
                    scale_factor = pred_total / adjusted_sum
                else:
                    scale_factor = 1.0

                final_profile = {
                    key: adjusted_values[key] * scale_factor
                    for key in period_cols
                    if key in adjusted_values
                }

                # Paso 3: Calcular deltas finales vs histórico
                final_diffs = build_diffs(
                    HistoricalDay(fecha=hist_profile.fecha, valores=final_profile)
                )

                return AdjustedComparison(
                    profile=final_profile,
                    adjusted_deltas=final_diffs,
                    scale_factor=scale_factor
                )

            if "referencia" in history:
                comparisons.referencia = build_diffs(history["referencia"])
                adjustments.referencia = build_adjusted(history["referencia"])

        return SingleDayPredictResponse(
            prediction=prediction_response,
            history=history,
            comparisons=comparisons,
            adjustments=adjustments
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /predict-day: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generando predicción de un día: {str(e)}"
        )


@app.post("/predict-day-scaled", response_model=SingleDayScaledResponse, status_code=status.HTTP_200_OK)
async def predict_single_day_scaled(request: SingleDayPredictRequest):
    """
    Endpoint para escalar el histórico (fecha_referencia) al total de la predicción del día solicitado.
    Retorna el factor de escala y el perfil histórico escalado.
    """
    try:
        target_dt = datetime.strptime(request.fecha, '%Y-%m-%d')
        derived_end_date = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')
        reference_dt = datetime.strptime(request.fecha_referencia, '%Y-%m-%d')

        predict_payload = PredictRequest(
            ucp=request.ucp,
            end_date=derived_end_date,
            n_days=1,
            force_retrain=request.force_retrain,
            offset_scalar=request.offset_scalar
        )

        logger.info(
            f"🎯 Predicción de un día (escalado histórico): {request.fecha} "
            f"(end_date derivado: {derived_end_date})"
        )
        prediction_response = await run_predict_flow(predict_payload)

        reference_date = reference_dt.strftime('%Y-%m-%d')
        history: Dict[str, HistoricalDay] = {}

        ref_profile = _get_historical_day_profile(request.ucp, reference_date)
        if ref_profile:
            history["referencia"] = ref_profile
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No hay histórico para fecha_referencia {reference_date}"
            )

        if not prediction_response.predictions:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No se generó predicción para el día solicitado"
            )

        period_cols = [f"P{i}" for i in range(1, 25)]
        hist_total = sum(float(ref_profile.valores.get(k, 0.0)) for k in period_cols)
        if hist_total <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El total del histórico es 0; no se puede calcular el factor de escala"
            )

        # Ajuste fijo: aumentar el histórico en 2%
        scale_factor = 1.02
        scaled_profile = {
            k: float(ref_profile.valores.get(k, 0.0)) * scale_factor
            for k in period_cols
        }
        total_scaled = hist_total * scale_factor

        return SingleDayScaledResponse(
            prediction=prediction_response,
            history=history,
            scaled=ScaledHistoricalProfile(
                profile=scaled_profile,
                scale_factor=scale_factor,
                total_pred=total_scaled,
                total_hist=hist_total
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /predict-day-scaled: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generando perfil escalado: {str(e)}"
        )


@app.post(
    "/predict-with-base-curve",
    response_model=PredictWithBaseCurveResponse,
    status_code=status.HTTP_200_OK
)
async def predict_with_base_curve(request: PredictWithBaseCurveRequest):
    """
    Endpoint combinado que ejecuta la lógica de /predict y /base-curve en una sola llamada.
    Genera la predicción y luego calcula la curva base para el mismo rango (o el provisto).
    """
    try:
        logger.info("\n🔀 Iniciando flujo combinado de predicción + curva base")

        # Determinar rango de fechas usando fecha_inicio/fecha_fin o, en su defecto, las fechas del array data
        fechas_candidatas: List[str] = []
        if request.fecha_inicio:
            fechas_candidatas.append(request.fecha_inicio)
        if request.fecha_fin:
            fechas_candidatas.append(request.fecha_fin)
        if request.data:
            fechas_candidatas.extend([item.fecha for item in request.data])

        if len(fechas_candidatas) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Se requiere fecha_inicio/fecha_fin o al menos una fecha en data[]"
            )

        fecha_inicio_str = request.fecha_inicio or min(fechas_candidatas)
        fecha_fin_str = request.fecha_fin or max(fechas_candidatas)

        fecha_inicio_dt = datetime.strptime(fecha_inicio_str, '%Y-%m-%d')
        fecha_fin_dt = datetime.strptime(fecha_fin_str, '%Y-%m-%d')

        if fecha_fin_dt < fecha_inicio_dt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="fecha_fin debe ser igual o posterior a fecha_inicio"
            )

        # Derivar parámetros para la predicción:
        # - end_date: día anterior a fecha_inicio (según solicitud)
        # - n_days: cantidad de días entre fecha_inicio y fecha_fin (incluyendo ambos)
        derived_end_date = (fecha_inicio_dt - timedelta(days=1)).strftime('%Y-%m-%d')
        derived_n_days = (fecha_fin_dt - fecha_inicio_dt).days + 1
        if derived_n_days < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Rango de fechas inválido: n_days calculado < 1"
            )

        predict_payload = PredictRequest(
            ucp=request.ucp,
            end_date=derived_end_date,
            n_days=derived_n_days,
            force_retrain=request.force_retrain,
            offset_scalar=request.offset_scalar
        )

        # Ejecutar flujo de predicción existente con los parámetros derivados
        prediction_response = await run_predict_flow(predict_payload)

        # Determinar rango para curvas base (ya validado arriba)
        fecha_inicio = fecha_inicio_str
        fecha_fin = fecha_fin_str

        logger.info(f"Calculando curva base para rango {fecha_inicio} -> {fecha_fin} (n_days={derived_n_days})")

        # Calcular curvas base reutilizando la lógica existente
        curves = await run_in_threadpool(
            calculate_base_curves,
            request.ucp,
            fecha_inicio,
            fecha_fin
        )

        logger.info("✅ Flujo combinado completado")

        # Preparar tipopronostico por fecha, filtrando solo las fechas que están en la predicción
        data_curves=BaseCurveResponse(curves=curves).curves
        hourly_predictions = prediction_response.predictions
        data_prediction = {
            item.fecha: item.model_dump()
            for item in hourly_predictions
        }
        forecast_list = request.data
        resultado = {}

        for item in forecast_list:
            fecha = item.fecha

            # MODELO IA → ya viene como dict con P1..P24
            if item.tipopronostico == "Modelo IA":
                pred = data_prediction[fecha]

                resultado[fecha] = {
                    f"P{i}": pred[f"P{i}"]
                    for i in range(1, 25)
                }

            # MODELO BASE → viene como lista [0..23]
            elif item.tipopronostico == "Modelo Base":
                curva = data_curves[fecha]

                resultado[fecha] = {
                    f"P{i}": curva[i - 1]
                    for i in range(1, 25)
                }
        print(resultado)

        return PredictWithBaseCurveResponse(
            # prediction=prediction_response,
            # base_curve=BaseCurveResponse(curves=curves),
            resultado=resultado
            
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en flujo combinado: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ejecutando flujo combinado: {str(e)}"
        )

@app.post('/get_events', response_model=EventsResponse, status_code=status.HTTP_200_OK)
async def get_events(request: EventsRequest):   
    logger.info("\n📅 PASO 6: Obteniendo eventos futuros que podrían afectar la demanda...")
    try:
        # Obtener rango de fechas de las predicciones
        fecha_inicio_pred = request.fecha_inicio
        fecha_fin_pred = request.fecha_fin

        # Llamar a OpenAI para obtener eventos
        events = await get_future_events_from_openai(
            ucp=request.ucp,
            fecha_inicio=fecha_inicio_pred,
            fecha_fin=fecha_fin_pred
        )

        logger.info(f"✓ Eventos futuros identificados: {len(events)}")
        if events:
            for fecha_evento, nombre_evento in list(events.items())[:3]:  # Log primeros 3
                logger.info(f"    {fecha_evento}: {nombre_evento}")
        return EventsResponse(events=events)
    except Exception as e:
        logger.warning(f"⚠ Error obteniendo eventos futuros: {e}")
        events = {}


async def analyze_deviation_with_openai(
    ucp: str,
    fecha: str,
    mape: float
) -> str:
    """
    Analiza las posibles causas de un desvío específico usando OpenAI con búsqueda en internet

    Args:
        ucp: Nombre del UCP (ej: 'Atlantico', 'Oriente')
        fecha: Fecha del desvío (YYYY-MM-DD)
        mape: MAPE del desvío (puede ser positivo o negativo)

    Returns:
        str: Análisis detallado de OpenAI sobre posibles causas
    """
    try:
        api_key = os.getenv('API_KEY')
        if not api_key:
            logger.warning("⚠ API_KEY no encontrada en .env, saltando análisis de OpenAI")
            return "Análisis no disponible (API_KEY no configurada)"

        client = OpenAI(api_key=api_key)

        # Determinar si fue sobreestimación o subestimación
        if mape > 0:
            tipo_desvio = "subestimación (demanda real mayor a la predicha)"
        else:
            tipo_desvio = "sobreestimación (demanda real menor a la predicha)"

        prompt = f"""Eres un analista energético experto. Necesito que investigues en internet las posibles causas de un desvío en la predicción de demanda energética.

**Contexto:**
- UCP: {ucp}, Colombia
- Fecha del desvío: {fecha}
- Magnitud del error: {abs(mape):.2f}%
- Tipo de desvío: {tipo_desvio}

**Tarea:**
Busca en internet eventos, acontecimientos o situaciones que ocurrieron en {ucp}, Colombia en la fecha {fecha} o días cercanos que pudieron causar este desvío en la demanda de energía eléctrica.

Considera:
- Eventos climáticos (tormentas, olas de calor/frío, lluvias intensas)
- Eventos públicos masivos (conciertos, partidos de fútbol, festivales)
- Días festivos locales o nacionales
- Apagones o fallas en el suministro eléctrico
- Eventos políticos o sociales (paros, manifestaciones)
- Cambios en actividad industrial o comercial
- Cualquier otro acontecimiento relevante

Proporciona un análisis conciso (máximo 2-3 oraciones) explicando las causas más probables encontradas."""

        logger.info(f"🤖 Consultando OpenAI para análisis de desvío en {fecha}...")

        response = await run_in_threadpool(
            lambda: client.responses.create(
                model="gpt-5-mini",
                input=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                tools=[
                    {
                        "type": "web_search"
                    }
                ]
            )
        )

        analysis = response.output_text.strip()
        logger.info(f"✓ Análisis recibido para {fecha}: {len(analysis)} caracteres")

        return analysis

    except Exception as e:
        logger.error(f"Error en análisis de OpenAI para {fecha}: {e}")
        return f"Análisis no disponible (error: {str(e)})"


@app.post('/analyze-deviation', response_model=AnalyzeDeviationResponse, status_code=status.HTTP_200_OK)
async def analyze_deviation(request: AnalyzeDeviationRequest):
    """
    Analiza desvíos de predicción y busca eventos que pudieron causarlos.

    Recibe una lista de desvíos (fecha, MAPE) y consulta OpenAI para identificar
    eventos o situaciones que pudieron causar cada desvío.
    """
    logger.info("="*80)
    logger.info(f"🔍 ANALIZANDO DESVÍOS PARA {request.ucp}")
    logger.info(f"   Total de desvíos a analizar: {len(request.desvios)}")
    logger.info("="*80)

    try:
        resultados = []

        for i, desvio in enumerate(request.desvios, 1):
            logger.info(f"\n📊 Analizando desvío {i}/{len(request.desvios)}: {desvio.fecha} (MAPE: {desvio.mape:+.2f}%)")

            analisis = await analyze_deviation_with_openai(
                ucp=request.ucp,
                fecha=desvio.fecha,
                mape=desvio.mape
            )

            resultados.append(DeviationAnalysisItem(
                fecha=desvio.fecha,
                mape=desvio.mape,
                analisis=analisis
            ))

        # Generar resumen general si hay múltiples desvíos
        if len(request.desvios) > 1:
            fechas_str = ", ".join([d.fecha for d in request.desvios])
            mapes_str = ", ".join([f"{d.mape:+.2f}%" for d in request.desvios])

            resumen_prompt = f"""Basándote en los siguientes desvíos de predicción de demanda energética en {request.ucp}, Colombia, proporciona un breve resumen (2-3 oraciones) identificando patrones comunes o tendencias:

Fechas: {fechas_str}
MAPEs: {mapes_str}

Análisis individuales:
{chr(10).join([f"- {r.fecha}: {r.analisis}" for r in resultados])}

Resume los patrones o causas comunes identificadas."""

            try:
                api_key = os.getenv('API_KEY')
                if api_key:
                    client = OpenAI(api_key=api_key)
                    response = await run_in_threadpool(
                        lambda: client.responses.create(
                            model="gpt-5-mini",
                            input=[{"role": "user", "content": resumen_prompt}]
                        )
                    )
                    resumen_general = response.output_text.strip()
                else:
                    resumen_general = "Resumen no disponible (API_KEY no configurada)"
            except Exception as e:
                resumen_general = f"Resumen no disponible (error: {str(e)})"
        else:
            resumen_general = resultados[0].analisis if resultados else "Sin análisis disponible"

        logger.info(f"\n✓ Análisis completado para {len(resultados)} desvíos")

        return AnalyzeDeviationResponse(
            ucp=request.ucp,
            total_desvios=len(resultados),
            resultados=resultados,
            resumen_general=resumen_general
        )

    except Exception as e:
        logger.error(f"Error analizando desvíos: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analizando desvíos: {str(e)}"
        )


def calculate_base_curves(
    ucp: str,
    fecha_inicio: str,
    fecha_fin: str
) -> Dict[str, List[float]]:
    """
    Calcula curvas base para un rango de fechas.
    
    Para días festivos: usa promedio histórico del mismo día en años anteriores.
    Para días normales: usa clusters + total promedio histórico.
    
    Args:
        ucp: UCP para filtrar datos
        fecha_inicio: Fecha inicio (YYYY-MM-DD)
        fecha_fin: Fecha fin (YYYY-MM-DD)
    
    Returns:
        Dict con formato {'YYYY-MM-DD': [P1, P2, ..., P24]}
    """
    from src.prediction.hourly import HourlyDisaggregationEngine, CalendarClassifier
    from src.config.settings import FEATURES_DATA_DIR, RAW_DATA_DIR
    import numpy as np
    
    logger.info(f"📊 Calculando curvas base para UCP={ucp}, {fecha_inicio} a {fecha_fin}")
    
    # Cargar datos históricos filtrados por UCP
    # Intentar primero desde data/raw/{ucp}/datos.csv
    historical_paths = [
        Path(RAW_DATA_DIR) / ucp / "datos.csv",
        Path(FEATURES_DATA_DIR) / "data_with_features_latest.csv"
    ]
    
    df_historico = None
    for path in historical_paths:
        if path.exists():
            try:
                df_historico = pd.read_csv(path)
                # Normalizar nombres de columnas
                if 'FECHA' in df_historico.columns:
                    df_historico.rename(columns={'FECHA': 'fecha'}, inplace=True)
                df_historico['fecha'] = pd.to_datetime(df_historico['fecha'])
                
                # Filtrar por UCP si existe la columna
                if 'UCP' in df_historico.columns:
                    df_historico = df_historico[df_historico['UCP'] == ucp].copy()
                    logger.info(f"✓ Datos cargados desde {path} (filtrado por UCP={ucp}): {len(df_historico)} registros")
                else:
                    logger.info(f"✓ Datos cargados desde {path}: {len(df_historico)} registros (sin columna UCP)")
                
                if len(df_historico) > 0:
                    break
            except Exception as e:
                logger.warning(f"⚠ Error cargando {path}: {e}")
                continue
    
    if df_historico is None or len(df_historico) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No se encontraron datos históricos para UCP={ucp}"
        )
    
    # Validar que tenga columnas P1-P24
    period_cols = [f'P{i}' for i in range(1, 25)]
    missing_cols = [col for col in period_cols if col not in df_historico.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Faltan columnas de períodos horarios: {missing_cols}"
        )
    
    # Inicializar componentes
    calendar_classifier = CalendarClassifier(ucp=ucp)

    # Cargar motor de desagregación horaria (para días normales)
    try:
        models_dir = Path(f'models/{ucp}') if Path(f'models/{ucp}').exists() else Path('models')
        hourly_engine = HourlyDisaggregationEngine(
            auto_load=True,
            models_dir=str(models_dir),
            ucp=ucp
        )
        logger.info(f"✓ Motor de desagregación horaria cargado")
    except Exception as e:
        logger.warning(f"⚠ No se pudo cargar motor de desagregación: {e}")
        hourly_engine = None
    
    # Generar rango de fechas
    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    fecha_fin_dt = pd.to_datetime(fecha_fin)
    fecha_range = pd.date_range(start=fecha_inicio_dt, end=fecha_fin_dt, freq='D')
    
    curves = {}
    
    # Función helper para detectar temporada navideña (23 dic - 6 ene)
    def is_christmas_season(date):
        """Verifica si una fecha está en temporada navideña (23 dic - 6 ene)"""
        month = date.month
        day = date.day
        # Del 23 al 31 de diciembre
        if month == 12 and day >= 23:
            return True
        # Del 1 al 6 de enero
        if month == 1 and day <= 6:
            return True
        return False
    
    for fecha in fecha_range:
        fecha_str = fecha.strftime('%Y-%m-%d')
        mmdd = fecha.strftime('%m-%d')  # Para buscar mismo día en años anteriores
        
        # Clasificar día
        is_holiday = calendar_classifier.is_holiday(fecha)
        is_weekend = calendar_classifier.is_weekend(fecha)
        is_christmas = is_christmas_season(fecha)
        
        # MÉTODO 0: Si está en temporada navideña, usar estrategia híbrida que considera día de la semana
        # ESTRATEGIA MEJORADA: Usar cluster normal según día de la semana, pero escalado con total histórico del mismo día navideño
        # Esto captura tanto el patrón del día de la semana como el nivel de demanda navideña
        if is_christmas:
            # Identificar días muy especiales (Navidad y Año Nuevo)
            is_very_special = (fecha.month == 12 and fecha.day == 25) or (fecha.month == 1 and fecha.day == 1)
            
            # Buscar todos los años anteriores con mismo mes-día (últimos 3 años)
            historical_same_day = df_historico[
                (df_historico['fecha'].dt.month == fecha.month) &
                (df_historico['fecha'].dt.day == fecha.day) &
                (df_historico['fecha'] < fecha)  # Solo años anteriores
            ].copy()
            
            if len(historical_same_day) > 0:
                # Ordenar por fecha descendente y usar últimos 3 años
                historical_same_day = historical_same_day.sort_values('fecha', ascending=False)
                if len(historical_same_day) > 3:
                    historical_same_day = historical_same_day.head(3)
                
                # Calcular total promedio histórico del mismo día navideño
                avg_total_historico = historical_same_day['TOTAL'].mean() if 'TOTAL' in historical_same_day.columns else historical_same_day[period_cols].sum(axis=1).mean()
                
                if is_very_special:
                    # Para Navidad y Año Nuevo: usar perfil histórico normalizado del mismo día
                    # IMPORTANTE: Usar solo los últimos 2 años para evitar incluir años muy antiguos
                    if len(historical_same_day) > 1:
                        historical_recent = historical_same_day.head(2)
                        totals_historical = historical_recent[period_cols].sum(axis=1)
                        normalized_profiles = historical_recent[period_cols].div(totals_historical, axis=0)
                        avg_normalized_profile = normalized_profiles.mean().values
                        avg_total_historico_reciente = historical_recent['TOTAL'].mean() if 'TOTAL' in historical_recent.columns else historical_recent[period_cols].sum(axis=1).mean()
                        hourly_profile = avg_normalized_profile * avg_total_historico_reciente
                        logger.debug(f"  {fecha_str} (día muy especial): perfil histórico normalizado × total reciente ({avg_total_historico_reciente:.2f})")
                    else:
                        totals_historical = historical_same_day[period_cols].sum(axis=1)
                        normalized_profiles = historical_same_day[period_cols].div(totals_historical, axis=0)
                        avg_normalized_profile = normalized_profiles.mean().values
                        hourly_profile = avg_normalized_profile * avg_total_historico
                        logger.debug(f"  {fecha_str} (día muy especial): perfil histórico normalizado × total histórico ({avg_total_historico:.2f})")
                else:
                    # Para otros días navideños: usar cluster normal según día de la semana, pero escalado con total ajustado
                    # ESTRATEGIA MEJORADA: Aplicar ajustes más agresivos para días hábiles durante toda la temporada navideña
                    is_late_christmas = (fecha.month == 12 and fecha.day >= 28) or (fecha.month == 1 and fecha.day <= 6)
                    is_weekday = fecha.dayofweek < 5  # Lunes a Viernes
                    
                    if hourly_engine and hourly_engine.normal_disaggregator.is_fitted:
                        try:
                            # Obtener perfil normalizado del cluster según día de la semana
                            result = hourly_engine.normal_disaggregator.predict_hourly_profile(
                                fecha, 1.0, return_normalized=True  # Usar 1.0 para obtener solo el perfil normalizado
                            )
                            if isinstance(result, tuple):
                                _, normalized_profile, _ = result
                            else:
                                # Si no retorna tupla, calcular perfil desde cluster
                                dayofweek = fecha.dayofweek
                                cluster_id = hourly_engine.normal_disaggregator.cluster_by_dayofweek.get(dayofweek, 0)
                                normalized_profile = hourly_engine.normal_disaggregator.cluster_profiles.loc[cluster_id].values
                            
                            # Calcular total a usar según el día - ESTRATEGIA MEJORADA
                            fecha_corte_reciente = fecha - pd.DateOffset(days=60)
                            df_recientes_habiles = df_historico[
                                (df_historico['fecha'] >= fecha_corte_reciente) &
                                (df_historico['fecha'] < fecha) &
                                (df_historico['fecha'].dt.dayofweek == fecha.dayofweek) &  # Mismo día de la semana
                                (~df_historico['fecha'].apply(lambda x: calendar_classifier.is_holiday(pd.to_datetime(x))))
                            ].copy()
                            
                            if is_weekday:
                                # Días hábiles: usar enfoque híbrido más agresivo
                                if len(df_recientes_habiles) > 0:
                                    avg_total_reciente_habiles = df_recientes_habiles['TOTAL'].mean() if 'TOTAL' in df_recientes_habiles.columns else df_recientes_habiles[period_cols].sum(axis=1).mean()
                                    
                                    if is_late_christmas:
                                        # Días hábiles del 28 en adelante: 60% histórico navideño + 40% reciente hábiles
                                        total_final = avg_total_historico * 0.6 + avg_total_reciente_habiles * 0.4
                                        logger.debug(f"  {fecha_str} (temporada navideña tardía hábil): cluster × total mixto 60/40 (hist={avg_total_historico:.2f}, reciente={avg_total_reciente_habiles:.2f}, final={total_final:.2f})")
                                    else:
                                        # Días hábiles tempranos (23-27 dic): 50% histórico navideño + 50% reciente hábiles
                                        # Esto captura mejor que días hábiles tienen demanda más alta incluso en temporada navideña
                                        total_final = avg_total_historico * 0.5 + avg_total_reciente_habiles * 0.5
                                        logger.debug(f"  {fecha_str} (temporada navideña temprana hábil): cluster × total mixto 50/50 (hist={avg_total_historico:.2f}, reciente={avg_total_reciente_habiles:.2f}, final={total_final:.2f})")
                                else:
                                    # Sin datos recientes: usar ajustes más agresivos
                                    if is_late_christmas:
                                        total_final = avg_total_historico * 1.10  # +10% para días tardíos hábiles
                                    else:
                                        total_final = avg_total_historico * 1.08  # +8% para días tempranos hábiles
                                    logger.debug(f"  {fecha_str} (temporada navideña hábil): cluster × total histórico ajustado ({total_final:.2f})")
                            else:
                                # Días no hábiles (sábado/domingo)
                                if is_late_christmas:
                                    total_final = avg_total_historico * 1.05  # +5% para días tardíos no hábiles
                                else:
                                    total_final = avg_total_historico * 1.03  # +3% para días tempranos no hábiles
                                logger.debug(f"  {fecha_str} (temporada navideña no hábil): cluster × total histórico ajustado ({total_final:.2f})")
                            
                            hourly_profile = normalized_profile * total_final
                        except Exception as e:
                            logger.warning(f"  {fecha_str} (temporada navideña): error con cluster ({e}), usando perfil histórico")
                            # Fallback: usar perfil histórico normalizado con ajustes mejorados
                            totals_historical = historical_same_day[period_cols].sum(axis=1)
                            normalized_profiles = historical_same_day[period_cols].div(totals_historical, axis=0)
                            avg_normalized_profile = normalized_profiles.mean().values
                            
                            # Aplicar ajustes mejorados
                            fecha_corte_reciente = fecha - pd.DateOffset(days=60)
                            df_recientes_habiles = df_historico[
                                (df_historico['fecha'] >= fecha_corte_reciente) &
                                (df_historico['fecha'] < fecha) &
                                (df_historico['fecha'].dt.dayofweek == fecha.dayofweek) &
                                (~df_historico['fecha'].apply(lambda x: calendar_classifier.is_holiday(pd.to_datetime(x))))
                            ].copy()
                            
                            if is_weekday and len(df_recientes_habiles) > 0:
                                avg_total_reciente_habiles = df_recientes_habiles['TOTAL'].mean() if 'TOTAL' in df_recientes_habiles.columns else df_recientes_habiles[period_cols].sum(axis=1).mean()
                                if is_late_christmas:
                                    total_final = avg_total_historico * 0.6 + avg_total_reciente_habiles * 0.4
                                else:
                                    total_final = avg_total_historico * 0.5 + avg_total_reciente_habiles * 0.5
                            elif is_weekday:
                                total_final = avg_total_historico * 1.10 if is_late_christmas else avg_total_historico * 1.08
                            else:
                                total_final = avg_total_historico * 1.05 if is_late_christmas else avg_total_historico * 1.03
                            
                            hourly_profile = avg_normalized_profile * total_final
                    else:
                        # Sin cluster disponible: usar perfil histórico normalizado con ajustes mejorados
                        totals_historical = historical_same_day[period_cols].sum(axis=1)
                        normalized_profiles = historical_same_day[period_cols].div(totals_historical, axis=0)
                        avg_normalized_profile = normalized_profiles.mean().values
                        
                        # Aplicar ajustes mejorados
                        fecha_corte_reciente = fecha - pd.DateOffset(days=60)
                        df_recientes_habiles = df_historico[
                            (df_historico['fecha'] >= fecha_corte_reciente) &
                            (df_historico['fecha'] < fecha) &
                            (df_historico['fecha'].dt.dayofweek == fecha.dayofweek) &
                            (~df_historico['fecha'].apply(lambda x: calendar_classifier.is_holiday(pd.to_datetime(x))))
                        ].copy()
                        
                        if is_weekday and len(df_recientes_habiles) > 0:
                            avg_total_reciente_habiles = df_recientes_habiles['TOTAL'].mean() if 'TOTAL' in df_recientes_habiles.columns else df_recientes_habiles[period_cols].sum(axis=1).mean()
                            if is_late_christmas:
                                total_final = avg_total_historico * 0.6 + avg_total_reciente_habiles * 0.4
                            else:
                                total_final = avg_total_historico * 0.5 + avg_total_reciente_habiles * 0.5
                        elif is_weekday:
                            total_final = avg_total_historico * 1.10 if is_late_christmas else avg_total_historico * 1.08
                        else:
                            total_final = avg_total_historico * 1.05 if is_late_christmas else avg_total_historico * 1.03
                        
                        hourly_profile = avg_normalized_profile * total_final
                        logger.debug(f"  {fecha_str} (temporada navideña): perfil histórico normalizado × total ({total_final:.2f})")
            else:
                # Fallback: usar promedio de todos los días de temporada navideña del mismo día en años anteriores (últimos 3 años)
                historical_christmas = df_historico[
                    (df_historico['fecha'].dt.month == fecha.month) &
                    (df_historico['fecha'].dt.day == fecha.day) &
                    (df_historico['fecha'] < fecha) &
                    (df_historico['fecha'] >= (fecha - pd.DateOffset(years=3)))
                ].copy()
                
                if len(historical_christmas) > 0:
                    # Normalizar y promediar
                    totals = historical_christmas[period_cols].sum(axis=1)
                    normalized = historical_christmas[period_cols].div(totals, axis=0)
                    avg_normalized = normalized.mean().values
                    
                    # Identificar si es día muy especial
                    is_very_special = (fecha.month == 12 and fecha.day == 25) or (fecha.month == 1 and fecha.day == 1)
                    avg_total_historico = historical_christmas['TOTAL'].mean() if 'TOTAL' in historical_christmas.columns else historical_christmas[period_cols].sum(axis=1).mean()
                    
                    if is_very_special:
                        # Para Navidad y Año Nuevo: usar perfil histórico normalizado
                        hourly_profile = avg_normalized * avg_total_historico
                        logger.debug(f"  {fecha_str} (temporada navideña, día especial): fallback con perfil histórico")
                    else:
                        # Para otros días: usar cluster normal según día de la semana con ajuste para días tardíos
                        is_late_christmas = (fecha.month == 12 and fecha.day >= 28) or (fecha.month == 1 and fecha.day <= 6)
                        is_weekday = fecha.dayofweek < 5  # Lunes a Viernes
                        
                        if hourly_engine and hourly_engine.normal_disaggregator.is_fitted:
                            try:
                                result = hourly_engine.normal_disaggregator.predict_hourly_profile(
                                    fecha, 1.0, return_normalized=True
                                )
                                if isinstance(result, tuple):
                                    _, normalized_profile, _ = result
                                else:
                                    dayofweek = fecha.dayofweek
                                    cluster_id = hourly_engine.normal_disaggregator.cluster_by_dayofweek.get(dayofweek, 0)
                                    normalized_profile = hourly_engine.normal_disaggregator.cluster_profiles.loc[cluster_id].values
                                
                                # Aplicar ajuste para días tardíos y hábiles
                                if is_late_christmas and is_weekday:
                                    fecha_corte_reciente = fecha - pd.DateOffset(days=60)
                                    df_recientes_habiles = df_historico[
                                        (df_historico['fecha'] >= fecha_corte_reciente) &
                                        (df_historico['fecha'] < fecha) &
                                        (df_historico['fecha'].dt.dayofweek == fecha.dayofweek) &
                                        (~df_historico['fecha'].apply(lambda x: calendar_classifier.is_holiday(pd.to_datetime(x))))
                                    ].copy()
                                    
                                    if len(df_recientes_habiles) > 0:
                                        avg_total_reciente_habiles = df_recientes_habiles['TOTAL'].mean() if 'TOTAL' in df_recientes_habiles.columns else df_recientes_habiles[period_cols].sum(axis=1).mean()
                                        total_final = avg_total_historico * 0.7 + avg_total_reciente_habiles * 0.3
                                    else:
                                        total_final = avg_total_historico * 1.05
                                elif is_late_christmas:
                                    total_final = avg_total_historico * 1.03
                                else:
                                    total_final = avg_total_historico
                                
                                hourly_profile = normalized_profile * total_final
                                logger.debug(f"  {fecha_str} (temporada navideña): fallback con cluster día semana (total={total_final:.2f})")
                            except Exception as e:
                                logger.warning(f"  {fecha_str} (temporada navideña): error con cluster en fallback ({e})")
                                # Aplicar ajuste también en fallback
                                if is_late_christmas and is_weekday:
                                    total_final = avg_total_historico * 1.05
                                elif is_late_christmas:
                                    total_final = avg_total_historico * 1.03
                                else:
                                    total_final = avg_total_historico
                                hourly_profile = avg_normalized * total_final
                        else:
                            # Aplicar ajuste también sin cluster
                            if is_late_christmas and is_weekday:
                                total_final = avg_total_historico * 1.05
                            elif is_late_christmas:
                                total_final = avg_total_historico * 1.03
                            else:
                                total_final = avg_total_historico
                            hourly_profile = avg_normalized * total_final
                            logger.debug(f"  {fecha_str} (temporada navideña): fallback con perfil histórico (total={total_final:.2f})")
                else:
                    # Último fallback: usar cluster de días especiales
                    if hourly_engine and hourly_engine.special_disaggregator.is_fitted:
                        # Intentar usar total reciente para escalar
                        fecha_corte_reciente = fecha - pd.DateOffset(days=60)
                        df_recientes = df_historico[
                            (df_historico['fecha'] >= fecha_corte_reciente) &
                            (df_historico['fecha'] < fecha) &
                            (~df_historico['fecha'].apply(lambda x: calendar_classifier.is_holiday(pd.to_datetime(x))))
                        ].copy()
                        
                        if len(df_recientes) > 0:
                            avg_total = df_recientes['TOTAL'].mean() if 'TOTAL' in df_recientes.columns else df_recientes[period_cols].sum(axis=1).mean()
                        else:
                            avg_total = df_historico['TOTAL'].mean() if 'TOTAL' in df_historico.columns else df_historico[period_cols].sum(axis=1).mean()
                        
                        result = hourly_engine.special_disaggregator.predict_hourly_profile(fecha, avg_total, return_normalized=False)
                        if result is not None:
                            if isinstance(result, tuple):
                                hourly_profile = result[0]
                            else:
                                hourly_profile = result
                            if not isinstance(hourly_profile, np.ndarray):
                                hourly_profile = np.array(hourly_profile)
                        else:
                            hourly_profile = np.zeros(24)
                        logger.debug(f"  {fecha_str} (temporada navideña): usando cluster especial")
                    else:
                        # Fallback final: perfil plano con total reciente
                        fecha_corte_reciente = fecha - pd.DateOffset(days=60)
                        df_recientes = df_historico[
                            (df_historico['fecha'] >= fecha_corte_reciente) &
                            (df_historico['fecha'] < fecha) &
                            (~df_historico['fecha'].apply(lambda x: calendar_classifier.is_holiday(pd.to_datetime(x))))
                        ].copy()
                        
                        if len(df_recientes) > 0:
                            avg_total = df_recientes['TOTAL'].mean() if 'TOTAL' in df_recientes.columns else df_recientes[period_cols].sum(axis=1).mean()
                        else:
                            avg_total = df_historico['TOTAL'].mean() if 'TOTAL' in df_historico.columns else df_historico[period_cols].sum(axis=1).mean()
                        
                        hourly_profile = np.full(24, avg_total / 24)
                        logger.warning(f"  {fecha_str} (temporada navideña): usando perfil plano (sin datos)")
        
        # MÉTODO 1: Si es festivo (pero no temporada navideña), usar promedio histórico del mismo día
        elif is_holiday:
            # Buscar todos los años anteriores con mismo mes-día
            historical_same_day = df_historico[
                (df_historico['fecha'].dt.month == fecha.month) &
                (df_historico['fecha'].dt.day == fecha.day) &
                (df_historico['fecha'] < fecha)  # Solo años anteriores
            ].copy()
            
            if len(historical_same_day) > 0:
                # Ordenar por fecha descendente y usar últimos años (máximo 5 años para festivos)
                historical_same_day = historical_same_day.sort_values('fecha', ascending=False)
                if len(historical_same_day) > 5:
                    historical_same_day = historical_same_day.head(5)
                    logger.debug(f"  {fecha_str} (festivo): usando últimos 5 años de {len(historical_same_day)} disponibles")
                
                # Promediar perfiles horarios
                hourly_profile = historical_same_day[period_cols].mean().values
                logger.debug(f"  {fecha_str} (festivo): {len(historical_same_day)} años históricos")
            else:
                # Fallback: usar promedio de todos los festivos del mismo mes
                historical_holidays = df_historico[
                    (df_historico['fecha'].dt.month == fecha.month) &
                    (df_historico['fecha'].apply(lambda x: calendar_classifier.is_holiday(pd.to_datetime(x))))
                ].copy()
                
                if len(historical_holidays) > 0:
                    # Normalizar y promediar
                    totals = historical_holidays[period_cols].sum(axis=1)
                    normalized = historical_holidays[period_cols].div(totals, axis=0)
                    avg_normalized = normalized.mean().values
                    
                    # Escalar con total promedio
                    avg_total = historical_holidays['TOTAL'].mean() if 'TOTAL' in historical_holidays.columns else historical_holidays[period_cols].sum(axis=1).mean()
                    hourly_profile = avg_normalized * avg_total
                    logger.debug(f"  {fecha_str} (festivo): fallback con {len(historical_holidays)} festivos del mes")
                else:
                    # Último fallback: usar cluster de días especiales
                    if hourly_engine and hourly_engine.special_disaggregator.is_fitted:
                        avg_total = df_historico['TOTAL'].mean() if 'TOTAL' in df_historico.columns else df_historico[period_cols].sum(axis=1).mean()
                        result = hourly_engine.special_disaggregator.predict_hourly_profile(fecha, avg_total, return_normalized=False)
                        if result is not None:
                            if isinstance(result, tuple):
                                hourly_profile = result[0]
                            else:
                                hourly_profile = result
                            if not isinstance(hourly_profile, np.ndarray):
                                hourly_profile = np.array(hourly_profile)
                        else:
                            hourly_profile = np.zeros(24)
                        logger.debug(f"  {fecha_str} (festivo): usando cluster especial")
                    else:
                        # Fallback final: perfil plano
                        avg_total = df_historico['TOTAL'].mean() if 'TOTAL' in df_historico.columns else df_historico[period_cols].sum(axis=1).mean()
                        hourly_profile = np.full(24, avg_total / 24)
                        logger.warning(f"  {fecha_str} (festivo): usando perfil plano (sin datos)")
        
        # MÉTODO 2: Si es día normal, usar cluster + total promedio histórico
        # ESTRATEGIA: Usar días del mismo día de la semana del mes más reciente disponible
        # Esto captura mejor el comportamiento actual de la demanda
        else:
            # Obtener el día de la semana (0=lunes, 6=domingo)
            target_dow = fecha.dayofweek
            
            # Buscar días del mismo día de la semana en el mes más reciente disponible
            # IMPORTANTE: Excluir días festivos para no sesgar el promedio
            # Intentar primero el mes anterior completo
            month_ago = fecha - pd.DateOffset(months=1)
            historical_recent = df_historico[
                (df_historico['fecha'].dt.year == month_ago.year) &
                (df_historico['fecha'].dt.month == month_ago.month) &
                (df_historico['fecha'].dt.dayofweek == target_dow) &
                (df_historico['fecha'] < fecha)
            ].copy()
            
            # Filtrar días festivos
            if len(historical_recent) > 0:
                historical_recent = historical_recent[
                    ~historical_recent['fecha'].apply(lambda x: calendar_classifier.is_holiday(pd.to_datetime(x)))
                ].copy()
            
            if len(historical_recent) == 0:
                # Si no hay datos del mes anterior, buscar en el mes actual (si hay datos anteriores a la fecha)
                historical_recent = df_historico[
                    (df_historico['fecha'].dt.year == fecha.year) &
                    (df_historico['fecha'].dt.month == fecha.month) &
                    (df_historico['fecha'].dt.dayofweek == target_dow) &
                    (df_historico['fecha'] < fecha)
                ].copy()
                
                # Filtrar días festivos
                if len(historical_recent) > 0:
                    historical_recent = historical_recent[
                        ~historical_recent['fecha'].apply(lambda x: calendar_classifier.is_holiday(pd.to_datetime(x)))
                    ].copy()
            
            if len(historical_recent) == 0:
                # Si aún no hay datos, buscar en los últimos 2 meses (cualquier mes)
                two_months_ago = fecha - pd.DateOffset(months=2)
                historical_recent = df_historico[
                    (df_historico['fecha'] >= two_months_ago) &
                    (df_historico['fecha'] < fecha) &
                    (df_historico['fecha'].dt.dayofweek == target_dow)
                ].copy()
                
                # Filtrar días festivos
                if len(historical_recent) > 0:
                    historical_recent = historical_recent[
                        ~historical_recent['fecha'].apply(lambda x: calendar_classifier.is_holiday(pd.to_datetime(x)))
                    ].copy()
            
            # Variable para almacenar datos históricos a usar para perfil horario
            historical_for_profile = None
            
            if len(historical_recent) > 0:
                # Ordenar por fecha descendente (más recientes primero)
                historical_recent = historical_recent.sort_values('fecha', ascending=False)
                historical_for_profile = historical_recent
                
                # Calcular promedio de los días encontrados
                avg_total = historical_recent['TOTAL'].mean() if 'TOTAL' in historical_recent.columns else historical_recent[period_cols].sum(axis=1).mean()
                logger.debug(f"  {fecha_str} (normal): usando {len(historical_recent)} días recientes del mismo día de semana, total={avg_total:.2f}")
            else:
                # Fallback: usar mismo día de la semana en el mismo mes de años anteriores (últimos 2 años)
                historical_same_dow = df_historico[
                    (df_historico['fecha'].dt.month == fecha.month) &
                    (df_historico['fecha'].dt.dayofweek == target_dow) &
                    (df_historico['fecha'] < fecha) &
                    (df_historico['fecha'] >= (fecha - pd.DateOffset(years=2)))
                ].copy()
                
                # Filtrar días festivos
                if len(historical_same_dow) > 0:
                    historical_same_dow = historical_same_dow[
                        ~historical_same_dow['fecha'].apply(lambda x: calendar_classifier.is_holiday(pd.to_datetime(x)))
                    ].copy()
                
                if len(historical_same_dow) > 0:
                    historical_for_profile = historical_same_dow
                    avg_total = historical_same_dow['TOTAL'].mean() if 'TOTAL' in historical_same_dow.columns else historical_same_dow[period_cols].sum(axis=1).mean()
                    logger.debug(f"  {fecha_str} (normal): fallback con {len(historical_same_dow)} días históricos del mismo mes (festivos excluidos)")
                else:
                    # Último fallback: promedio general de últimos 2 años (excluyendo festivos)
                    df_recent = df_historico[df_historico['fecha'] >= (fecha - pd.DateOffset(years=2))].copy()
                    if len(df_recent) > 0:
                        # Filtrar días festivos
                        df_recent = df_recent[
                            ~df_recent['fecha'].apply(lambda x: calendar_classifier.is_holiday(pd.to_datetime(x)))
                        ].copy()
                        if len(df_recent) > 0:
                            historical_for_profile = df_recent
                            avg_total = df_recent['TOTAL'].mean() if 'TOTAL' in df_recent.columns else df_recent[period_cols].sum(axis=1).mean()
                        else:
                            # Si después de filtrar festivos no quedan datos, usar todos
                            df_recent = df_historico[df_historico['fecha'] >= (fecha - pd.DateOffset(years=2))].copy()
                            historical_for_profile = df_recent
                            avg_total = df_recent['TOTAL'].mean() if 'TOTAL' in df_recent.columns else df_recent[period_cols].sum(axis=1).mean()
                    else:
                        historical_for_profile = None
                        avg_total = df_historico['TOTAL'].mean() if 'TOTAL' in df_historico.columns else df_historico[period_cols].sum(axis=1).mean()
                    logger.warning(f"  {fecha_str} (normal): usando promedio general como último recurso")
            
            # Usar cluster para obtener perfil normalizado
            if hourly_engine and hourly_engine.normal_disaggregator.is_fitted:
                try:
                    result = hourly_engine.normal_disaggregator.predict_hourly_profile(
                        fecha, avg_total, return_normalized=True
                    )
                    # Cuando return_normalized=True, retorna tupla (hourly, normalized, cluster_id)
                    if isinstance(result, tuple):
                        hourly_profile = result[0]
                    else:
                        hourly_profile = result
                    # Asegurar que es array numpy
                    if not isinstance(hourly_profile, np.ndarray):
                        hourly_profile = np.array(hourly_profile)
                    logger.debug(f"  {fecha_str} (normal): usando cluster, total={avg_total:.2f}")
                except Exception as e:
                    logger.warning(f"  {fecha_str} (normal): error con cluster ({e}), usando promedio reciente")
                    # Fallback: usar perfil horario promedio de datos históricos disponibles
                    if historical_for_profile is not None and len(historical_for_profile) > 0:
                        hourly_profile = historical_for_profile[period_cols].mean().values
                        logger.debug(f"  {fecha_str} (normal): usando perfil promedio de {len(historical_for_profile)} días históricos")
                    else:
                        # Perfil plano como último recurso
                        hourly_profile = np.full(24, avg_total / 24)
                        logger.warning(f"  {fecha_str} (normal): usando perfil plano (sin datos históricos)")
            else:
                # Sin cluster: usar perfil horario promedio de datos históricos disponibles
                if historical_for_profile is not None and len(historical_for_profile) > 0:
                    hourly_profile = historical_for_profile[period_cols].mean().values
                    logger.debug(f"  {fecha_str} (normal): usando perfil promedio de {len(historical_for_profile)} días históricos")
                else:
                    # Perfil plano
                    hourly_profile = np.full(24, avg_total / 24)
                    logger.warning(f"  {fecha_str} (normal): usando perfil plano (sin datos históricos)")
        
        # Convertir a lista de floats y asegurar 24 valores
        hourly_list = [float(x) for x in hourly_profile[:24]]
        if len(hourly_list) < 24:
            hourly_list.extend([0.0] * (24 - len(hourly_list)))
        
        curves[fecha_str] = hourly_list
    
    logger.info(f"✓ Curvas base calculadas para {len(curves)} días")
    return curves


@app.post("/base-curve", response_model=BaseCurveResponse, status_code=status.HTTP_200_OK)
async def get_base_curve(request: BaseCurveRequest):
    """
    Endpoint para obtener curvas base de demanda horaria para un rango de fechas.
    
    Retorna perfiles horarios con valores absolutos (MW) en formato JSON.
    """
    try:
        logger.info(f"📊 Solicitud de curva base: UCP={request.ucp}, {request.fecha_inicio} a {request.fecha_fin}")
        
        # Validar rango de fechas
        fecha_inicio_dt = pd.to_datetime(request.fecha_inicio)
        fecha_fin_dt = pd.to_datetime(request.fecha_fin)
        
        if fecha_inicio_dt > fecha_fin_dt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="fecha_inicio debe ser anterior o igual a fecha_fin"
            )
        
        # Calcular curvas base
        curves = await run_in_threadpool(
            calculate_base_curves,
            request.ucp,
            request.fecha_inicio,
            request.fecha_fin
        )
        
        logger.info(f"✅ Curvas base generadas exitosamente para {len(curves)} días")
        
        return BaseCurveResponse(curves=curves)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generando curvas base: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generando curvas base: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check(ucp: Optional[str] = None):
    """
    Health check del sistema

    Verifica estado de:
    - Modelo de predicción
    - Sistema de desagregación horaria
    - Datos históricos

    Args:
        ucp: (Opcional) Nombre del UCP para verificar estado específico

    Returns:
        HealthResponse con estado de componentes
    """
    components = {}

    if ucp:
        # Verificar modelo específico del UCP
        model_exists, model_path = check_model_exists(ucp)
        components['prediction_model'] = {
            'status': 'healthy' if model_exists else 'missing',
            'ucp': ucp,
            'path': str(model_path) if model_path else None,
            'last_modified': model_path.stat().st_mtime if model_path and model_path.exists() else None
        }

        # Verificar desagregación horaria del UCP
        hourly_trained = check_hourly_disaggregation_trained(ucp)
        components['hourly_disaggregation'] = {
            'status': 'healthy' if hourly_trained else 'missing',
            'ucp': ucp,
            'models': {
                'normal': Path(f'models/{ucp}/hourly_disaggregator.pkl').exists(),
                'special': Path(f'models/{ucp}/special_days_disaggregator.pkl').exists()
            }
        }

        # Verificar datos históricos del UCP
        features_path = Path(f'data/features/{ucp}/data_with_features_latest.csv')
        components['historical_data'] = {
            'status': 'healthy' if features_path.exists() else 'missing',
            'ucp': ucp,
            'path': str(features_path) if features_path.exists() else None,
            'last_modified': features_path.stat().st_mtime if features_path.exists() else None
        }
    else:
        # Verificar sistema general (retrocompatibilidad)
        # Buscar todos los UCPs disponibles
        models_base = Path('models')
        ucps_disponibles = [d.name for d in models_base.iterdir() if d.is_dir() and (d / 'registry').exists()]

        components['system'] = {
            'status': 'healthy',
            'ucps_disponibles': ucps_disponibles,
            'total_ucps': len(ucps_disponibles)
        }

    # Estado general
    all_healthy = all(comp.get('status') == 'healthy' for comp in components.values())

    return HealthResponse(
        status='healthy' if all_healthy else 'degraded',
        timestamp=datetime.now().isoformat(),
        version='1.0.0',
        components=components
    )


@app.post("/retrain", status_code=status.HTTP_200_OK)
async def retrain_model(ucp: str):
    """
    Trigger manual de reentrenamiento completo del modelo

    Args:
        ucp: Nombre del UCP a reentrenar (ej: 'Atlantico', 'Oriente')

    Returns:
        Dict con información del reentrenamiento

    Flujo:
    1. Actualiza datos del UCP (full_update_csv)
    2. Ejecuta pipeline de feature engineering
    3. Entrena los 3 modelos (XGBoost, LightGBM, RandomForest)
    4. Selecciona automáticamente el mejor modelo (por rMAPE)
    5. Promociona a producción como champion_model
    6. Retorna métricas del nuevo modelo
    """
    try:
        logger.info("="*80)
        logger.info(f"🔧 INICIANDO REENTRENAMIENTO MANUAL PARA {ucp}")
        logger.info("="*80)

        # PASO 1: Actualizar datos CSV
        logger.info(f"\n📊 PASO 1: Actualizando datos para {ucp}...")
        await run_in_threadpool(full_update_csv, ucp)

        # PASO 2: Ejecutar pipeline de feature engineering
        logger.info(f"\n⚙️ PASO 2: Ejecutando pipeline de feature engineering...")

        power_data_path = f'data/raw/{ucp}/datos.csv'
        weather_data_path = f'data/raw/{ucp}/clima_new.csv'
        output_dir = Path(f'data/features/{ucp}')

        df_with_features, _ = run_automated_pipeline(
            power_data_path=power_data_path,
            weather_data_path=weather_data_path,
            start_date='2015-01-01',
            end_date=None,  # Hasta el día más reciente disponible
            output_dir=output_dir
        )

        logger.info(f"✓ Pipeline completado: {len(df_with_features)} registros")

        # PASO 3: Forzar reentrenamiento de modelos
        logger.info(f"\n🤖 PASO 3: Reentrenando modelos para {ucp}...")

        model_path, train_metrics = train_model_if_needed(
            df_with_features=df_with_features,
            ucp=ucp,
            force_retrain=True  # ← FORZAR reentrenamiento
        )

        logger.info(f"✓ Reentrenamiento completado exitosamente")

        # PASO 4: Retornar información del nuevo modelo
        logger.info("\n" + "="*80)
        logger.info("✅ REENTRENAMIENTO COMPLETADO")
        logger.info("="*80)

        return {
            "status": "success",
            "message": f"Modelo reentrenado exitosamente para {ucp}",
            "ucp": ucp,
            "modelo_champion": model_path.stem,
            "modelo_path": str(model_path),
            "fecha_reentrenamiento": datetime.now().isoformat(),
            "datos_entrenamiento": {
                "total_registros": len(df_with_features),
                "fecha_inicio": df_with_features['FECHA'].min().strftime('%Y-%m-%d') if 'FECHA' in df_with_features.columns else None,
                "fecha_fin": df_with_features['FECHA'].max().strftime('%Y-%m-%d') if 'FECHA' in df_with_features.columns else None
            },
            "metricas": train_metrics if train_metrics else "Modelo ya existía (no se reentrenó)"
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Datos no encontrados para UCP '{ucp}': {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error en reentrenamiento: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en reentrenamiento: {str(e)}"
        )


@app.get("/models", status_code=status.HTTP_200_OK)
async def list_models(ucp: Optional[str] = None):
    """
    Lista modelos disponibles en el sistema

    Args:
        ucp: (Opcional) Nombre del UCP para listar modelos específicos

    Returns:
        Dict con información de modelos entrenados
    """
    if ucp:
        # Listar modelos de un UCP específico
        models_dir = Path(f'models/{ucp}/trained')
        registry_path = Path(f'models/{ucp}/registry/champion_model.joblib')

        models = []

        # Listar modelos entrenados del UCP
        if models_dir.exists():
            for model_file in sorted(models_dir.glob('*.joblib'), key=lambda p: p.stat().st_mtime, reverse=True):
                models.append({
                    'name': model_file.stem,
                    'path': str(model_file),
                    'size_mb': round(model_file.stat().st_size / (1024 * 1024), 2),
                    'created': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                })

        # Modelo campeón del UCP
        champion = None
        if registry_path.exists():
            champion = {
                'name': 'champion_model',
                'path': str(registry_path),
                'size_mb': round(registry_path.stat().st_size / (1024 * 1024), 2),
                'created': datetime.fromtimestamp(registry_path.stat().st_mtime).isoformat()
            }

        return {
            'ucp': ucp,
            'total_models': len(models),
            'champion': champion,
            'models': models
        }
    else:
        # Listar todos los UCPs y sus modelos
        models_base = Path('models')
        ucps_info = []

        if models_base.exists():
            for ucp_dir in models_base.iterdir():
                if ucp_dir.is_dir():
                    registry_path = ucp_dir / 'registry' / 'champion_model.joblib'
                    trained_dir = ucp_dir / 'trained'

                    if registry_path.exists() or (trained_dir.exists() and list(trained_dir.glob('*.joblib'))):
                        ucps_info.append({
                            'ucp': ucp_dir.name,
                            'has_champion': registry_path.exists(),
                            'trained_models': len(list(trained_dir.glob('*.joblib'))) if trained_dir.exists() else 0,
                            'champion_path': str(registry_path) if registry_path.exists() else None
                        })

        return {
            'total_ucps': len(ucps_info),
            'ucps': ucps_info,
            'note': 'Use ?ucp=<name> to get detailed info for a specific UCP'
        }


@app.post(
    "/validate-hourly-adjustments",
    response_model=ValidateAdjustmentsResponse,
    status_code=status.HTTP_200_OK,
    summary="Valida y optimiza ajustes a predicciones horarias",
    description="""
    Valida ajustes manuales del usuario a predicciones horarias mientras preserva
    la forma característica del cluster histórico.

    - Carga el cluster correspondiente a la fecha y UCP
    - Aplica soft constraints (tolerancia 15%) para preservar la forma
    - Retorna valores ajustados optimizados + comparación detallada
    - Stateless: no persiste cambios, solo valida y devuelve

    Casos de uso:
    - Usuario ajusta manualmente horas específicas del pronóstico
    - Sistema verifica que ajustes no distorsionen patrón histórico
    - Devuelve sugerencias optimizadas manteniendo intención del usuario
    """
)
async def validate_hourly_adjustments(
    request: ValidateAdjustmentsRequest
) -> ValidateAdjustmentsResponse:
    """
    Valida y optimiza ajustes manuales a predicciones horarias.

    El algoritmo aplica soft constraints iterativos para preservar la forma
    del cluster mientras respeta la intención del usuario en sus ajustes.
    """
    try:
        logger.info(f"Validando ajustes horarios para {request.fecha}, UCP: {request.ucp}")

        # 1. Determinar directorio de modelos para el UCP
        models_dir = Path('models') / request.ucp
        if not models_dir.exists():
            logger.warning(f"Models directory {models_dir} no encontrado, usando directorio default")
            models_dir = Path('models')

        logger.info(f"Cargando modelos desde: {models_dir}")

        # 2. Cargar motor de desagregación horaria
        try:
            hourly_engine = HourlyDisaggregationEngine(
                auto_load=True,
                models_dir=str(models_dir),
                ucp=request.ucp
            )
        except Exception as e:
            logger.error(f"Error cargando HourlyDisaggregationEngine: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error cargando modelos de desagregación horaria: {str(e)}"
            )

        # Verificar que los modelos estén entrenados
        if not hourly_engine.normal_disaggregator.is_fitted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Modelo de desagregación normal no entrenado para UCP {request.ucp}"
            )

        # 3. Determinar tipo de día y obtener cluster
        fecha = pd.to_datetime(request.fecha)

        # Determinar si es día especial
        is_special = False
        if hourly_engine.special_disaggregator.is_fitted:
            is_special = hourly_engine.special_disaggregator.is_special_day(fecha)

        # Seleccionar disaggregator y obtener cluster info
        if is_special:
            logger.info(f"Fecha {request.fecha} identificada como día especial")
            disaggregator = hourly_engine.special_disaggregator
            method = "special"

            # Obtener cluster para la fecha
            mmdd = fecha.strftime("%m-%d")
            cluster_id = disaggregator.cluster_by_date.get(mmdd, -1)

            if cluster_id == -1:
                # Usando perfil promedio de festivos
                logger.info(f"Usando perfil promedio de festivos para {mmdd}")
                senda_referencia = disaggregator.average_holiday_profile
            else:
                logger.info(f"Usando cluster {cluster_id} para fecha especial {mmdd}")
                senda_referencia = disaggregator.cluster_profiles.loc[cluster_id].values
        else:
            logger.info(f"Fecha {request.fecha} identificada como día normal")
            disaggregator = hourly_engine.normal_disaggregator
            method = "normal"

            # Obtener cluster para el día de la semana
            dayofweek = fecha.dayofweek
            cluster_id = disaggregator.cluster_by_dayofweek.get(dayofweek, 0)

            logger.info(f"Usando cluster {cluster_id} para día de semana {dayofweek}")
            senda_referencia = disaggregator.cluster_profiles.loc[cluster_id].values

        # Log información del cluster
        logger.info(
            f"Cluster info: id={cluster_id}, method={method}, "
            f"senda_sum={senda_referencia.sum():.4f}"
        )

        # 4. Crear validador y ejecutar
        validator = HourlyAdjustmentValidator(
            default_tolerance=0.15,  # 15% desviación permitida
            max_iterations=3
        )

        # 5. Validar ajustes
        result = validator.validate_adjustments(
            predicciones_actuales=request.predicciones_actuales,
            ajustes_solicitados=request.ajustes_solicitados,
            senda_referencia=senda_referencia,
            cluster_id=cluster_id,
            method=method
        )

        logger.info(
            f"Validación completa: "
            f"total_ajustado={result['metadata']['total_ajustado']:.2f} MWh, "
            f"periodos_modificados={result['metadata']['periodos_modificados']}, "
            f"desviacion_forma={result['metadata']['desviacion_promedio_forma']:.2f}%"
        )

        # 6. Construir y retornar respuesta
        return ValidateAdjustmentsResponse(**result)

    except HTTPException:
        # Re-lanzar excepciones HTTP
        raise
    except FileNotFoundError as e:
        logger.error(f"Archivos de modelo no encontrados: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelos no encontrados para UCP {request.ucp}. Por favor entrene los modelos primero."
        )
    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error en validación de datos: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error inesperado validando ajustes: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al validar ajustes: {str(e)}"
        )


@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """
    Endpoint raíz con información de la API
    """
    return {
        "api": "EPM Energy Demand Forecasting API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "POST /predict": "Genera predicción de demanda con granularidad horaria",
            "POST /api/v1/base-curve": "Obtiene curvas base de demanda horaria para un rango de fechas",
            "POST /validate-hourly-adjustments": "Valida y optimiza ajustes manuales a predicciones horarias",
            "POST /retrain": "Reentrenamiento manual del modelo (actualiza datos + entrena nuevo modelo)",
            "GET /health": "Estado del sistema",
            "GET /models": "Lista de modelos disponibles"
        }
    }


# ============================================================================
# INICIALIZACIÓN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Evento de inicialización de la API"""
    logger.info("="*80)
    logger.info("🚀 INICIANDO API DE PRONÓSTICO DE DEMANDA ENERGÉTICA - EPM")
    logger.info("="*80)
    logger.info(f"Versión: 1.0.0")
    logger.info(f"Documentación: http://localhost:8000/docs")
    logger.info("="*80)


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre de la API"""
    logger.info("🛑 Apagando API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
