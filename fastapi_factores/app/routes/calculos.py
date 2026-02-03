"""
Router de endpoints para cálculos FDA/FDP y Clustering.
"""

from typing import List
from fastapi import APIRouter, HTTPException, Query
import logging

from app.schemas.factores import (
    CalculoFDARequest,
    CalculoFDPRequest,
    ClusteringRequest,
    CurvasTipicasRequest,
)
from app.services import calculos_service as service

# Configurar logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/factores/calculos", tags=["calculos"])


# =============================================================================
# ENDPOINTS - CLUSTERING
# =============================================================================

@router.post(
    "/clustering",
    summary="Aplicar clustering a medidas",
    description="""
    Aplica factores multiplicadores a las medidas y las agrupa por barra y fecha.

    **Algoritmo**:
    1. Obtiene códigos RPM asociados a la barra
    2. Obtiene factores de cada código RPM
    3. Multiplica cada periodo (P1-P24) por su factor correspondiente
    4. Agrupa por barra+fecha sumando los periodos
    5. Retorna medidas ajustadas

    **Casos de uso**:
    - Preparación de datos para cálculos FDA/FDP
    - Análisis de medidas consolidadas por barra
    - Validación de factores aplicados
    """
)
def aplicar_clustering(payload: ClusteringRequest):
    """
    Aplica factores multiplicadores a medidas agrupadas por barra.
    """
    try:
        logger.info(
            f"Clustering: barra={payload.barra}, mc={payload.mc}, "
            f"tipo_dia={payload.tipo_dia or 'TODOS'}, flujo={payload.flujo_tipo}"
        )
        
        # Validaciones
        from datetime import datetime
        from app.services import factores_service
        
        # Validar rango de fechas
        fecha_ini = datetime.strptime(payload.fecha_inicial, "%Y-%m-%d")
        fecha_fin = datetime.strptime(payload.fecha_final, "%Y-%m-%d")
        if fecha_fin < fecha_ini:
            raise ValueError("fecha_final debe ser mayor o igual a fecha_inicial")
        
        # Validar tipo de flujo
        if payload.flujo_tipo not in ['A', 'R']:
            raise ValueError("flujo_tipo debe ser 'A' (Activa) o 'R' (Reactiva)")
        
        # Validar que la barra existe
        barra_exists = factores_service.consultar_barra_nombre(payload.barra)
        if not barra_exists:
            raise ValueError(f"La barra '{payload.barra}' no existe")

        data = service.aplicar_clustering(
            payload.fecha_inicial,
            payload.fecha_final,
            payload.mc,
            payload.barra,
            payload.flujo_tipo,
            payload.tipo_dia or "",
        )

        if not data:
            logger.warning(f"No se encontraron medidas para la barra {payload.barra}")
            return {
                "ok": True,
                "data": [],
                "mensaje": "No se encontraron medidas en el rango especificado"
            }

        logger.info(f"Clustering exitoso: {len(data)} registros procesados")
        return {"ok": True, "data": data}

    except ValueError as e:
        logger.error(f"Error de validación en clustering: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error interno en clustering: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno en clustering: {str(e)}")


@router.post(
    "/curvas-tipicas",
    summary="Obtener las N curvas más típicas del histórico",
    description="""
    Detecta patrones en el histórico y devuelve hasta N curvas más típicas (forma y nivel).

    **Flujo**:
    1. Obtiene todas las curvas clusterizadas en el rango/MC/tipo_día (una barra o todas las del MC).
    2. Normaliza por forma (L2) y mide centralidad (distancia media a las demás).
    3. Devuelve hasta n_max más típicas; si el cluster solo encuentra menos, devuelve esas.

    **Uso**: Llamar primero a curvas-tipicas, luego pasar esa lista a FDA y FDP para que se calculen solo sobre las curvas seleccionadas.
    """
)
def obtener_curvas_tipicas(payload: CurvasTipicasRequest):
    try:
        from datetime import datetime
        from app.services import factores_service

        fecha_ini = datetime.strptime(payload.fecha_inicial, "%Y-%m-%d")
        fecha_fin = datetime.strptime(payload.fecha_final, "%Y-%m-%d")
        if fecha_fin < fecha_ini:
            raise ValueError("fecha_final debe ser mayor o igual a fecha_inicial")
        if payload.flujo_tipo not in ("A", "R"):
            raise ValueError("flujo_tipo debe ser 'A' o 'R'")
        if payload.tipo_dia not in ("ORDINARIO", "SABADO", "FESTIVO"):
            raise ValueError("tipo_dia debe ser ORDINARIO, SABADO o FESTIVO")
        if payload.barra:
            barra_exists = factores_service.consultar_barra_nombre(payload.barra)
            if not barra_exists:
                raise ValueError(f"La barra '{payload.barra}' no existe")

        data = service.obtener_curvas_tipicas(
            payload.fecha_inicial,
            payload.fecha_final,
            payload.mc,
            payload.tipo_dia,
            payload.flujo_tipo,
            payload.n_max,
            payload.barra,
        )
        return {"ok": True, "data": data, "n": len(data)}
    except ValueError as e:
        logger.error(f"Error de validación en curvas-tipicas: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error interno en curvas-tipicas: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# =============================================================================
# ENDPOINTS - FDA
# =============================================================================

@router.post(
    "/fda",
    summary="Calcular FDA (Factor de Demanda Ajustada) sobre curvas típicas",
    description="""
    Calcula FDA **solo sobre las curvas típicas** indicadas (salida de curvas-tipicas).

    **Flujo**: Llamar primero a POST /curvas-tipicas, luego pasar esa lista aquí.

    **Algoritmo FDA** (sobre esas curvas):
    1. Obtiene medidas clusterizadas solo para las (barra, fecha) indicadas
    2. Aplica ajuste: 1.0 - suma por periodo, aplicado solo al máximo de cada periodo
    3. Resultado: factores normalizados que suman exactamente 1.0
    """
)
def calcular_fda(payload: CalculoFDARequest):
    try:
        logger.info(
            f"FDA: mc={payload.mc}, tipo_dia={payload.tipo_dia}, "
            f"curvas_tipicas={len(payload.curvas_tipicas)}"
        )
        from datetime import datetime

        fecha_ini = datetime.strptime(payload.fecha_inicial, "%Y-%m-%d")
        fecha_fin = datetime.strptime(payload.fecha_final, "%Y-%m-%d")
        if fecha_fin < fecha_ini:
            raise ValueError("fecha_final debe ser mayor o igual a fecha_inicial")
        if not payload.curvas_tipicas:
            raise ValueError("curvas_tipicas no puede estar vacío")
        if payload.tipo_dia not in ("ORDINARIO", "SABADO", "FESTIVO"):
            raise ValueError("tipo_dia debe ser ORDINARIO, SABADO o FESTIVO")

        curvas = [{"barra": c.barra, "fecha": c.fecha} for c in payload.curvas_tipicas]
        resultado = service.calcular_fda_para_tipo_dia(
            payload.fecha_inicial,
            payload.fecha_final,
            payload.mc,
            payload.tipo_dia,
            curvas,
        )
        suma = resultado.get("suma_total", 0)
        if resultado.get("n_registros", 0) > 0 and abs(suma - 1.0) > 1e-5:
            logger.warning(f"FDA {payload.tipo_dia}: suma_total={suma:.6f} (esperado: 1.0)")

        return {
            "ok": True,
            "mc": payload.mc,
            "tipo_dia": payload.tipo_dia,
            "fecha_inicial": payload.fecha_inicial,
            "fecha_final": payload.fecha_final,
            "resultado": resultado,
        }
    except ValueError as e:
        logger.error(f"Error de validación en FDA: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error interno en FDA: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno en FDA: {str(e)}")


# =============================================================================
# ENDPOINTS - FDP
# =============================================================================

@router.post(
    "/fdp",
    summary="Calcular FDP (Factor de Demanda Pronóstico) sobre curvas típicas",
    description="""
    Calcula FDP **solo sobre las curvas típicas** indicadas (salida de curvas-tipicas).

    **Flujo**: Llamar primero a POST /curvas-tipicas, luego pasar esa lista aquí.

    **Fórmula**: FDP = Cos(Atan(Potencia_Reactiva / Potencia_Activa)). Requiere medidas A y R para esas curvas.
    """
)
def calcular_fdp(payload: CalculoFDPRequest):
    try:
        logger.info(
            f"FDP: mc={payload.mc}, tipo_dia={payload.tipo_dia}, "
            f"curvas_tipicas={len(payload.curvas_tipicas)}"
        )
        from datetime import datetime

        fecha_ini = datetime.strptime(payload.fecha_inicial, "%Y-%m-%d")
        fecha_fin = datetime.strptime(payload.fecha_final, "%Y-%m-%d")
        if fecha_fin < fecha_ini:
            raise ValueError("fecha_final debe ser mayor o igual a fecha_inicial")
        if not payload.curvas_tipicas:
            raise ValueError("curvas_tipicas no puede estar vacío")
        if payload.tipo_dia not in ("ORDINARIO", "SABADO", "FESTIVO"):
            raise ValueError("tipo_dia debe ser ORDINARIO, SABADO o FESTIVO")

        curvas = [{"barra": c.barra, "fecha": c.fecha} for c in payload.curvas_tipicas]
        resultado = service.calcular_fdp_para_tipo_dia(
            payload.fecha_inicial,
            payload.fecha_final,
            payload.mc,
            payload.tipo_dia,
            curvas,
        )
        return {
            "ok": True,
            "mc": payload.mc,
            "tipo_dia": payload.tipo_dia,
            "fecha_inicial": payload.fecha_inicial,
            "fecha_final": payload.fecha_final,
            "resultado": resultado,
        }
    except ValueError as e:
        logger.error(f"Error de validación en FDP: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error interno en FDP: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno en FDP: {str(e)}")


# =============================================================================
# ENDPOINTS - COMBINADO
# =============================================================================

@router.post(
    "/fda-fdp",
    summary="Calcular FDA y FDP en una sola llamada sobre curvas típicas",
    description="""
    Calcula FDA y FDP **solo sobre las curvas típicas** indicadas en una sola operación.
    Flujo: llamar primero a curvas-tipicas, luego pasar esa lista aquí.
    """
)
def calcular_fda_y_fdp(payload: CalculoFDARequest):
    try:
        logger.info(
            f"FDA+FDP: mc={payload.mc}, tipo_dia={payload.tipo_dia}, "
            f"curvas_tipicas={len(payload.curvas_tipicas)}"
        )
        from datetime import datetime

        fecha_ini = datetime.strptime(payload.fecha_inicial, "%Y-%m-%d")
        fecha_fin = datetime.strptime(payload.fecha_final, "%Y-%m-%d")
        if fecha_fin < fecha_ini:
            raise ValueError("fecha_final debe ser mayor o igual a fecha_inicial")
        if not payload.curvas_tipicas:
            raise ValueError("curvas_tipicas no puede estar vacío")
        if payload.tipo_dia not in ("ORDINARIO", "SABADO", "FESTIVO"):
            raise ValueError("tipo_dia debe ser ORDINARIO, SABADO o FESTIVO")

        curvas = [{"barra": c.barra, "fecha": c.fecha} for c in payload.curvas_tipicas]
        fda_resultado = service.calcular_fda_para_tipo_dia(
            payload.fecha_inicial,
            payload.fecha_final,
            payload.mc,
            payload.tipo_dia,
            curvas,
        )
        fdp_resultado = service.calcular_fdp_para_tipo_dia(
            payload.fecha_inicial,
            payload.fecha_final,
            payload.mc,
            payload.tipo_dia,
            curvas,
        )
        return {
            "ok": True,
            "mc": payload.mc,
            "tipo_dia": payload.tipo_dia,
            "fecha_inicial": payload.fecha_inicial,
            "fecha_final": payload.fecha_final,
            "fda": fda_resultado,
            "fdp": fdp_resultado,
        }
    except ValueError as e:
        logger.error(f"Error de validación en FDA+FDP: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error interno en FDA+FDP: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# =============================================================================
# ENDPOINTS - UTILIDADES
# =============================================================================

@router.get(
    "/health",
    summary="Health check del servicio de cálculos",
    description="Verifica que el servicio de cálculos esté funcionando correctamente."
)
def health_check():
    """Verifica que el servicio de cálculos esté funcionando."""
    return {
        "ok": True,
        "service": "calculos_fda_fdp",
        "version": "1.0.0",
        "endpoints": {
            "clustering": "/factores/calculos/clustering",
            "curvas_tipicas": "/factores/calculos/curvas-tipicas",
            "fda": "/factores/calculos/fda",
            "fdp": "/factores/calculos/fdp",
            "fda_fdp": "/factores/calculos/fda-fdp"
        }
    }
