"""
Sistema de Validación de Ajustes Horarios

Este módulo implementa el algoritmo de validación y optimización de ajustes
manuales a predicciones horarias, preservando la forma característica del cluster.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HourlyAdjustmentValidator:
    """
    Validador y optimizador de ajustes manuales a predicciones horarias.

    Aplica soft constraints para preservar la forma del cluster mientras
    respeta la intención del usuario en sus ajustes.

    Algoritmo:
    1. Aplica ajustes iniciales del usuario
    2. Itera verificando desviación de cada periodo respecto al cluster
    3. Limita desviaciones excesivas (>tolerancia) mediante clipping de ratios
    4. Converge en máximo max_iterations iteraciones

    Características:
    - Tolerancia permisiva configurable (default: 15%)
    - Permite cambio en total diario (no fuerza suma original)
    - Manejo de edge cases (negativos, extremos)
    - Retorna comparación detallada de ajustes
    """

    def __init__(
        self,
        default_tolerance: float = 0.15,
        max_iterations: int = 3,
        extreme_threshold: float = 1.0,
        min_value_threshold: float = 0.0
    ):
        """
        Inicializa el validador.

        Args:
            default_tolerance: Tolerancia máxima de desviación por periodo (default: 15%)
            max_iterations: Número máximo de iteraciones para convergencia
            extreme_threshold: Umbral para detectar ajustes extremos (>100% cambio)
            min_value_threshold: Valor mínimo permitido para un periodo (default: 0.0)
        """
        self.default_tolerance = default_tolerance
        self.max_iterations = max_iterations
        self.extreme_threshold = extreme_threshold
        self.min_value_threshold = min_value_threshold

        logger.info(
            f"HourlyAdjustmentValidator inicializado: "
            f"tolerance={default_tolerance*100:.1f}%, "
            f"max_iterations={max_iterations}"
        )

    def validate_adjustments(
        self,
        predicciones_actuales: List[float],
        ajustes_solicitados: List[float],
        senda_referencia: np.ndarray,
        cluster_id: int,
        method: str,
        tolerance: Optional[float] = None
    ) -> Dict:
        """
        Valida y optimiza ajustes manuales a predicciones horarias.

        Args:
            predicciones_actuales: Lista de 24 valores actuales (P1-P24) en MWh
            ajustes_solicitados: Lista de 24 ajustes (+/-/0) en MWh
            senda_referencia: Array normalizado del cluster (24 valores, sum=1.0)
            cluster_id: ID del cluster usado en la predicción original
            method: Método de desagregación ('normal' o 'special')
            tolerance: Tolerancia personalizada (si None, usa default_tolerance)

        Returns:
            Dict con:
                - valores_ajustados: List[float] con 24 valores optimizados
                - comparacion: List[Dict] con comparación periodo por periodo
                - metadata: Dict con información del proceso
        """
        # Validaciones
        if len(predicciones_actuales) != 24:
            raise ValueError(f"predicciones_actuales debe tener 24 valores, tiene {len(predicciones_actuales)}")
        if len(ajustes_solicitados) != 24:
            raise ValueError(f"ajustes_solicitados debe tener 24 valores, tiene {len(ajustes_solicitados)}")
        if len(senda_referencia) != 24:
            raise ValueError(f"senda_referencia debe tener 24 valores, tiene {len(senda_referencia)}")

        # Usar tolerancia especificada o default
        tol = tolerance if tolerance is not None else self.default_tolerance

        # Log de inicio
        total_original = sum(predicciones_actuales)
        ajuste_total_solicitado = sum(ajustes_solicitados)
        logger.info(
            f"Validando ajustes: cluster_id={cluster_id}, method={method}, "
            f"total_original={total_original:.2f} MWh, "
            f"ajuste_total={ajuste_total_solicitado:+.2f} MWh"
        )

        # Convertir a numpy arrays
        valores_originales = np.array(predicciones_actuales, dtype=float)
        ajustes = np.array(ajustes_solicitados, dtype=float)
        senda = np.array(senda_referencia, dtype=float)

        # Normalizar senda si no está normalizada
        if not np.isclose(senda.sum(), 1.0, atol=0.01):
            logger.warning(f"senda_referencia no suma 1.0 (suma={senda.sum():.4f}), normalizando...")
            senda = senda / senda.sum()

        # Paso 1: Aplicar ajustes iniciales
        valores_ajustados = valores_originales + ajustes

        # Detectar ajustes extremos
        extreme_adjustments = self._detect_extreme_adjustments(
            valores_originales, ajustes, valores_ajustados
        )
        if extreme_adjustments:
            logger.warning(f"Detectados {len(extreme_adjustments)} ajustes extremos: {extreme_adjustments}")

        # Guardar valores solicitados para comparación
        valores_solicitados = valores_ajustados.copy()

        # Paso 2: Iteración de constraints
        iteration = 0
        reasons = {}  # periodo_index -> razón de modificación

        while iteration < self.max_iterations:
            modified = False
            total_ajustado = valores_ajustados.sum()

            # Prevenir división por cero
            if total_ajustado <= 0:
                logger.error("Total ajustado es cero o negativo, abortando")
                raise ValueError("Los ajustes resultan en un total diario inválido (≤0)")

            # Verificar cada periodo
            for i in range(24):
                # Calcular ratios
                ratio_actual = valores_ajustados[i] / total_ajustado
                ratio_cluster = senda[i]

                # Calcular desviación relativa
                if ratio_cluster > 0:
                    deviation = abs(ratio_actual - ratio_cluster) / ratio_cluster
                else:
                    # Si el cluster tiene ratio ~0, permitir más flexibilidad
                    deviation = 0 if ratio_actual < 0.001 else 1.0

                # Si excede tolerancia, aplicar constraint
                if deviation > tol:
                    # Calcular límites
                    ratio_min = ratio_cluster * (1 - tol)
                    ratio_max = ratio_cluster * (1 + tol)

                    # Clampear ratio
                    ratio_constrained = np.clip(ratio_actual, ratio_min, ratio_max)

                    # Recalcular valor
                    valor_constrained = ratio_constrained * total_ajustado

                    # Aplicar constraint solo si cambia significativamente
                    if abs(valor_constrained - valores_ajustados[i]) > 0.01:
                        # Guardar razón si es la primera vez que se modifica
                        if i not in reasons:
                            ajuste_original = ajustes[i]
                            ajuste_aplicado = valor_constrained - valores_originales[i]
                            reasons[i] = self._generate_reason(
                                i, ajuste_original, ajuste_aplicado,
                                deviation, tol, iteration
                            )

                        valores_ajustados[i] = valor_constrained
                        modified = True

            # Si no hubo modificaciones, converged
            if not modified:
                logger.info(f"Convergencia alcanzada en iteración {iteration}")
                break

            iteration += 1

        # Log final de iteraciones
        if iteration >= self.max_iterations:
            logger.warning(f"Alcanzado máximo de iteraciones ({self.max_iterations})")

        # Paso 3: Manejo de valores negativos
        valores_ajustados = self._handle_negative_values(valores_ajustados, valores_originales, reasons)

        # Paso 4: Construir respuesta
        comparacion = self._build_comparison(
            valores_originales,
            ajustes,
            valores_ajustados,
            reasons
        )

        # Calcular metadata
        total_ajustado_final = valores_ajustados.sum()
        avg_deviation = self._calculate_shape_deviation(valores_ajustados, senda)

        num_modified = len(reasons)
        num_extreme = len(extreme_adjustments)

        metadata = {
            'cluster_id': int(cluster_id),
            'metodo': method,
            'total_original': float(total_original),
            'total_ajustado': float(total_ajustado_final),
            'cambio_total': float(total_ajustado_final - total_original),
            'desviacion_promedio_forma': float(avg_deviation * 100),  # En porcentaje
            'iteraciones_convergencia': iteration,
            'periodos_modificados': num_modified,
            'ajustes_extremos_detectados': num_extreme,
            'tolerancia_aplicada': float(tol * 100)  # En porcentaje
        }

        logger.info(
            f"Validación completa: "
            f"total_ajustado={total_ajustado_final:.2f} MWh, "
            f"desviación_forma={avg_deviation*100:.2f}%, "
            f"periodos_modificados={num_modified}"
        )

        return {
            'valores_ajustados': valores_ajustados.tolist(),
            'comparacion': comparacion,
            'metadata': metadata
        }

    def _constrain_period(
        self,
        valor_original: float,
        valor_ajustado: float,
        senda_ratio: float,
        total_ajustado: float,
        tolerance: float
    ) -> Tuple[float, Optional[str]]:
        """
        Aplica constraint a un periodo individual.

        Args:
            valor_original: Valor original de predicción
            valor_ajustado: Valor después de ajuste del usuario
            senda_ratio: Ratio normalizado del cluster (0-1)
            total_ajustado: Suma total de todos los valores ajustados
            tolerance: Tolerancia de desviación

        Returns:
            Tupla (valor_constrained, razón)
        """
        ratio_actual = valor_ajustado / total_ajustado

        # Calcular desviación
        if senda_ratio > 0:
            deviation = abs(ratio_actual - senda_ratio) / senda_ratio
        else:
            deviation = 0 if ratio_actual < 0.001 else 1.0

        # Si dentro de tolerancia, no modificar
        if deviation <= tolerance:
            return valor_ajustado, None

        # Aplicar límites
        ratio_min = senda_ratio * (1 - tolerance)
        ratio_max = senda_ratio * (1 + tolerance)
        ratio_constrained = np.clip(ratio_actual, ratio_min, ratio_max)

        # Recalcular valor
        valor_constrained = ratio_constrained * total_ajustado

        # Generar razón
        ajuste_original = valor_ajustado - valor_original
        ajuste_aplicado = valor_constrained - valor_original

        razon = (
            f"Ajuste reducido de {ajuste_original:+.1f} MWh a {ajuste_aplicado:+.1f} MWh "
            f"para preservar forma del cluster (desviación {deviation*100:.1f}% > {tolerance*100:.0f}%)"
        )

        return valor_constrained, razon

    def _calculate_shape_deviation(
        self,
        valores_ajustados: np.ndarray,
        senda_referencia: np.ndarray
    ) -> float:
        """
        Calcula la desviación promedio de forma respecto al cluster.

        Args:
            valores_ajustados: Array de 24 valores ajustados
            senda_referencia: Array normalizado del cluster (24 valores, sum=1.0)

        Returns:
            Desviación promedio (0-1, donde 0=sin desviación)
        """
        total = valores_ajustados.sum()

        if total <= 0:
            return 0.0

        # Normalizar valores ajustados
        ratios_ajustados = valores_ajustados / total

        # Calcular desviaciones absolutas relativas
        deviations = []
        for i in range(24):
            if senda_referencia[i] > 0:
                dev = abs(ratios_ajustados[i] - senda_referencia[i]) / senda_referencia[i]
                deviations.append(dev)

        # Promedio
        if deviations:
            return np.mean(deviations)
        else:
            return 0.0

    def _handle_negative_values(
        self,
        valores_ajustados: np.ndarray,
        valores_originales: np.ndarray,
        reasons: Dict[int, str]
    ) -> np.ndarray:
        """
        Maneja valores negativos clampeándolos al umbral mínimo.

        Args:
            valores_ajustados: Array de valores ajustados
            valores_originales: Array de valores originales
            reasons: Dict de razones de modificación (se actualiza in-place)

        Returns:
            Array de valores ajustados sin negativos
        """
        for i in range(len(valores_ajustados)):
            if valores_ajustados[i] < self.min_value_threshold:
                logger.warning(
                    f"Periodo P{i+1}: valor negativo {valores_ajustados[i]:.2f} MWh, "
                    f"clampeando a {self.min_value_threshold:.2f} MWh"
                )

                # Guardar razón
                if i not in reasons:
                    ajuste_original = valores_ajustados[i] - valores_originales[i]
                    reasons[i] = (
                        f"Ajuste {ajuste_original:+.1f} MWh resultaba en valor negativo "
                        f"({valores_ajustados[i]:.1f} MWh), clampeado a {self.min_value_threshold:.1f} MWh"
                    )

                valores_ajustados[i] = self.min_value_threshold

        return valores_ajustados

    def _detect_extreme_adjustments(
        self,
        valores_originales: np.ndarray,
        ajustes: np.ndarray,
        valores_ajustados: np.ndarray
    ) -> List[int]:
        """
        Detecta ajustes extremos (>extreme_threshold de cambio relativo).

        Args:
            valores_originales: Valores originales
            ajustes: Ajustes solicitados
            valores_ajustados: Valores después de ajustes

        Returns:
            Lista de índices de periodos con ajustes extremos
        """
        extreme_indices = []

        for i in range(len(valores_originales)):
            if valores_originales[i] > 0:
                cambio_relativo = abs(ajustes[i]) / valores_originales[i]
                if cambio_relativo > self.extreme_threshold:
                    extreme_indices.append(i)

        return extreme_indices

    def _generate_reason(
        self,
        periodo_index: int,
        ajuste_original: float,
        ajuste_aplicado: float,
        deviation: float,
        tolerance: float,
        iteration: int
    ) -> str:
        """
        Genera una razón legible de por qué se modificó un ajuste.

        Args:
            periodo_index: Índice del periodo (0-23)
            ajuste_original: Ajuste solicitado por el usuario
            ajuste_aplicado: Ajuste realmente aplicado
            deviation: Desviación que causó la modificación
            tolerance: Tolerancia aplicada
            iteration: Iteración en la que se detectó

        Returns:
            String con explicación
        """
        periodo = f"P{periodo_index + 1}"

        # Determinar tipo de modificación
        if abs(ajuste_aplicado) < abs(ajuste_original):
            tipo = "reducido"
        elif abs(ajuste_aplicado) > abs(ajuste_original):
            tipo = "incrementado"
        else:
            tipo = "ajustado"

        # Determinar razón general
        if periodo_index < 7:  # P1-P7: valle nocturno
            contexto = "valle nocturno"
        elif periodo_index >= 18 and periodo_index <= 20:  # P19-P21: pico tarde
            contexto = "pico de tarde"
        elif periodo_index >= 10 and periodo_index <= 13:  # P11-P14: pico mañana
            contexto = "pico de mañana"
        else:
            contexto = "patrón del cluster"

        razon = (
            f"Ajuste {tipo} de {ajuste_original:+.1f} MWh a {ajuste_aplicado:+.1f} MWh "
            f"para preservar {contexto} "
            f"(desviación {deviation*100:.1f}% > {tolerance*100:.0f}%)"
        )

        return razon

    def _build_comparison(
        self,
        valores_originales: np.ndarray,
        ajustes_solicitados: np.ndarray,
        valores_ajustados: np.ndarray,
        reasons: Dict[int, str]
    ) -> List[Dict]:
        """
        Construye la lista de comparación periodo por periodo.

        Args:
            valores_originales: Valores originales
            ajustes_solicitados: Ajustes solicitados por usuario
            valores_ajustados: Valores finales después de validación
            reasons: Dict con razones de modificación por índice

        Returns:
            Lista de 24 dicts con comparación detallada
        """
        comparacion = []

        for i in range(24):
            ajuste_solicitado = ajustes_solicitados[i]
            ajuste_aplicado = valores_ajustados[i] - valores_originales[i]

            comparacion.append({
                'periodo': f'P{i + 1}',
                'valor_original': float(valores_originales[i]),
                'ajuste_solicitado': float(ajuste_solicitado),
                'ajuste_aplicado': float(ajuste_aplicado),
                'valor_final': float(valores_ajustados[i]),
                'razon_modificacion': reasons.get(i, None)
            })

        return comparacion


# ============== FUNCIONES DE UTILIDAD ==============

def quick_validate(
    predicciones: List[float],
    ajustes: List[float],
    senda: np.ndarray,
    tolerance: float = 0.15
) -> Dict:
    """
    Función de conveniencia para validación rápida sin metadatos completos.

    Args:
        predicciones: 24 valores de predicción actual
        ajustes: 24 ajustes solicitados
        senda: 24 valores de senda normalizada
        tolerance: Tolerancia de desviación (default: 15%)

    Returns:
        Dict con valores_ajustados y metadata básica
    """
    validator = HourlyAdjustmentValidator(default_tolerance=tolerance)

    result = validator.validate_adjustments(
        predicciones_actuales=predicciones,
        ajustes_solicitados=ajustes,
        senda_referencia=senda,
        cluster_id=-1,
        method='unknown'
    )

    return {
        'valores_ajustados': result['valores_ajustados'],
        'total_original': sum(predicciones),
        'total_ajustado': sum(result['valores_ajustados']),
        'desviacion_forma': result['metadata']['desviacion_promedio_forma']
    }
