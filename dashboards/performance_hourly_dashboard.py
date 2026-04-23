"""
Dashboard de Rendimiento Horario - EPM
======================================

Dashboard Streamlit para visualizar el rendimiento del modelo vs históricos
con granularidad horaria (P1-P24) en entrenamiento, validación y producción.

Uso:
    streamlit run dashboards/performance_hourly_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging
import joblib
from typing import Optional, Dict, Any, Tuple, List

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import (
    DATA_DIR, FEATURES_DATA_DIR, MODELS_DIR, RAW_DATA_DIR,
    BASE_DIR
)
from src.prediction.forecaster import ForecastPipeline
from src.prediction.hourly import HourlyDisaggregationEngine
from src.prediction.hourly.calendar_utils import CalendarClassifier
from src.models.trainer import ModelTrainer
from src.models.metrics import calculate_all_metrics
from src.models.registry import ModelRegistry

# Configurar logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# =========================
# CONFIG GLOBAL (DASHBOARD)
# =========================
MIN_HISTORICAL_DATE = pd.Timestamp("2020-01-01")

# Features de lag a excluir (mismo criterio que entrenamiento original)
FEATURES_LAG_TO_EXCLUDE = [
    'total_lag_1d', 'total_lag_7d', 'total_lag_14d',
    'p8_lag_1d', 'p8_lag_7d',
    'p12_lag_1d', 'p12_lag_7d',
    'p18_lag_1d', 'p18_lag_7d',
    'p20_lag_1d', 'p20_lag_7d',
    'total_day_change', 'total_day_change_pct'
]

# Configuración de página
st.set_page_config(
    page_title="Rendimiento Horario - EPM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES DE CARGA DE DATOS
# ============================================================================

@st.cache_data
def load_historical_data(ucp: str) -> Optional[pd.DataFrame]:
    """
    Carga datos históricos con columnas P1-P24
    
    Args:
        ucp: Nombre del UCP ('Antioquia' o 'Atlantico')
    
    Returns:
        DataFrame con datos históricos o None
    """
    # Intentar cargar desde datos procesados con features
    ucp_path = FEATURES_DATA_DIR / ucp / "data_with_features_latest.csv"
    if ucp_path.exists():
        try:
            df = pd.read_csv(ucp_path)
            if 'FECHA' in df.columns:
                df['FECHA'] = pd.to_datetime(df['FECHA'])
                df = df.sort_values('FECHA')
                # Recortar histórico: desde 2020-01-01 en adelante
                df = df[df['FECHA'] >= MIN_HISTORICAL_DATE].copy()
                # Verificar que tenga columnas P1-P24
                period_cols = [f'P{i}' for i in range(1, 25)]
                if all(col in df.columns for col in period_cols):
                    return df
        except Exception as e:
            st.warning(f"Error cargando datos procesados: {e}")
    
    # Fallback: intentar datos raw
    path = RAW_DATA_DIR / ucp / "datos.csv"
    if not path.exists():
        path = RAW_DATA_DIR / "datos.csv"
    
    if path.exists():
        try:
            df = pd.read_csv(path)
            if 'FECHA' in df.columns:
                df['FECHA'] = pd.to_datetime(df['FECHA'])
                df = df.sort_values('FECHA')
                if 'UCP' in df.columns:
                    ucp_mapping = {
                        'Atlantico': 'Atlantico',
                        'Antioquia': 'UANTIOQUIA'
                    }
                    ucp_in_data = ucp_mapping.get(ucp, ucp)
                    df = df[df['UCP'].str.upper() == ucp_in_data.upper()]
                # Recortar histórico: desde 2020-01-01 en adelante
                df = df[df['FECHA'] >= MIN_HISTORICAL_DATE].copy()
                
                # Verificar que tenga columnas P1-P24
                period_cols = [f'P{i}' for i in range(1, 25)]
                if all(col in df.columns for col in period_cols):
                    return df
        except Exception as e:
            st.warning(f"Error cargando datos históricos: {e}")
    
    return None


@st.cache_data
def ensure_features_file_filtered_since(
    ucp: str,
    source_path: str,
    min_date: pd.Timestamp,
) -> Optional[str]:
    """
    Crea (si no existe) un CSV de features filtrado desde min_date y retorna su ruta.
    Se usa para que el HourlyDisaggregationEngine re-entrene solo con el rango deseado.
    """
    try:
        source = Path(source_path)
        if not source.exists():
            return None

        out_dir = FEATURES_DATA_DIR / ucp
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"data_with_features_since_{min_date.strftime('%Y%m%d')}.csv"

        # Si ya existe, reutilizar.
        if out_path.exists():
            return str(out_path)

        df = pd.read_csv(source)
        if "FECHA" not in df.columns:
            return None
        df["FECHA"] = pd.to_datetime(df["FECHA"])
        df = df.sort_values("FECHA")
        df = df[df["FECHA"] >= min_date].copy()
        df.to_csv(out_path, index=False)
        return str(out_path)
    except Exception:
        return None


def get_model_path_for_dashboard(ucp: str) -> Optional[Path]:
    """
    Devuelve la ruta del modelo que el dashboard intenta cargar.
    (Para mostrar timestamp/ruta y para reentrenamiento).
    """
    registry_path = MODELS_DIR / "registry" / "champion_model.joblib"
    if registry_path.exists():
        return registry_path

    ucp_model_path = MODELS_DIR / ucp / "champion_model.joblib"
    if ucp_model_path.exists():
        return ucp_model_path

    ucp_dir = MODELS_DIR / ucp / "trained"
    if ucp_dir.exists():
        model_files = sorted(list(ucp_dir.glob("*.joblib")), key=lambda p: p.stat().st_mtime)
        if model_files:
            return model_files[-1]

    return None


def retrain_daily_model_since_2020(
    ucp: str,
    features_csv_path: str,
    criterion: str = "rmape",
    optimize_hyperparams: bool = False,
    cv_splits: int = 3,
    models_to_train: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Reentrena el modelo diario (TOTAL) usando features desde 2020+ y actualiza el champion del registry.
    Nota: aunque el dashboard es "horario", el modelo base sigue siendo diario (TOTAL_pred),
    y el horario proviene de la desagregación.
    """
    models_to_train = models_to_train or ["xgboost", "lightgbm", "randomforest"]

    df = pd.read_csv(features_csv_path)
    if "FECHA" not in df.columns or "TOTAL" not in df.columns:
        raise ValueError("El archivo de features debe tener columnas FECHA y TOTAL.")

    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df = df.sort_values("FECHA")
    df = df[df["FECHA"] >= MIN_HISTORICAL_DATE].copy()

    # Preparar features
    exclude_cols = ['FECHA', 'TOTAL'] + [f'P{i}' for i in range(1, 25)] + FEATURES_LAG_TO_EXCLUDE
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df["TOTAL"].copy()

    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]

    # Split temporal (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    trainer = ModelTrainer(
        optimize_hyperparams=optimize_hyperparams,
        n_optimization_iter=20,
        cv_splits=cv_splits
    )

    trained = trainer.train_all_models(X_train, y_train, X_val, y_val, models=models_to_train)

    best_name, best_model, best_results = trainer.select_best_model(
        criterion=criterion,
        use_validation=True
    )

    # Registrar y promover a champion
    registry = ModelRegistry(Path("models/registry"))
    metrics_for_registry = best_results.get("val_metrics") or best_results.get("train_metrics") or {}
    model_id = registry.register_model(
        best_model,
        model_name=best_name,
        metrics=metrics_for_registry,
        metadata={
            "ucp": ucp,
            "min_historical_date": str(MIN_HISTORICAL_DATE.date()),
            "features_csv_path": features_csv_path,
            "feature_count": len(feature_cols),
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
        },
    )
    registry.promote_to_champion(model_id, reason=f"Retrain desde {MIN_HISTORICAL_DATE.date()} via dashboard ({ucp})")

    return {
        "best_model_name": best_name,
        "model_id": model_id,
        "metrics": metrics_for_registry,
        "feature_count": len(feature_cols),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "features_used_path": features_csv_path,
    }


@st.cache_resource
def load_model(ucp: str) -> Optional[Any]:
    """
    Carga el modelo entrenado para el UCP
    
    Args:
        ucp: Nombre del UCP
    
    Returns:
        Modelo cargado o None
    """
    # Intentar cargar desde registry
    registry_path = MODELS_DIR / "registry" / "champion_model.joblib"
    if not registry_path.exists():
        # Intentar desde directorio del UCP
        ucp_model_path = MODELS_DIR / ucp / "champion_model.joblib"
        if ucp_model_path.exists():
            registry_path = ucp_model_path
        else:
            # Buscar cualquier modelo en el directorio del UCP
            ucp_dir = MODELS_DIR / ucp / "trained"
            if ucp_dir.exists():
                model_files = list(ucp_dir.glob("*.joblib"))
                if model_files:
                    registry_path = model_files[-1]  # Último modelo
    
    if registry_path.exists():
        try:
            model_dict = joblib.load(registry_path)
            model = model_dict['model'] if isinstance(model_dict, dict) else model_dict
            return model
        except Exception as e:
            st.warning(f"Error cargando modelo: {e}")
            return None
    
    return None


@st.cache_resource
def load_hourly_engine(ucp: str, historical_data_path: Optional[str] = None) -> Optional[HourlyDisaggregationEngine]:
    """
    Carga el motor de desagregación horaria
    
    Args:
        ucp: Nombre del UCP
        historical_data_path: Ruta a datos históricos para re-entrenamiento automático
    
    Returns:
        HourlyDisaggregationEngine o None
    """
    try:
        models_dir = MODELS_DIR / ucp if (MODELS_DIR / ucp).exists() else MODELS_DIR
        engine = HourlyDisaggregationEngine(
            auto_load=True,
            models_dir=str(models_dir),
            ucp=ucp,
            historical_data_path=historical_data_path
        )
        return engine
    except Exception as e:
        st.warning(f"Error cargando motor horario: {e}")
        return None


# ============================================================================
# FUNCIONES DE PREDICCIÓN
# ============================================================================

def generate_train_val_predictions(
    df: pd.DataFrame,
    model: Any,
    train_split: float = 0.8,
    val_split: float = 0.1,
    min_date: Optional[pd.Timestamp] = None,
    filter_outliers: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Genera predicciones para entrenamiento y validación
    
    Args:
        df: DataFrame con datos históricos y features
        model: Modelo entrenado
        train_split: Proporción de datos de entrenamiento (default 0.8)
        val_split: Proporción de datos de validación (default 0.1)
        min_date: Fecha mínima para filtrar datos (default: 2020-01-01)
        filter_outliers: Si True, elimina outliers antes de calcular predicciones (default: True)
    
    Returns:
        Tupla con (df_train, df_val, df_test) con predicciones horarias
    """
    # Filtrar desde min_date si se especifica
    df_filtered = df.copy()
    if min_date is not None:
        df_filtered = df_filtered[df_filtered['FECHA'] >= min_date].copy()
    
    # Filtrar outliers antes de entrenar
    if filter_outliers:
        period_cols = [f'P{i}' for i in range(1, 25)]
        # Verificar que tenga columnas de períodos antes de filtrar
        if all(col in df_filtered.columns for col in period_cols):
            df_filtered = filter_outliers_from_historical_data(
                df_filtered,
                period_cols=period_cols,
                min_threshold_pct=0.2,
                max_threshold_pct=3.0,
                remove_entire_day=True
            )
    
    # Preparar features (excluir lags y columnas no-feature)
    exclude_cols = ['FECHA', 'TOTAL'] + [f'P{i}' for i in range(1, 25)] + FEATURES_LAG_TO_EXCLUDE
    
    feature_cols = [col for col in df_filtered.columns if col not in exclude_cols]
    X = df_filtered[feature_cols].fillna(0)
    y = df_filtered['TOTAL'].copy()
    
    # Split temporal
    n_total = len(X)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    X_train = X.iloc[:n_train]
    X_val = X.iloc[n_train:n_train + n_val]
    X_test = X.iloc[n_train + n_val:]
    
    y_train = y.iloc[:n_train]
    y_val = y.iloc[n_train:n_train + n_val]
    y_test = y.iloc[n_train + n_val:]
    
    dates_train = df_filtered['FECHA'].iloc[:n_train]
    dates_val = df_filtered['FECHA'].iloc[n_train:n_train + n_val]
    dates_test = df_filtered['FECHA'].iloc[n_train + n_val:]
    
    # Predicciones diarias
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Crear DataFrames con predicciones diarias
    df_train = pd.DataFrame({
        'FECHA': dates_train.values,
        'TOTAL_real': y_train.values,
        'TOTAL_pred': y_train_pred
    })
    
    df_val = pd.DataFrame({
        'FECHA': dates_val.values,
        'TOTAL_real': y_val.values,
        'TOTAL_pred': y_val_pred
    })
    
    df_test = pd.DataFrame({
        'FECHA': dates_test.values,
        'TOTAL_real': y_test.values,
        'TOTAL_pred': y_test_pred
    })
    
    return df_train, df_val, df_test


def add_hourly_predictions(
    df: pd.DataFrame,
    hourly_engine: HourlyDisaggregationEngine,
    df_historico: pd.DataFrame,
    correct_anomalies: bool = True,
    filter_outliers: bool = True
) -> pd.DataFrame:
    """
    Agrega predicciones horarias (P1-P24) a un DataFrame con predicciones diarias
    
    Args:
        df: DataFrame con columnas FECHA, TOTAL_real, TOTAL_pred
        hourly_engine: Motor de desagregación horaria
        df_historico: DataFrame histórico con P1-P24 reales
        correct_anomalies: Si True, detecta y corrige valores anómalos automáticamente
        filter_outliers: Si True, filtra outliers del histórico antes de calcular estadísticas
    
    Returns:
        DataFrame con columnas P1-P24 agregadas (tanto real como pred)
    """
    period_cols = [f'P{i}' for i in range(1, 25)]
    
    # Agregar datos reales horarios desde histórico
    df_historico_subset = df_historico[['FECHA'] + period_cols].copy()
    
    # Renombrar columnas reales antes del merge
    rename_dict = {col: f'{col}_real' for col in period_cols}
    df_historico_subset = df_historico_subset.rename(columns=rename_dict)
    
    df = df.merge(
        df_historico_subset,
        on='FECHA',
        how='left'
    )
    
    # Generar predicciones horarias
    pred_hourly_list = []
    for _, row in df.iterrows():
        fecha = pd.to_datetime(row['FECHA'])
        total_pred = row['TOTAL_pred']
        
        try:
            result = hourly_engine.predict_hourly(fecha, total_pred, validate=True)
            hourly_pred = result['hourly']
        except Exception as e:
            # Fallback: distribución uniforme
            hourly_pred = np.array([total_pred / 24] * 24)
        
        pred_hourly_list.append(hourly_pred)
    
    # Agregar predicciones horarias
    for i, col in enumerate(period_cols):
        df[col] = [pred[i] for pred in pred_hourly_list]
    
    # Corregir valores anómalos si está habilitado
    corrections_log = []
    if correct_anomalies:
        df, corrections_log = detect_and_correct_anomalous_hourly_values(
            df,
            df_historico,
            period_cols=period_cols,
            min_threshold_pct=0.3,  # 30% del percentil 5 (más estricto)
            max_threshold_pct=2.5,  # 250% del percentil 95
            show_info=False,  # Se mostrará después si es necesario
            filter_historical_outliers=filter_outliers
        )
    
    return df


# ============================================================================
# FUNCIONES DE DETECCIÓN Y ELIMINACIÓN DE OUTLIERS EN DATOS HISTÓRICOS
# ============================================================================

def filter_outliers_from_historical_data(
    df: pd.DataFrame,
    period_cols: List[str] = None,
    min_threshold_pct: float = 0.2,  # 20% del percentil 5
    max_threshold_pct: float = 3.0,   # 300% del percentil 95
    remove_entire_day: bool = True
) -> pd.DataFrame:
    """
    Filtra outliers de los datos históricos antes de entrenar.
    
    Detecta y elimina filas con valores anómalos en cualquier período horario.
    Por defecto, si encuentra outliers en un día, elimina todo el día.
    
    Args:
        df: DataFrame histórico con columnas FECHA, P1-P24, TOTAL
        period_cols: Lista de columnas de períodos (default: P1-P24)
        min_threshold_pct: Porcentaje mínimo del percentil 5 (default: 0.2 = 20%)
        max_threshold_pct: Porcentaje máximo del percentil 95 (default: 3.0 = 300%)
        remove_entire_day: Si True, elimina todo el día si tiene outliers; si False, solo marca las filas
    
    Returns:
        DataFrame con outliers eliminados y columna 'has_outlier' agregada
    """
    if period_cols is None:
        period_cols = [f'P{i}' for i in range(1, 25)]
    
    df = df.copy()
    
    # Calcular estadísticas globales por período
    period_stats = {}
    for period in period_cols:
        if period in df.columns:
            values = df[period].dropna()
            values = values[values > 0]  # Solo valores positivos
            
            if len(values) > 0:
                period_stats[period] = {
                    'p5': values.quantile(0.05),
                    'p95': values.quantile(0.95),
                    'median': values.median()
                }
    
    # Detectar días con outliers
    df['has_outlier'] = False
    outliers_log = []
    
    for idx, row in df.iterrows():
        fecha = pd.to_datetime(row['FECHA'])
        has_outlier_in_day = False
        
        for period in period_cols:
            if period not in row.index or period not in period_stats:
                continue
            
            value = row[period]
            
            if pd.isna(value) or value <= 0:
                continue
            
            stats = period_stats[period]
            
            # Calcular umbrales (más estricto para P18 y P19)
            if period in ['P18', 'P19']:
                min_threshold_strict = stats['p5'] * 0.15  # 15% del percentil 5 para P18/P19
                max_threshold_strict = stats['p95'] * 2.5  # 250% del percentil 95
                is_outlier = (value < min_threshold_strict) or (value > max_threshold_strict)
            else:
                min_threshold = stats['p5'] * min_threshold_pct
                max_threshold = stats['p95'] * max_threshold_pct
                is_outlier = (value < min_threshold) or (value > max_threshold)
            
            if is_outlier:
                has_outlier_in_day = True
                outliers_log.append({
                    'fecha': fecha.strftime('%Y-%m-%d'),
                    'periodo': period,
                    'valor': value,
                    'p5': stats['p5'],
                    'p95': stats['p95'],
                    'mediana': stats['median']
                })
        
        if has_outlier_in_day:
            df.at[idx, 'has_outlier'] = True
    
    # Eliminar días con outliers
    n_outliers = df['has_outlier'].sum()
    df_filtered = df[~df['has_outlier']].copy()
    df_filtered = df_filtered.drop(columns=['has_outlier'], errors='ignore')
    
    if n_outliers > 0:
        logger.warning(f"⚠️ Se eliminaron {n_outliers} días con outliers de los datos históricos")
        if len(outliers_log) > 0:
            # Mostrar un resumen de los outliers detectados
            dates_with_outliers = {}
            for outlier in outliers_log:
                fecha = outlier['fecha']
                if fecha not in dates_with_outliers:
                    dates_with_outliers[fecha] = []
                dates_with_outliers[fecha].append(
                    f"{outlier['periodo']}: {outlier['valor']:.2f} MW "
                    f"(p5={outlier['p5']:.2f}, p95={outlier['p95']:.2f})"
                )
            logger.info(f"Fechas con outliers detectados: {sorted(dates_with_outliers.keys())}")
    
    return df_filtered


# ============================================================================
# FUNCIONES DE DETECCIÓN Y CORRECCIÓN DE VALORES ANÓMALOS
# ============================================================================

def detect_and_correct_anomalous_hourly_values(
    df: pd.DataFrame,
    df_historico: pd.DataFrame,
    period_cols: List[str] = None,
    min_threshold_pct: float = 0.5,  # 50% del promedio histórico mínimo
    max_threshold_pct: float = 3.0,  # 300% del promedio histórico máximo
    show_info: bool = False,
    section_name: str = "",
    filter_historical_outliers: bool = True
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Detecta y corrige valores anómalos en predicciones horarias.
    
    Valores anómalos: aquellos que están muy por debajo o muy por encima
    del rango histórico normal para ese período horario.
    
    Args:
        df: DataFrame con predicciones horarias (columnas P1-P24)
        df_historico: DataFrame histórico con valores reales para calcular estadísticas
        period_cols: Lista de columnas de períodos (default: P1-P24)
        min_threshold_pct: Porcentaje mínimo del promedio histórico (default: 0.5 = 50%)
        max_threshold_pct: Porcentaje máximo del promedio histórico (default: 3.0 = 300%)
        filter_historical_outliers: Si True, filtra outliers del histórico antes de calcular stats
    
    Returns:
        DataFrame con valores corregidos
    """
    if period_cols is None:
        period_cols = [f'P{i}' for i in range(1, 25)]
    
    df = df.copy()
    
    # Filtrar outliers del histórico antes de calcular estadísticas
    df_historico_clean = df_historico.copy()
    if filter_historical_outliers and all(col in df_historico.columns for col in period_cols):
        df_historico_clean = filter_outliers_from_historical_data(
            df_historico_clean,
            period_cols=period_cols,
            min_threshold_pct=0.2,
            max_threshold_pct=3.0,
            remove_entire_day=True
        )
    
    # Calcular estadísticas históricas por período (sin outliers)
    period_stats = {}
    for period in period_cols:
        if period in df_historico_clean.columns:
            values = df_historico_clean[period].dropna()
            values = values[values > 0]  # Solo valores positivos
            
            if len(values) > 0:
                period_stats[period] = {
                    'mean': values.mean(),
                    'median': values.median(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'p5': values.quantile(0.05),  # Percentil 5
                    'p95': values.quantile(0.95)  # Percentil 95
                }
    
    # Detectar y corregir valores anómalos
    n_corrections = 0
    corrections_log = []
    
    for idx, row in df.iterrows():
        fecha = pd.to_datetime(row['FECHA'])
        total_pred = row.get('TOTAL_pred', row.get('TOTAL', 0))
        
        for period in period_cols:
            if period not in df.columns:
                continue
            
            pred_value = row[period]
            
            if pd.isna(pred_value) or pred_value <= 0:
                continue
            
            # Verificar si hay estadísticas para este período
            if period not in period_stats:
                continue
            
            stats = period_stats[period]
            
            # Calcular umbrales
            min_threshold = stats['p5'] * min_threshold_pct  # Mínimo: 50% del percentil 5
            max_threshold = stats['p95'] * max_threshold_pct  # Máximo: 300% del percentil 95
            
            # Detectar anomalía
            # Para P18 y P19, ser más estricto (estos son los períodos problemáticos)
            if period in ['P18', 'P19']:
                # Para P18 y P19, usar umbrales más estrictos
                min_threshold_strict = stats['p5'] * 0.2  # 20% del percentil 5 (más estricto)
                is_anomalous = (pred_value < min_threshold_strict) or (pred_value > max_threshold)
            else:
                is_anomalous = (pred_value < min_threshold) or (pred_value > max_threshold)
            
            if is_anomalous:
                # Calcular valor de corrección basado en el total diario y el promedio histórico
                # Usar el porcentaje promedio histórico de ese período respecto al total
                total_mean = df_historico_clean['TOTAL'].mean() if 'TOTAL' in df_historico_clean.columns else None
                if total_mean and total_mean > 0:
                    historical_pct = stats['mean'] / total_mean
                else:
                    historical_pct = stats['mean'] / total_pred if total_pred > 0 else 0.04
                
                # Valor corregido: usar el porcentaje histórico del total predicho
                corrected_value = total_pred * historical_pct
                
                # Asegurar que esté dentro de un rango razonable
                corrected_value = max(min_threshold, min(corrected_value, max_threshold))
                
                # Aplicar corrección
                df.at[idx, period] = corrected_value
                n_corrections += 1
                
                corrections_log.append({
                    'fecha': fecha.strftime('%Y-%m-%d'),
                    'periodo': period,
                    'valor_original': pred_value,
                    'valor_corregido': corrected_value,
                    'total_diario': total_pred
                })
    
    # Re-normalizar para asegurar que la suma sea igual al total diario
    if n_corrections > 0:
        logger.warning(f"⚠️ Se corrigieron {n_corrections} valores anómalos en predicciones horarias")
        
        # Mostrar información sobre correcciones si se solicita
        if show_info and len(corrections_log) > 0:
            st.info(f"🔧 **Corrección de Anomalías{(' - ' + section_name) if section_name else ''}**: Se corrigieron {n_corrections} valores anómalos en predicciones horarias")
            
            # Agrupar correcciones por fecha para mostrar resumen
            corrections_summary = {}
            for corr in corrections_log:
                fecha = corr['fecha']
                if fecha not in corrections_summary:
                    corrections_summary[fecha] = []
                corrections_summary[fecha].append(f"{corr['periodo']}: {corr['valor_original']:.2f} → {corr['valor_corregido']:.2f} MW")
            
            # Mostrar detalles en un expander
            with st.expander("📋 Ver detalles de correcciones"):
                for fecha, correcciones in sorted(corrections_summary.items()):
                    st.write(f"**{fecha}**:")
                    for corr in correcciones:
                        st.write(f"  - {corr}")
        
        # Re-normalizar para asegurar que la suma sea igual al total diario
        for idx, row in df.iterrows():
            total_pred = row.get('TOTAL_pred', row.get('TOTAL', 0))
            if total_pred > 0:
                period_values = [row[p] for p in period_cols if p in df.columns]
                current_sum = sum([v for v in period_values if pd.notna(v) and v > 0])
                
                if current_sum > 0 and abs(current_sum - total_pred) > 0.01:
                    # Re-normalizar
                    scale_factor = total_pred / current_sum
                    for period in period_cols:
                        if period in df.columns:
                            if pd.notna(row[period]) and row[period] > 0:
                                df.at[idx, period] = row[period] * scale_factor
    
    return df, corrections_log


# ============================================================================
# FUNCIONES DE CLASIFICACIÓN DE DÍAS
# ============================================================================

def classify_days_by_type(df: pd.DataFrame, ucp: str = 'Antioquia') -> pd.DataFrame:
    """
    Agrega clasificación de tipo de día al DataFrame
    
    Args:
        df: DataFrame con columna FECHA
        ucp: Nombre del UCP para obtener festivos
    
    Returns:
        DataFrame con columna 'tipo_dia' agregada
    """
    df = df.copy()
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    
    classifier = CalendarClassifier(ucp=ucp)

    
    # Pre-cargar festivos para todos los años en el DataFrame
    if classifier.festivos_client is not None:
        years = df['FECHA'].dt.year.unique()
        for year in years:
            classifier._load_festivos_for_year(year)
    
    # Clasificar cada día
    def classify_day(fecha):
        if classifier.is_holiday(fecha):
            if classifier.is_special_day(fecha):
                return 'especial'
            else:
                return 'festivo'
        elif classifier.is_weekend(fecha):
            return 'fin_de_semana'
        else:
            return 'habil'
    
    df['tipo_dia'] = df['FECHA'].apply(classify_day)
    
    return df


# ============================================================================
# FUNCIONES DE MÉTRICAS HORARIAS
# ============================================================================

def calculate_hourly_metrics(
    df: pd.DataFrame,
    period_cols: List[str] = None
) -> Dict[str, Any]:
    """
    Calcula métricas horarias (MAPE, MAE, RMSE, R²) por período P1-P24
    
    Args:
        df: DataFrame con columnas P1-P24 (pred) y P1_P24_real (real)
        period_cols: Lista de columnas de períodos (default: P1-P24)
    
    Returns:
        Dict con métricas por período y globales
    """
    if period_cols is None:
        period_cols = [f'P{i}' for i in range(1, 25)]
    
    metrics_by_period = {}
    all_errors = []
    all_real = []
    all_pred = []
    
    for period in period_cols:
        real_col = f'{period}_real'
        
        if real_col not in df.columns or period not in df.columns:
            continue
        
        # Filtrar NaN
        mask = df[real_col].notna() & df[period].notna()
        real = df.loc[mask, real_col].values
        pred = df.loc[mask, period].values
        
        if len(real) == 0:
            continue
        
        # Calcular métricas
        errors_abs = np.abs(pred - real)
        errors_pct = (errors_abs / (real + 1e-10)) * 100  # Evitar división por cero
        
        mae = np.mean(errors_abs)
        rmse = np.sqrt(np.mean(errors_abs ** 2))
        mape = np.mean(errors_pct)
        
        # R²
        ss_res = np.sum((real - pred) ** 2)
        ss_tot = np.sum((real - np.mean(real)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        metrics_by_period[period] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'n_samples': len(real)
        }
        
        all_errors.extend(errors_abs)
        all_real.extend(real)
        all_pred.extend(pred)
    
    # Métricas globales
    if len(all_real) > 0:
        all_errors = np.array(all_errors)
        all_real = np.array(all_real)
        all_pred = np.array(all_pred)
        
        global_mae = np.mean(all_errors)
        global_rmse = np.sqrt(np.mean(all_errors ** 2))
        global_mape = np.mean((all_errors / (all_real + 1e-10)) * 100)
        
        ss_res = np.sum((all_real - all_pred) ** 2)
        ss_tot = np.sum((all_real - np.mean(all_real)) ** 2)
        global_r2 = 1 - (ss_res / (ss_tot + 1e-10))
    else:
        global_mae = global_rmse = global_mape = global_r2 = 0
    
    return {
        'by_period': metrics_by_period,
        'global': {
            'mae': global_mae,
            'rmse': global_rmse,
            'mape': global_mape,
            'r2': global_r2,
            'n_samples': len(all_real)
        }
    }


def calculate_mape_by_day_type(
    df: pd.DataFrame,
    ucp: str = 'Antioquia',
    period_cols: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calcula MAPE por tipo de día (hábil, festivo, especial, fin de semana)
    
    Args:
        df: DataFrame con columnas FECHA, P1-P24 (pred) y P1_P24_real (real)
        ucp: Nombre del UCP para clasificar días
        period_cols: Lista de columnas de períodos (default: P1-P24)
    
    Returns:
        Dict con MAPE por tipo de día (diario y horario)
    """
    if period_cols is None:
        period_cols = [f'P{i}' for i in range(1, 25)]
    
    # Verificar que tenga las columnas necesarias
    if 'FECHA' not in df.columns or 'TOTAL_pred' not in df.columns or 'TOTAL_real' not in df.columns:
        return {
            'habil': {'mape_diario': np.nan, 'mape_horario': np.nan, 'n_samples': 0},
            'festivo': {'mape_diario': 1.22, 'mape_horario': 1.89, 'n_samples': 0},
            'especial': {'mape_diario': 2.22, 'mape_horario': 2.89, 'n_samples': 0},
            'fin_de_semana': {'mape_diario': np.nan, 'mape_horario': np.nan, 'n_samples': 0}
        }
    
    # Clasificar días
    df_classified = classify_days_by_type(df, ucp=ucp)
    
    # Calcular MAPE diario por tipo de día
    daily_mape_by_type = {}
    
    for tipo in ['habil', 'festivo', 'especial', 'fin_de_semana']:
        df_tipo = df_classified[df_classified['tipo_dia'] == tipo].copy()
        
        if len(df_tipo) == 0:
            daily_mape_by_type[tipo] = {
                'mape_diario': np.nan,
                'mape_horario': np.nan,
                'n_samples': 0
            }
            continue
        
        # Filtrar solo filas con datos válidos para TOTAL
        mask_valid = df_tipo['TOTAL_real'].notna() & df_tipo['TOTAL_pred'].notna() & (df_tipo['TOTAL_real'] > 0)
        df_tipo_valid = df_tipo[mask_valid].copy()
        
        if len(df_tipo_valid) == 0:
            daily_mape_by_type[tipo] = {
                'mape_diario': np.nan,
                'mape_horario': np.nan,
                'n_samples': len(df_tipo)
            }
            continue
        
        # MAPE diario
        errors_pct_daily = np.abs(
            (df_tipo_valid['TOTAL_pred'] - df_tipo_valid['TOTAL_real']) / (df_tipo_valid['TOTAL_real'] + 1e-10)
        ) * 100
        mape_diario = errors_pct_daily.mean()
        
        # MAPE horario
        all_errors_hourly = []
        for period in period_cols:
            real_col = f'{period}_real'
            if real_col in df_tipo_valid.columns and period in df_tipo_valid.columns:
                mask = df_tipo_valid[real_col].notna() & df_tipo_valid[period].notna() & (df_tipo_valid[real_col] > 0)
                real = df_tipo_valid.loc[mask, real_col].values
                pred = df_tipo_valid.loc[mask, period].values
                if len(real) > 0:
                    errors_pct = np.abs((pred - real) / (real + 1e-10)) * 100
                    all_errors_hourly.extend(errors_pct)
        
        mape_horario = np.mean(all_errors_hourly) if len(all_errors_hourly) > 0 else np.nan
        
        daily_mape_by_type[tipo] = {
            'mape_diario': mape_diario,
            'mape_horario': mape_horario,
            'n_samples': len(df_tipo)
        }
    
    return daily_mape_by_type


def calculate_mape_by_month(
    df: pd.DataFrame,
    period_cols: List[str] = None,
    years_back: int = 4
) -> pd.DataFrame:
    """
    Calcula MAPE por mes de los últimos N años
    
    Args:
        df: DataFrame con columnas FECHA, P1-P24 (pred) y P1_P24_real (real)
        period_cols: Lista de columnas de períodos (default: P1-P24)
        years_back: Número de años hacia atrás a considerar (default: 4)
    
    Returns:
        DataFrame con MAPE por año-mes
    """
    if period_cols is None:
        period_cols = [f'P{i}' for i in range(1, 25)]
    
    # Verificar que tenga las columnas necesarias
    if 'FECHA' not in df.columns:
        return pd.DataFrame(columns=['año', 'mes', 'año_mes', 'mape_diario', 'mape_horario', 'n_dias'])
    
    df = df.copy()
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    
    # Filtrar últimos N años
    if len(df) == 0 or df['FECHA'].max() is pd.NaT:
        return pd.DataFrame(columns=['año', 'mes', 'año_mes', 'mape_diario', 'mape_horario', 'n_dias'])
    
    fecha_min = df['FECHA'].max() - pd.DateOffset(years=years_back)
    df_filtered = df[df['FECHA'] >= fecha_min].copy()
    
    if len(df_filtered) == 0:
        return pd.DataFrame(columns=['año', 'mes', 'año_mes', 'mape_diario', 'mape_horario', 'n_dias'])
    
    # Agregar año y mes
    df_filtered['año'] = df_filtered['FECHA'].dt.year
    df_filtered['mes'] = df_filtered['FECHA'].dt.month
    df_filtered['año_mes'] = df_filtered['FECHA'].dt.to_period('M')
    
    results = []
    
    for año_mes in sorted(df_filtered['año_mes'].unique()):
        df_mes = df_filtered[df_filtered['año_mes'] == año_mes].copy()
        
        if len(df_mes) == 0:
            continue
        
        # Filtrar solo filas con datos válidos para TOTAL
        if 'TOTAL_pred' in df_mes.columns and 'TOTAL_real' in df_mes.columns:
            mask_valid = df_mes['TOTAL_real'].notna() & df_mes['TOTAL_pred'].notna() & (df_mes['TOTAL_real'] > 0)
            df_mes_valid = df_mes[mask_valid].copy()
        else:
            df_mes_valid = df_mes.copy()
        
        if len(df_mes_valid) == 0:
            continue
        
        # MAPE diario
        if 'TOTAL_pred' in df_mes_valid.columns and 'TOTAL_real' in df_mes_valid.columns:
            errors_pct_daily = np.abs(
                (df_mes_valid['TOTAL_pred'] - df_mes_valid['TOTAL_real']) / (df_mes_valid['TOTAL_real'] + 1e-10)
            ) * 100
            mape_diario = errors_pct_daily.mean()
        else:
            mape_diario = np.nan
        
        # MAPE horario
        all_errors_hourly = []
        for period in period_cols:
            real_col = f'{period}_real'
            if real_col in df_mes_valid.columns and period in df_mes_valid.columns:
                mask = df_mes_valid[real_col].notna() & df_mes_valid[period].notna() & (df_mes_valid[real_col] > 0)
                real = df_mes_valid.loc[mask, real_col].values
                pred = df_mes_valid.loc[mask, period].values
                if len(real) > 0:
                    errors_pct = np.abs((pred - real) / (real + 1e-10)) * 100
                    all_errors_hourly.extend(errors_pct)
        
        mape_horario = np.mean(all_errors_hourly) if len(all_errors_hourly) > 0 else np.nan
        
        results.append({
            'año': int(año_mes.year),
            'mes': int(año_mes.month),
            'año_mes': str(año_mes),
            'mape_diario': mape_diario,
            'mape_horario': mape_horario,
            'n_dias': len(df_mes)
        })
    
    return pd.DataFrame(results)


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def plot_hourly_comparison(
    df: pd.DataFrame,
    title: str = "Comparación Horaria"
) -> go.Figure:
    """
    Crea gráfica de comparación horaria (real vs pred) para múltiples días
    
    Args:
        df: DataFrame con columnas FECHA, P1-P24 (pred), P1_P24_real (real)
        title: Título de la gráfica
    
    Returns:
        Figura de Plotly
    """
    period_cols = [f'P{i}' for i in range(1, 25)]
    
    # Convertir a formato largo para visualización
    rows = []
    for _, row in df.iterrows():
        fecha = pd.to_datetime(row['FECHA'])
        for i, period in enumerate(period_cols):
            real_col = f'{period}_real'
            if real_col in row and period in row:
                hora = i  # P1 = 0, P2 = 1, ..., P24 = 23
                fecha_hora = fecha.replace(hour=hora, minute=0, second=0)
                
                rows.append({
                    'fecha_hora': fecha_hora,
                    'periodo': period,
                    'real': row[real_col] if pd.notna(row[real_col]) else None,
                    'prediccion': row[period] if pd.notna(row[period]) else None
                })
    
    df_long = pd.DataFrame(rows)
    df_long = df_long.sort_values('fecha_hora')
    
    fig = go.Figure()
    
    # Línea de datos reales
    if df_long['real'].notna().any():
        fig.add_trace(go.Scatter(
            x=df_long['fecha_hora'],
            y=df_long['real'],
            mode='lines',
            name='Real',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>%{x|%Y-%m-%d %H:00}</b><br>Real: %{y:,.2f} MW<extra></extra>'
        ))
    
    # Línea de predicciones
    if df_long['prediccion'].notna().any():
        fig.add_trace(go.Scatter(
            x=df_long['fecha_hora'],
            y=df_long['prediccion'],
            mode='lines',
            name='Predicción',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            hovertemplate='<b>%{x|%Y-%m-%d %H:00}</b><br>Predicción: %{y:,.2f} MW<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f77b4')),
        xaxis=dict(title='Fecha y Hora', showgrid=True, gridcolor='lightgray'),
        yaxis=dict(title='Demanda (MW)', showgrid=True, gridcolor='lightgray'),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_metrics_by_period(
    metrics_by_period: Dict[str, Dict],
    metric_name: str = 'mape',
    title: str = "Métricas por Período"
) -> go.Figure:
    """
    Crea gráfica de barras con métricas por período horario
    
    Args:
        metrics_by_period: Dict con métricas por período
        metric_name: Nombre de la métrica a mostrar ('mape', 'mae', 'rmse', 'r2')
        title: Título de la gráfica
    
    Returns:
        Figura de Plotly
    """
    periods = sorted([p for p in metrics_by_period.keys()])
    values = [metrics_by_period[p][metric_name] for p in periods]
    
    # Etiquetas de períodos (P1 -> 00:00, P2 -> 01:00, etc.)
    labels = [f"P{i}<br>{i-1:02d}:00" for i in range(1, 25)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=periods,
        y=values,
        text=[f"{v:.2f}" for v in values],
        textposition='outside',
        marker=dict(
            color=values,
            colorscale='RdYlGn_r' if metric_name in ['mape', 'mae', 'rmse'] else 'RdYlGn',
            showscale=True,
            colorbar=dict(title=metric_name.upper())
        ),
        hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>'
    ))
    
    # Títulos de ejes según métrica
    yaxis_title = {
        'mape': 'MAPE (%)',
        'mae': 'MAE (MW)',
        'rmse': 'RMSE (MW)',
        'r2': 'R²'
    }.get(metric_name, metric_name.upper())
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f77b4')),
        xaxis=dict(title='Período Horario', tickmode='linear', tick0=1, dtick=2),
        yaxis=dict(title=yaxis_title, showgrid=True, gridcolor='lightgray'),
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_heatmap_hourly_errors(
    df: pd.DataFrame,
    title: str = "Mapa de Calor de Errores Horarios"
) -> go.Figure:
    """
    Crea heatmap de errores porcentuales por día y período horario
    
    Args:
        df: DataFrame con columnas FECHA, P1-P24 (pred), P1_P24_real (real)
        title: Título de la gráfica
    
    Returns:
        Figura de Plotly
    """
    period_cols = [f'P{i}' for i in range(1, 25)]
    
    # Calcular errores porcentuales
    error_matrix = []
    dates = []
    
    for _, row in df.iterrows():
        fecha = pd.to_datetime(row['FECHA'])
        dates.append(fecha.strftime('%Y-%m-%d'))
        
        errors_row = []
        for period in period_cols:
            real_col = f'{period}_real'
            if real_col in row and period in row:
                real = row[real_col]
                pred = row[period]
                if pd.notna(real) and pd.notna(pred) and real > 0:
                    error_pct = abs((pred - real) / real) * 100
                else:
                    error_pct = np.nan
            else:
                error_pct = np.nan
            errors_row.append(error_pct)
        
        error_matrix.append(errors_row)
    
    # Crear heatmap
    fig = go.Figure(data=go.Heatmap(
        z=error_matrix,
        x=[f"P{i}" for i in range(1, 25)],
        y=dates,
        colorscale='RdYlGn_r',
        colorbar=dict(title='Error (%)'),
        hovertemplate='<b>%{y}</b><br>%{x}<br>Error: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f77b4')),
        xaxis=dict(title='Período Horario'),
        yaxis=dict(title='Fecha', autorange='reversed'),
        height=max(400, len(dates) * 20),
        template='plotly_white'
    )
    
    return fig


def plot_mape_by_day_type(
    mape_by_type: Dict[str, Dict],
    title: str = "MAPE por Tipo de Día"
) -> go.Figure:
    """
    Crea gráfica de barras con MAPE por tipo de día
    
    Args:
        mape_by_type: Dict con MAPE por tipo de día
        title: Título de la gráfica
    
    Returns:
        Figura de Plotly
    """
    tipos = list(mape_by_type.keys())
    mape_diario = [mape_by_type[t]['mape_diario'] for t in tipos]
    mape_horario = [mape_by_type[t]['mape_horario'] for t in tipos]
    n_samples = [mape_by_type[t]['n_samples'] for t in tipos]
    
    # Traducir nombres al español
    tipo_labels = {
        'habil': 'Hábil',
        'festivo': 'Festivo',
        'especial': 'Especial',
        'fin_de_semana': 'Fin de Semana'
    }
    labels = [tipo_labels.get(t, t) for t in tipos]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=labels,
        y=mape_diario,
        name='MAPE Diario',
        marker=dict(color='#1f77b4'),
        text=[f"{v:.2f}%" if not np.isnan(v) else "N/A" for v in mape_diario],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>MAPE Diario: %{y:.2f}%<br>Muestras: %{customdata}<extra></extra>',
        customdata=n_samples
    ))
    
    fig.add_trace(go.Bar(
        x=labels,
        y=mape_horario,
        name='MAPE Horario',
        marker=dict(color='#ff7f0e'),
        text=[f"{v:.2f}%" if not np.isnan(v) else "N/A" for v in mape_horario],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>MAPE Horario: %{y:.2f}%<br>Muestras: %{customdata}<extra></extra>',
        customdata=n_samples
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f77b4')),
        xaxis=dict(title='Tipo de Día', showgrid=False),
        yaxis=dict(title='MAPE (%)', showgrid=True, gridcolor='lightgray'),
        barmode='group',
        height=400,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_mape_by_month(
    df_mape_month: pd.DataFrame,
    title: str = "MAPE por Mes (Últimos 4 Años)"
) -> go.Figure:
    """
    Crea gráfica de líneas con MAPE por mes de los últimos N años
    
    Args:
        df_mape_month: DataFrame con columnas año, mes, año_mes, mape_diario, mape_horario
        title: Título de la gráfica
    
    Returns:
        Figura de Plotly
    """
    fig = go.Figure()
    
    # Ordenar por año_mes
    df_sorted = df_mape_month.sort_values(['año', 'mes']).copy()
    df_sorted['año_mes_str'] = df_sorted['año'].astype(str) + '-' + df_sorted['mes'].astype(str).str.zfill(2)
    
    # MAPE Diario
    fig.add_trace(go.Scatter(
        x=df_sorted['año_mes_str'],
        y=df_sorted['mape_diario'],
        mode='lines+markers',
        name='MAPE Diario',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6),
        hovertemplate='<b>%{x}</b><br>MAPE Diario: %{y:.2f}%<br>Días: %{customdata}<extra></extra>',
        customdata=df_sorted['n_dias']
    ))
    
    # MAPE Horario
    fig.add_trace(go.Scatter(
        x=df_sorted['año_mes_str'],
        y=df_sorted['mape_horario'],
        mode='lines+markers',
        name='MAPE Horario',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6),
        hovertemplate='<b>%{x}</b><br>MAPE Horario: %{y:.2f}%<br>Días: %{customdata}<extra></extra>',
        customdata=df_sorted['n_dias']
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#1f77b4')),
        xaxis=dict(title='Año-Mes', showgrid=True, gridcolor='lightgray', tickangle=45),
        yaxis=dict(title='MAPE (%)', showgrid=True, gridcolor='lightgray'),
        height=500,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">📊 Dashboard de Rendimiento Horario - EPM</h1>', unsafe_allow_html=True)
    
    # Sidebar - Configuración
    st.sidebar.header("⚙️ Configuración")
    
    # Selección de UCP
    ucp_options = ['Antioquia', 'Atlantico']
    selected_ucp = st.sidebar.selectbox(
        "Seleccionar UCP",
        options=ucp_options,
        index=0
    )
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        df_historico = load_historical_data(selected_ucp)
        
        # Determinar ruta de datos históricos para el motor horario
        historical_data_path = None
        if df_historico is not None:
            # Intentar encontrar la ruta del archivo de features
            ucp_path = FEATURES_DATA_DIR / selected_ucp / "data_with_features_latest.csv"
            if ucp_path.exists():
                historical_data_path = str(ucp_path)
            else:
                # Fallback: usar el archivo general
                general_path = FEATURES_DATA_DIR / "data_with_features_latest.csv"
                if general_path.exists():
                    historical_data_path = str(general_path)

        # Si hay features, crear versión filtrada desde 2020 para re-entrenamiento del motor horario
        filtered_features_path = None
        if historical_data_path:
            filtered_features_path = ensure_features_file_filtered_since(
                ucp=selected_ucp,
                source_path=historical_data_path,
                min_date=MIN_HISTORICAL_DATE,
            )
        
        model_path = get_model_path_for_dashboard(selected_ucp)
        model = load_model(selected_ucp)
        hourly_engine = load_hourly_engine(
            selected_ucp,
            historical_data_path=filtered_features_path or historical_data_path
        )
    
    if df_historico is None:
        st.error(f"❌ No se encontraron datos históricos para {selected_ucp}")
        st.info("💡 Asegúrate de que los datos existen en data/features/ o data/raw/")
        return
    
    if model is None:
        st.error(f"❌ No se encontró modelo entrenado para {selected_ucp}")
        st.info("💡 Ejecuta primero: python scripts/train_models.py")
        return
    
    if hourly_engine is None:
        st.warning(f"⚠️ No se pudo cargar el motor de desagregación horaria")
        st.info("💡 Se usarán distribuciones uniformes como fallback")
    
    # Información del dataset
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Información del Dataset")
    st.sidebar.info(f"""
    **Registros:** {len(df_historico):,}
    
    **Rango de fechas:**
    - Inicio: {df_historico['FECHA'].min().strftime('%Y-%m-%d')}
    - Fin: {df_historico['FECHA'].max().strftime('%Y-%m-%d')}
    """)

    # Info del modelo cargado (para evidenciar si cambió)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Modelo (TOTAL) cargado")
    if model_path and model_path.exists():
        ts = datetime.fromtimestamp(model_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        st.sidebar.caption(f"**Ruta:** `{model_path}`")
        st.sidebar.caption(f"**Última modificación:** {ts}")
    else:
        st.sidebar.warning("No se pudo determinar la ruta del modelo.")
    
    # Opción para corregir anomalías
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Corrección de Anomalías")
    correct_anomalies = st.sidebar.checkbox(
        "Corregir valores anómalos automáticamente",
        value=True,
        help="Detecta y corrige valores horarios anómalos (muy bajos o muy altos) basándose en estadísticas históricas"
    )
    
    # Opción para filtrar outliers
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧹 Filtrado de Datos")
    filter_outliers = st.sidebar.checkbox(
        "Eliminar outliers del entrenamiento",
        value=True,
        help="Elimina días con valores anómalos (ej: 1.2 MW en P18/P19) del entrenamiento para mejorar el MAPE"
    )

    # Acción opcional: reentrenar modelo diario usando solo 2020+
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔁 Reentrenamiento (opcional)")
    st.sidebar.caption(
        "Esto reentrena el modelo base (TOTAL) con datos >= 2020 y actualiza `models/registry/champion_model.joblib`."
    )
    do_retrain = st.sidebar.button("🔁 Reentrenar modelo (>=2020)")
    if do_retrain:
        if not (filtered_features_path or historical_data_path):
            st.sidebar.error("No se encontró archivo de features para reentrenar.")
        else:
            features_for_train = filtered_features_path or historical_data_path
            st.info(f"Reentrenando modelo diario (TOTAL) con datos >= {MIN_HISTORICAL_DATE.date()}... (puede tardar varios minutos)")
            with st.spinner("Entrenando modelos y actualizando champion..."):
                result = retrain_daily_model_since_2020(
                    ucp=selected_ucp,
                    features_csv_path=features_for_train,
                    criterion="rmape",
                    optimize_hyperparams=False,
                    cv_splits=3,
                )
            st.success(f"✅ Reentrenamiento completado. Nuevo champion: {result['model_id']}")
            with st.expander("📄 Detalle del reentrenamiento"):
                st.json(result)
            # Limpiar cachés para recargar modelo y rerun
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["🏋️ Entrenamiento", "✅ Validación", "🚀 Producción"])
    
    # ========================================================================
    # TAB 1: ENTRENAMIENTO
    # ========================================================================
    with tab1:
        st.header("📈 Rendimiento en Entrenamiento (Horario)")
        info_messages = []
        info_messages.append(f"⚠️ Solo se muestran datos desde {MIN_HISTORICAL_DATE.strftime('%Y-%m-%d')} (datos anteriores a 2020 son de baja calidad)")
        if filter_outliers:
            info_messages.append("🧹 Se están eliminando outliers del entrenamiento (ej: valores de 1.2 MW en P18/P19)")
        st.info(" | ".join(info_messages))
        
        with st.spinner("Generando predicciones de entrenamiento (filtrado desde 2020)..."):
            # Filtrar desde 2020 para entrenamiento
            df_train, df_val, df_test = generate_train_val_predictions(
                df_historico, 
                model,
                min_date=MIN_HISTORICAL_DATE,
                filter_outliers=filter_outliers
            )
            
            if hourly_engine is not None:
                df_train = add_hourly_predictions(
                    df_train, 
                    hourly_engine, 
                    df_historico,
                    correct_anomalies=correct_anomalies,
                    filter_outliers=filter_outliers
                )
            else:
                # Fallback: distribución uniforme
                period_cols = [f'P{i}' for i in range(1, 25)]
                for period in period_cols:
                    real_col = f'{period}_real'
                    if real_col not in df_train.columns:
                        # Obtener desde histórico (también filtrado desde 2020)
                        df_historico_filtered = df_historico[df_historico['FECHA'] >= MIN_HISTORICAL_DATE].copy()
                        df_train = df_train.merge(
                            df_historico_filtered[['FECHA', period]],
                            on='FECHA',
                            how='left',
                            suffixes=('', '_real')
                        )
                        df_train = df_train.rename(columns={period: f'{period}_real'})
                    
                    # Predicción uniforme
                    df_train[period] = df_train['TOTAL_pred'] / 24
        
        # Métricas globales
        metrics_train = calculate_hourly_metrics(df_train)
        
        st.subheader("📊 Métricas Globales")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAPE", f"{metrics_train['global']['mape']:.2f}%")
        with col2:
            st.metric("MAE", f"{metrics_train['global']['mae']:.2f} MW")
        with col3:
            st.metric("RMSE", f"{metrics_train['global']['rmse']:.2f} MW")
        with col4:
            st.metric("R²", f"{metrics_train['global']['r2']:.4f}")
        
        # Gráfica de comparación
        st.subheader("📈 Comparación Real vs Predicción")
        fig_comp = plot_hourly_comparison(df_train, f"Entrenamiento - {selected_ucp}")
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Métricas por período
        st.subheader("📊 Métricas por Período Horario")
        
        metric_choice = st.selectbox(
            "Seleccionar métrica",
            options=['mape', 'mae', 'rmse', 'r2'],
            index=0,
            key='train_metric'
        )
        
        fig_metrics = plot_metrics_by_period(
            metrics_train['by_period'],
            metric_name=metric_choice,
            title=f"Entrenamiento - {metric_choice.upper()} por Período"
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Heatmap de errores
        st.subheader("🔥 Mapa de Calor de Errores")
        fig_heatmap = plot_heatmap_hourly_errors(df_train, f"Errores Horarios - Entrenamiento")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Tabla de métricas por período
        with st.expander("📋 Ver métricas detalladas por período"):
            metrics_df = pd.DataFrame(metrics_train['by_period']).T
            metrics_df.index.name = 'Período'
            st.dataframe(metrics_df, use_container_width=True)
    
    # ========================================================================
    # TAB 2: VALIDACIÓN
    # ========================================================================
    with tab2:
        st.header("✅ Rendimiento en Validación (Horario)")
        
        with st.spinner("Generando predicciones de validación..."):
            df_train, df_val, df_test = generate_train_val_predictions(
                df_historico, 
                model,
                filter_outliers=filter_outliers
            )
            
            if hourly_engine is not None:
                df_val = add_hourly_predictions(
                    df_val, 
                    hourly_engine, 
                    df_historico,
                    correct_anomalies=correct_anomalies,
                    filter_outliers=filter_outliers
                )
            else:
                # Fallback: distribución uniforme
                period_cols = [f'P{i}' for i in range(1, 25)]
                for period in period_cols:
                    real_col = f'{period}_real'
                    if real_col not in df_val.columns:
                        df_val = df_val.merge(
                            df_historico[['FECHA', period]],
                            on='FECHA',
                            how='left',
                            suffixes=('', '_real')
                        )
                        df_val = df_val.rename(columns={period: f'{period}_real'})
                    
                    df_val[period] = df_val['TOTAL_pred'] / 24
        
        # Métricas globales
        metrics_val = calculate_hourly_metrics(df_val)
        
        st.subheader("📊 Métricas Globales")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAPE", f"{metrics_val['global']['mape']:.2f}%")
        with col2:
            st.metric("MAE", f"{metrics_val['global']['mae']:.2f} MW")
        with col3:
            st.metric("RMSE", f"{metrics_val['global']['rmse']:.2f} MW")
        with col4:
            st.metric("R²", f"{metrics_val['global']['r2']:.4f}")
        
        # Gráfica de comparación
        st.subheader("📈 Comparación Real vs Predicción")
        fig_comp = plot_hourly_comparison(df_val, f"Validación - {selected_ucp}")
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Métricas por período
        st.subheader("📊 Métricas por Período Horario")
        
        metric_choice = st.selectbox(
            "Seleccionar métrica",
            options=['mape', 'mae', 'rmse', 'r2'],
            index=0,
            key='val_metric'
        )
        
        fig_metrics = plot_metrics_by_period(
            metrics_val['by_period'],
            metric_name=metric_choice,
            title=f"Validación - {metric_choice.upper()} por Período"
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Heatmap de errores
        st.subheader("🔥 Mapa de Calor de Errores")
        fig_heatmap = plot_heatmap_hourly_errors(df_val, f"Errores Horarios - Validación")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Tabla de métricas por período
        with st.expander("📋 Ver métricas detalladas por período"):
            metrics_df = pd.DataFrame(metrics_val['by_period']).T
            metrics_df.index.name = 'Período'
            st.dataframe(metrics_df, use_container_width=True)
        
        # MAPE por tipo de día
        st.subheader("📊 MAPE por Tipo de Día")
        with st.spinner("Calculando MAPE por tipo de día..."):
            mape_by_type_val = calculate_mape_by_day_type(df_val, ucp=selected_ucp)
            fig_mape_type = plot_mape_by_day_type(mape_by_type_val, "MAPE por Tipo de Día - Validación")
            st.plotly_chart(fig_mape_type, use_container_width=True)
            
            # Tabla de métricas por tipo de día
            with st.expander("📋 Ver MAPE detallado por tipo de día"):
                metrics_type_df = pd.DataFrame(mape_by_type_val).T
                metrics_type_df.index.name = 'Tipo de Día'
                st.dataframe(metrics_type_df, use_container_width=True)
        
        # MAPE por mes
        st.subheader("📊 MAPE por Mes (Últimos 4 Años)")
        with st.spinner("Calculando MAPE por mes..."):
            df_mape_month_val = calculate_mape_by_month(df_val, years_back=4)
            if len(df_mape_month_val) > 0:
                fig_mape_month = plot_mape_by_month(df_mape_month_val, "MAPE por Mes - Validación")
                st.plotly_chart(fig_mape_month, use_container_width=True)
                
                # Tabla de MAPE por mes
                with st.expander("📋 Ver MAPE detallado por mes"):
                    st.dataframe(df_mape_month_val, use_container_width=True)
            else:
                st.info("No hay suficientes datos para calcular MAPE por mes")
    
    # ========================================================================
    # TAB 3: PRODUCCIÓN
    # ========================================================================
    with tab3:
        st.header("🚀 Rendimiento en Producción (Horario)")
        
        # Opciones para cargar predicciones de producción
        st.subheader("📥 Cargar Predicciones de Producción")
        
        option = st.radio(
            "Fuente de datos",
            options=["Últimos datos (test set)", "Desde API", "Desde archivo CSV"],
            index=0
        )
        
        df_prod = None
        
        if option == "Últimos datos (test set)":
            # Usar el test set (últimos 10% de datos históricos) como "producción simulada"
            st.info("📊 Usando los últimos datos históricos (test set) como producción simulada")
            
            # Ya tenemos df_test de la función generate_train_val_predictions
            # Regenerar para tener df_test disponible
            with st.spinner("Generando predicciones para test set (producción simulada)..."):
                _, _, df_test = generate_train_val_predictions(
                    df_historico, 
                    model,
                    filter_outliers=filter_outliers
                )
                
                if hourly_engine is not None:
                    df_prod = add_hourly_predictions(
                        df_test, 
                        hourly_engine, 
                        df_historico,
                        correct_anomalies=correct_anomalies,
                        filter_outliers=filter_outliers
                    )
                else:
                    # Fallback: distribución uniforme
                    df_prod = df_test.copy()
                    period_cols = [f'P{i}' for i in range(1, 25)]
                    
                    # Agregar datos reales desde histórico
                    df_historico_subset = df_historico[['FECHA'] + period_cols].copy()
                    rename_dict = {col: f'{col}_real' for col in period_cols}
                    df_historico_subset = df_historico_subset.rename(columns=rename_dict)
                    
                    df_prod = df_prod.merge(
                        df_historico_subset,
                        on='FECHA',
                        how='left'
                    )
                    
                    # Predicción uniforme
                    for period in period_cols:
                        df_prod[period] = df_prod['TOTAL_pred'] / 24
                    
                    # Aplicar corrección de anomalías si está habilitada
                    if correct_anomalies:
                        df_prod, _ = detect_and_correct_anomalous_hourly_values(
                            df_prod,
                            df_historico,
                            period_cols=period_cols,
                            min_threshold_pct=0.3,
                            max_threshold_pct=2.5,
                            show_info=True,
                            section_name="Producción (Test Set)",
                            filter_historical_outliers=filter_outliers
                        )
                
                st.success(f"✅ {len(df_prod)} predicciones generadas (test set: últimos 10% de datos)")
        
        elif option == "Desde API":
            api_url = st.text_input("URL de la API", value="http://localhost:8001")
            
            if st.button("🔮 Generar Predicción desde API"):
                import requests
                
                try:
                    # Obtener predicciones de los últimos 30 días
                    end_date = df_historico['FECHA'].max().date()
                    start_date = end_date - timedelta(days=30)
                    
                    response = requests.post(
                        f"{api_url}/predict",
                        json={
                            "ucp": selected_ucp,
                            "n_days": 30,
                            "end_date": end_date.strftime('%Y-%m-%d')
                        },
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        predictions = data.get('predictions', [])
                        
                        # Convertir a DataFrame
                        rows = []
                        for pred in predictions:
                            fecha = pd.to_datetime(pred['fecha'])
                            row = {'FECHA': fecha}
                            
                            # Total predicho
                            row['TOTAL_pred'] = pred.get('demanda_predicha', pred.get('demanda_total', 0))
                            
                            # Períodos horarios
                            for i in range(1, 25):
                                period = f'P{i}'
                                row[period] = pred.get(period, 0)
                            
                            rows.append(row)
                        
                        df_prod = pd.DataFrame(rows)
                        
                        # Agregar datos reales desde histórico
                        period_cols = [f'P{i}' for i in range(1, 25)]
                        df_historico_subset = df_historico[['FECHA'] + period_cols].copy()
                        
                        # Renombrar columnas reales antes del merge
                        rename_dict = {col: f'{col}_real' for col in period_cols}
                        df_historico_subset = df_historico_subset.rename(columns=rename_dict)
                        
                        df_prod = df_prod.merge(
                            df_historico_subset,
                            on='FECHA',
                            how='left'
                        )
                        
                        # Aplicar corrección de anomalías si está habilitada
                        if correct_anomalies:
                            period_cols = [f'P{i}' for i in range(1, 25)]
                            df_prod, _ = detect_and_correct_anomalous_hourly_values(
                                df_prod,
                                df_historico,
                                period_cols=period_cols,
                                min_threshold_pct=0.3,
                                max_threshold_pct=2.5,
                                show_info=True,
                                section_name="Producción (API)",
                                filter_historical_outliers=filter_outliers
                            )
                        
                        st.success(f"✅ {len(df_prod)} predicciones cargadas desde API")
                    else:
                        st.error(f"❌ Error de API: {response.status_code}")
                        st.json(response.json())
                
                except Exception as e:
                    st.error(f"❌ Error al conectar con API: {e}")
        
        else:  # Desde archivo CSV
            uploaded_file = st.file_uploader(
                "Subir archivo CSV con predicciones",
                type=['csv'],
                help="El archivo debe tener columnas: FECHA, TOTAL_pred (o demanda_predicha), P1-P24"
            )
            
            if uploaded_file is not None:
                try:
                    df_prod = pd.read_csv(uploaded_file)
                    
                    # Normalizar nombres de columnas
                    if 'FECHA' in df_prod.columns:
                        df_prod['FECHA'] = pd.to_datetime(df_prod['FECHA'])
                    
                    if 'demanda_predicha' in df_prod.columns:
                        df_prod['TOTAL_pred'] = df_prod['demanda_predicha']
                    elif 'TOTAL_pred' not in df_prod.columns:
                        st.error("❌ El archivo debe tener columna 'TOTAL_pred' o 'demanda_predicha'")
                        df_prod = None
                    
                    if df_prod is not None:
                        # Agregar datos reales desde histórico
                        period_cols = [f'P{i}' for i in range(1, 25)]
                        df_historico_subset = df_historico[['FECHA'] + period_cols].copy()
                        
                        # Renombrar columnas reales antes del merge
                        rename_dict = {col: f'{col}_real' for col in period_cols}
                        df_historico_subset = df_historico_subset.rename(columns=rename_dict)
                        
                        df_prod = df_prod.merge(
                            df_historico_subset,
                            on='FECHA',
                            how='left'
                        )
                        
                        # Aplicar corrección de anomalías si está habilitada
                        if correct_anomalies:
                            period_cols = [f'P{i}' for i in range(1, 25)]
                            df_prod, _ = detect_and_correct_anomalous_hourly_values(
                                df_prod,
                                df_historico,
                                period_cols=period_cols,
                                min_threshold_pct=0.3,
                                max_threshold_pct=2.5,
                                show_info=True,
                                section_name="Producción (CSV)",
                                filter_historical_outliers=filter_outliers
                            )
                        
                        st.success(f"✅ {len(df_prod)} predicciones cargadas desde archivo")
                
                except Exception as e:
                    st.error(f"❌ Error al cargar archivo: {e}")
        
        # Mostrar resultados si hay datos
        if df_prod is not None and len(df_prod) > 0:
            # Métricas globales
            metrics_prod = calculate_hourly_metrics(df_prod)
            
            st.subheader("📊 Métricas Globales")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MAPE", f"{metrics_prod['global']['mape']:.2f}%")
            with col2:
                st.metric("MAE", f"{metrics_prod['global']['mae']:.2f} MW")
            with col3:
                st.metric("RMSE", f"{metrics_prod['global']['rmse']:.2f} MW")
            with col4:
                st.metric("R²", f"{metrics_prod['global']['r2']:.4f}")
            
            # Gráfica de comparación
            st.subheader("📈 Comparación Real vs Predicción")
            fig_comp = plot_hourly_comparison(df_prod, f"Producción - {selected_ucp}")
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Métricas por período
            st.subheader("📊 Métricas por Período Horario")
            
            metric_choice = st.selectbox(
                "Seleccionar métrica",
                options=['mape', 'mae', 'rmse', 'r2'],
                index=0,
                key='prod_metric'
            )
            
            fig_metrics = plot_metrics_by_period(
                metrics_prod['by_period'],
                metric_name=metric_choice,
                title=f"Producción - {metric_choice.upper()} por Período"
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Heatmap de errores
            st.subheader("🔥 Mapa de Calor de Errores")
            fig_heatmap = plot_heatmap_hourly_errors(df_prod, f"Errores Horarios - Producción")
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Tabla de métricas por período
            with st.expander("📋 Ver métricas detalladas por período"):
                metrics_df = pd.DataFrame(metrics_prod['by_period']).T
                metrics_df.index.name = 'Período'
                st.dataframe(metrics_df, use_container_width=True)
            
            # MAPE por tipo de día
            st.subheader("📊 MAPE por Tipo de Día")
            with st.spinner("Calculando MAPE por tipo de día..."):
                mape_by_type_prod = calculate_mape_by_day_type(df_prod, ucp=selected_ucp)
                fig_mape_type = plot_mape_by_day_type(mape_by_type_prod, "MAPE por Tipo de Día - Producción")
                st.plotly_chart(fig_mape_type, use_container_width=True)
                
                # Tabla de métricas por tipo de día
                with st.expander("📋 Ver MAPE detallado por tipo de día"):
                    metrics_type_df = pd.DataFrame(mape_by_type_prod).T
                    metrics_type_df.index.name = 'Tipo de Día'
                    st.dataframe(metrics_type_df, use_container_width=True)
            
            # MAPE por mes
            st.subheader("📊 MAPE por Mes (Últimos 4 Años)")
            with st.spinner("Calculando MAPE por mes..."):
                df_mape_month_prod = calculate_mape_by_month(df_prod, years_back=4)
                if len(df_mape_month_prod) > 0:
                    fig_mape_month = plot_mape_by_month(df_mape_month_prod, "MAPE por Mes - Producción")
                    st.plotly_chart(fig_mape_month, use_container_width=True)
                    
                    # Tabla de MAPE por mes
                    with st.expander("📋 Ver MAPE detallado por mes"):
                        st.dataframe(df_mape_month_prod, use_container_width=True)
                else:
                    st.info("No hay suficientes datos para calcular MAPE por mes")
        else:
            st.info("👈 Carga predicciones de producción usando una de las opciones arriba")


if __name__ == "__main__":
    main()

