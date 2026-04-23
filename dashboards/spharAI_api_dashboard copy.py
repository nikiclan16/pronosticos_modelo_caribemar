"""
Dashboard de Predicción - EPM
==============================

Dashboard Streamlit para generar predicciones de demanda energética.
Permite seleccionar ciudad, fechas, número de días y opción de retrain.
Muestra gráficas comparando predicciones vs datos históricos.

Utiliza la API directamente mediante llamadas HTTP.

Uso:
    streamlit run dashboards/prediction_api_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging
import requests
from typing import Optional, Dict, Any

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import RAW_DATA_DIR

# Configurar logging
logging.basicConfig(level=logging.WARNING)

# Configuración de página
st.set_page_config(
    page_title="Predicción - EPM",
    page_icon="⚡",
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
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURACIÓN DE API
# ============================================================================

# URL base de la API (configurable)
DEFAULT_API_URL = "http://localhost:8001"

# ============================================================================
# FUNCIONES DE CARGA DE DATOS HISTÓRICOS
# ============================================================================

@st.cache_data
def load_historical_data_for_comparison(ucp: str):
    """
    Carga datos históricos para comparación visual.
    
    Args:
        ucp: Nombre del UCP ('Atlantico' o 'Antioquia')
    
    Returns:
        DataFrame con datos históricos o None si no existe
    """
    # Intentar cargar desde datos procesados con features
    ucp_path = Path(f'data/features/{ucp}/data_with_features_latest.csv')
    if ucp_path.exists():
        try:
            df = pd.read_csv(ucp_path)
            if 'FECHA' in df.columns:
                df['FECHA'] = pd.to_datetime(df['FECHA'])
                df = df.sort_values('FECHA')
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
                return df
        except Exception as e:
            st.warning(f"Error cargando datos históricos: {e}")
    
    return None


def get_historical_total(df_historico, fecha_inicio, fecha_fin):
    """
    Extrae datos históricos de TOTAL para un rango de fechas
    
    Args:
        df_historico: DataFrame con datos históricos
        fecha_inicio: Fecha inicio (datetime)
        fecha_fin: Fecha fin (datetime)
    
    Returns:
        DataFrame con columnas: fecha, demanda_total
    """
    if df_historico is None or len(df_historico) == 0:
        return pd.DataFrame()
    
    # Filtrar por rango de fechas
    mask = (df_historico['FECHA'] >= fecha_inicio) & (df_historico['FECHA'] <= fecha_fin)
    df_filtered = df_historico[mask].copy()
    
    if len(df_filtered) == 0:
        return pd.DataFrame()
    
    # Normalizar nombre de columna TOTAL
    total_col = 'TOTAL' if 'TOTAL' in df_filtered.columns else 'demanda_total'
    if total_col not in df_filtered.columns:
        # Intentar calcular desde columnas P1-P24
        period_cols = [f'P{i}' for i in range(1, 25)]
        if all(col in df_filtered.columns for col in period_cols):
            df_filtered['TOTAL'] = df_filtered[period_cols].sum(axis=1)
            total_col = 'TOTAL'
        else:
            return pd.DataFrame()
    
    # Crear DataFrame de salida
    result = pd.DataFrame({
        'fecha': df_filtered['FECHA'],
        'demanda_total': df_filtered[total_col]
    })
    
    return result.sort_values('fecha')


# ============================================================================
# FUNCIONES DE LLAMADAS A LA API
# ============================================================================

def call_predict_api(api_url: str, ucp: str, n_days: int, end_date: Optional[str] = None, force_retrain: bool = False, offset_scalar: Optional[float] = None) -> Dict[str, Any]:
    """
    Llama al endpoint /predict de la API
    
    Args:
        api_url: URL base de la API
        ucp: Nombre del UCP
        n_days: Número de días a predecir
        end_date: Fecha final de datos históricos (formato YYYY-MM-DD)
        force_retrain: Forzar reentrenamiento
        offset_scalar: Escalar opcional para ajustar todas las predicciones (ej: 1.2 para aumentar 20%)
    
    Returns:
        Dict con la respuesta de la API o None si hay error
    """
    url = f"{api_url}/predict"
    
    payload = {
        "ucp": ucp,
        "n_days": n_days,
        "force_retrain": force_retrain
    }
    
    if end_date:
        payload["end_date"] = end_date
    
    if offset_scalar is not None and offset_scalar > 0:
        payload["offset_scalar"] = offset_scalar
    
    try:
        response = requests.post(url, json=payload, timeout=600)  # 10 minutos timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"No se pudo conectar a la API en {api_url}. Asegúrate de que el servidor esté corriendo."}
    except requests.exceptions.Timeout:
        return {"error": "La solicitud a la API tardó demasiado. Intenta nuevamente."}
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = response.json().get("detail", str(e))
            return {"error": f"Error HTTP {response.status_code}: {error_detail}"}
        except:
            return {"error": f"Error HTTP {response.status_code}: {str(e)}"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}


def call_retrain_api(api_url: str, ucp: str) -> Dict[str, Any]:
    """
    Llama al endpoint /retrain de la API
    
    Args:
        api_url: URL base de la API
        ucp: Nombre del UCP
    
    Returns:
        Dict con la respuesta de la API o None si hay error
    """
    url = f"{api_url}/retrain"
    
    try:
        response = requests.post(url, params={"ucp": ucp}, timeout=600)  # 10 minutos timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"No se pudo conectar a la API en {api_url}. Asegúrate de que el servidor esté corriendo."}
    except requests.exceptions.Timeout:
        return {"error": "La solicitud a la API tardó demasiado. Intenta nuevamente."}
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = response.json().get("detail", str(e))
            return {"error": f"Error HTTP {response.status_code}: {error_detail}"}
        except:
            return {"error": f"Error HTTP {response.status_code}: {str(e)}"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}


def call_base_curve_api(api_url: str, ucp: str, fecha_inicio: str, fecha_fin: str) -> Dict[str, Any]:
    """
    Llama al endpoint /api/v1/base-curve de la API
    
    Args:
        api_url: URL base de la API
        ucp: Nombre del UCP
        fecha_inicio: Fecha inicio (formato YYYY-MM-DD)
        fecha_fin: Fecha fin (formato YYYY-MM-DD)
    
    Returns:
        Dict con la respuesta de la API o None si hay error
    """
    url = f"{api_url}/base-curve"
    
    payload = {
        "ucp": ucp,
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)  # 5 minutos timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"No se pudo conectar a la API en {api_url}. Asegúrate de que el servidor esté corriendo."}
    except requests.exceptions.Timeout:
        return {"error": "La solicitud a la API tardó demasiado. Intenta nuevamente."}
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = response.json().get("detail", str(e))
            return {"error": f"Error HTTP {response.status_code}: {error_detail}"}
        except:
            return {"error": f"Error HTTP {response.status_code}: {str(e)}"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}


def check_api_health(api_url: str) -> bool:
    """
    Verifica si la API está disponible
    
    Args:
        api_url: URL base de la API
    
    Returns:
        True si la API está disponible, False en caso contrario
    """
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def convert_predictions_to_dataframe(api_response: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Convierte la respuesta de la API a un DataFrame de predicciones
    
    Args:
        api_response: Respuesta JSON de la API
    
    Returns:
        DataFrame con predicciones o None si hay error
    """
    if "error" in api_response:
        return None
    
    if "predictions" not in api_response:
        return None
    
    predictions = api_response["predictions"]
    
    if not predictions or len(predictions) == 0:
        return None
    
    # Convertir lista de predicciones a DataFrame
    rows = []
    for pred in predictions:
        fecha = pred.get("fecha")
        demanda_predicha = pred.get("demanda_predicha") or pred.get("demanda_total")
        
        if fecha and demanda_predicha is not None:
            rows.append({
                "fecha": pd.to_datetime(fecha),
                "demanda_predicha": demanda_predicha,
                "demanda_total": demanda_predicha  # Alias para compatibilidad
            })
    
    if len(rows) == 0:
        return None
    
    df = pd.DataFrame(rows)
    df = df.sort_values("fecha")
    return df


def convert_predictions_to_hourly_dataframe(api_response: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Convierte la respuesta de la API a un DataFrame con datos horarios (P1-P24)
    
    Args:
        api_response: Respuesta JSON de la API
    
    Returns:
        DataFrame con columnas: fecha_hora, demanda_horaria, tipo (prediccion/historico)
    """
    if "error" in api_response:
        return None
    
    if "predictions" not in api_response:
        return None
    
    predictions = api_response["predictions"]
    
    if not predictions or len(predictions) == 0:
        return None
    
    # Convertir lista de predicciones a DataFrame horario
    rows = []
    for pred in predictions:
        fecha_str = pred.get("fecha")
        if not fecha_str:
            continue
        
        fecha = pd.to_datetime(fecha_str)
        
        # Extraer períodos P1-P24
        for periodo in range(1, 25):
            periodo_key = f"P{periodo}"
            if periodo_key in pred:
                # Crear datetime con la hora correspondiente (periodo 1 = 00:00, periodo 2 = 01:00, etc.)
                # Los períodos van de 00:00-01:00 (P1) a 23:00-00:00 (P24)
                hora = periodo - 1  # P1 = 0, P2 = 1, ..., P24 = 23
                fecha_hora = fecha.replace(hour=hora, minute=0, second=0)
                
                rows.append({
                    "fecha_hora": fecha_hora,
                    "demanda_horaria": pred[periodo_key],
                    "tipo": "prediccion",
                    "periodo": periodo
                })
    
    if len(rows) == 0:
        return None
    
    df = pd.DataFrame(rows)
    df = df.sort_values("fecha_hora")
    return df


def get_historical_hourly_data(df_historico, fecha_inicio, fecha_fin):
    """
    Extrae datos históricos horarios (P1-P24) para un rango de fechas
    
    Args:
        df_historico: DataFrame con datos históricos (debe tener columnas P1-P24)
        fecha_inicio: Fecha inicio (datetime)
        fecha_fin: Fecha fin (datetime)
    
    Returns:
        DataFrame con columnas: fecha_hora, demanda_horaria, tipo
    """
    if df_historico is None or len(df_historico) == 0:
        return pd.DataFrame()
    
    # Filtrar por rango de fechas
    mask = (df_historico['FECHA'] >= fecha_inicio) & (df_historico['FECHA'] <= fecha_fin)
    df_filtered = df_historico[mask].copy()
    
    if len(df_filtered) == 0:
        return pd.DataFrame()
    
    # Verificar que tenga columnas P1-P24
    period_cols = [f'P{i}' for i in range(1, 25)]
    if not all(col in df_filtered.columns for col in period_cols):
        return pd.DataFrame()
    
    # Convertir a formato horario
    rows = []
    for _, row in df_filtered.iterrows():
        fecha = pd.to_datetime(row['FECHA'])
        
        for periodo in range(1, 25):
            periodo_key = f'P{periodo}'
            if periodo_key in row and pd.notna(row[periodo_key]):
                hora = periodo - 1  # P1 = 0, P2 = 1, ..., P24 = 23
                fecha_hora = fecha.replace(hour=hora, minute=0, second=0)
                
                rows.append({
                    "fecha_hora": fecha_hora,
                    "demanda_horaria": row[periodo_key],
                    "tipo": "historico",
                    "periodo": periodo
                })
    
    if len(rows) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df = df.sort_values("fecha_hora")
    return df


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def plot_predictions_vs_historical(predictions_df, historical_df, title="Predicciones vs Históricos"):
    """
    Crea gráfica comparando predicciones con datos históricos
    
    Args:
        predictions_df: DataFrame con predicciones (debe tener 'fecha' y 'demanda_predicha' o 'demanda_total')
        historical_df: DataFrame con datos históricos (debe tener 'fecha' y 'demanda_total')
        title: Título de la gráfica
    
    Returns:
        Figura de Plotly
    """
    fig = go.Figure()
    
    # Agregar línea de predicciones
    if predictions_df is not None and len(predictions_df) > 0:
        # Convertir fecha si es necesario
        pred_dates = pd.to_datetime(predictions_df['fecha'])
        # Intentar diferentes nombres de columna para demanda
        if 'demanda_predicha' in predictions_df.columns:
            pred_values = predictions_df['demanda_predicha']
        elif 'demanda_total' in predictions_df.columns:
            pred_values = predictions_df['demanda_total']
        elif 'TOTAL' in predictions_df.columns:
            pred_values = predictions_df['TOTAL']
        else:
            pred_values = None
        
        if pred_values is not None:
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=pred_values,
                mode='lines+markers',
                name='Predicción',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=6),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Predicción: %{y:,.0f} MWh<extra></extra>'
            ))
    
    # Agregar línea de datos históricos
    if historical_df is not None and len(historical_df) > 0:
        hist_dates = pd.to_datetime(historical_df['fecha'])
        hist_values = historical_df['demanda_total']
        
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=hist_values,
            mode='lines+markers',
            name='Histórico Real',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Real: %{y:,.0f} MWh<extra></extra>'
        ))
    
    # Configurar layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='#1f77b4')
        ),
        xaxis=dict(
            title='Fecha',
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Demanda (MWh)',
            showgrid=True,
            gridcolor='lightgray'
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_hourly_predictions_vs_historical(hourly_predictions_df, hourly_historical_df, title="Predicciones Horarias vs Históricos"):
    """
    Crea gráfica comparando predicciones horarias con datos históricos horarios
    
    Args:
        hourly_predictions_df: DataFrame con predicciones horarias (debe tener 'fecha_hora' y 'demanda_horaria')
        hourly_historical_df: DataFrame con datos históricos horarios (debe tener 'fecha_hora' y 'demanda_horaria')
        title: Título de la gráfica
    
    Returns:
        Figura de Plotly
    """
    fig = go.Figure()
    
    # Agregar línea de predicciones horarias
    if hourly_predictions_df is not None and len(hourly_predictions_df) > 0:
        pred_dates = pd.to_datetime(hourly_predictions_df['fecha_hora'])
        pred_values = hourly_predictions_df['demanda_horaria']
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_values,
            mode='lines',
            name='Predicción Horaria',
            line=dict(color='#ff7f0e', width=2),
            hovertemplate='<b>%{x|%Y-%m-%d %H:00}</b><br>Predicción: %{y:,.2f} MWh<extra></extra>'
        ))
    
    # Agregar línea de datos históricos horarios
    if hourly_historical_df is not None and len(hourly_historical_df) > 0:
        hist_dates = pd.to_datetime(hourly_historical_df['fecha_hora'])
        hist_values = hourly_historical_df['demanda_horaria']
        
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=hist_values,
            mode='lines',
            name='Histórico Real Horario',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>%{x|%Y-%m-%d %H:00}</b><br>Real: %{y:,.2f} MWh<extra></extra>'
        ))
    
    # Configurar layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='#1f77b4')
        ),
        xaxis=dict(
            title='Fecha y Hora',
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Demanda Horaria (MWh)',
            showgrid=True,
            gridcolor='lightgray'
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        template='plotly_white'
    )
    
    return fig


def convert_base_curves_to_hourly_dataframe(api_response: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Convierte la respuesta del endpoint base-curve a un DataFrame con datos horarios
    
    Args:
        api_response: Respuesta JSON de la API con formato {"curves": {"YYYY-MM-DD": [P1, P2, ..., P24]}}
    
    Returns:
        DataFrame con columnas: fecha_hora, demanda_horaria, tipo
    """
    if "error" in api_response:
        return None
    
    if "curves" not in api_response:
        return None
    
    curves = api_response["curves"]
    
    if not curves or len(curves) == 0:
        return None
    
    # Convertir diccionario de curvas a DataFrame horario
    rows = []
    for fecha_str, periodos in curves.items():
        fecha = pd.to_datetime(fecha_str)
        
        # Cada día tiene 24 períodos (P1-P24)
        for periodo in range(1, 25):
            if periodo <= len(periodos):
                # Crear datetime con la hora correspondiente (periodo 1 = 00:00, periodo 2 = 01:00, etc.)
                hora = periodo - 1  # P1 = 0, P2 = 1, ..., P24 = 23
                fecha_hora = fecha.replace(hour=hora, minute=0, second=0)
                
                rows.append({
                    "fecha_hora": fecha_hora,
                    "demanda_horaria": periodos[periodo - 1],  # Índice 0-based
                    "tipo": "curva_base",
                    "periodo": periodo
                })
    
    if len(rows) == 0:
        return None
    
    df = pd.DataFrame(rows)
    df = df.sort_values("fecha_hora")
    return df


def plot_base_curves_vs_historical(base_curves_df, historical_df, title="Curvas Base vs Históricos"):
    """
    Crea gráfica comparando curvas base con datos históricos horarios
    
    Args:
        base_curves_df: DataFrame con curvas base (debe tener 'fecha_hora' y 'demanda_horaria')
        historical_df: DataFrame con datos históricos horarios (debe tener 'fecha_hora' y 'demanda_horaria')
        title: Título de la gráfica
    
    Returns:
        Figura de Plotly
    """
    fig = go.Figure()
    
    # Agregar línea de curvas base
    if base_curves_df is not None and len(base_curves_df) > 0:
        base_dates = pd.to_datetime(base_curves_df['fecha_hora'])
        base_values = base_curves_df['demanda_horaria']
        
        fig.add_trace(go.Scatter(
            x=base_dates,
            y=base_values,
            mode='lines',
            name='Curva Base',
            line=dict(color='#2ca02c', width=2),
            hovertemplate='<b>%{x|%Y-%m-%d %H:00}</b><br>Curva Base: %{y:,.2f} MW<extra></extra>'
        ))
    
    # Agregar línea de datos históricos horarios
    if historical_df is not None and len(historical_df) > 0:
        hist_dates = pd.to_datetime(historical_df['fecha_hora'])
        hist_values = historical_df['demanda_horaria']
        
        fig.add_trace(go.Scatter(
            x=hist_dates,
            y=hist_values,
            mode='lines',
            name='Histórico Real',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>%{x|%Y-%m-%d %H:00}</b><br>Real: %{y:,.2f} MW<extra></extra>'
        ))
    
    # Configurar layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='#1f77b4')
        ),
        xaxis=dict(
            title='Fecha y Hora',
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Demanda Horaria (MW)',
            showgrid=True,
            gridcolor='lightgray'
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        template='plotly_white'
    )
    
    return fig


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">⚡ Dashboard de Predicción - EPM</h1>', unsafe_allow_html=True)
    
    # Sidebar - Configuración
    st.sidebar.header("⚙️ Configuración")
    
    # URL de la API
    api_url = st.sidebar.text_input(
        "URL de la API",
        value=DEFAULT_API_URL,
        help="URL base del servidor de la API (ej: http://localhost:8001)"
    )
    
    # Verificar conexión con la API
    api_available = check_api_health(api_url)
    if api_available:
        st.sidebar.success("✅ API conectada")
    else:
        st.sidebar.error(f"❌ No se pudo conectar a la API en {api_url}")
        st.sidebar.info("💡 Asegúrate de que el servidor de la API esté corriendo")
    
    # Selección de UCP
    ucp_options = ['Atlantico', 'Antioquia']
    selected_ucp = st.sidebar.selectbox(
        "Seleccionar Ciudad/UCP",
        options=ucp_options,
        index=0
    )
    
    # Fecha de fin de datos históricos (end_date)
    st.sidebar.subheader("📅 Fechas")
    
    # Obtener última fecha disponible en histórico
    df_historico_full = load_historical_data_for_comparison(selected_ucp)
    default_end_date = datetime.now().date() - timedelta(days=1)
    min_date = datetime(2015, 1, 1).date()
    max_date = datetime.now().date()
    
    if df_historico_full is not None and len(df_historico_full) > 0:
        max_historical_date = df_historico_full['FECHA'].max().date()
        min_historical_date = df_historico_full['FECHA'].min().date()
        
        if max_historical_date < default_end_date:
            default_end_date = max_historical_date
        
        if min_historical_date > min_date:
            min_date = min_historical_date
        if max_historical_date < max_date:
            max_date = max_historical_date
        
        st.sidebar.info(f"📊 Rango disponible: {min_historical_date.strftime('%Y-%m-%d')} a {max_historical_date.strftime('%Y-%m-%d')}")
    else:
        st.sidebar.warning("⚠️ No se encontraron datos históricos")
    
    end_date = st.sidebar.date_input(
        "Fecha final de datos históricos",
        value=default_end_date,
        min_value=min_date,
        max_value=max_date,
        help="Última fecha con datos históricos a usar. Las predicciones comenzarán desde el día siguiente."
    )
    
    if df_historico_full is not None and len(df_historico_full) > 0:
        prediction_start_date = end_date + timedelta(days=1)
        st.sidebar.info(f"🔮 Las predicciones comenzarán desde: {prediction_start_date.strftime('%Y-%m-%d')}")
    
    # Número de días a predecir
    n_days = st.sidebar.number_input(
        "Número de días a predecir",
        min_value=1,
        max_value=90,
        value=30,
        help="Cantidad de días futuros a predecir (máximo 90)"
    )
    
    # Opción de force_retrain
    force_retrain = st.sidebar.checkbox(
        "Forzar reentrenamiento",
        value=False,
        help="Si está marcado, el modelo se reentrenará antes de generar predicciones"
    )
    
    # Offset scalar para ajustar predicciones
    st.sidebar.subheader("🔧 Ajustes de Predicción")
    offset_scalar = st.sidebar.slider(
        "Offset Scalar",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.01,
        help="Escalar para ajustar todas las predicciones (1.0 = sin ajuste, 1.2 = +20%, 0.8 = -20%)"
    )
    
    # Mostrar información del offset
    if offset_scalar != 1.0:
        cambio_pct = (offset_scalar - 1.0) * 100
        if cambio_pct > 0:
            st.sidebar.info(f"📈 Aumento: +{cambio_pct:.1f}%")
        else:
            st.sidebar.info(f"📉 Disminución: {cambio_pct:.1f}%")
    
    # Botones de acción
    st.sidebar.markdown("---")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        predict_button = st.button("🔮 Predecir", type="primary", use_container_width=True)
    
    with col2:
        train_button = st.button("🔧 Entrenar", use_container_width=True)
    
    # ========================================================================
    # PROCESAMIENTO DE ACCIONES
    # ========================================================================
    
    # Inicializar estado de sesión
    if 'predictions_df' not in st.session_state:
        st.session_state.predictions_df = None
    if 'api_response' not in st.session_state:
        st.session_state.api_response = None
    if 'last_ucp' not in st.session_state:
        st.session_state.last_ucp = None
    if 'last_n_days' not in st.session_state:
        st.session_state.last_n_days = None
    
    # Acción: Entrenar
    if train_button:
        if not api_available:
            st.error(f"❌ No se pudo conectar a la API. Verifica que el servidor esté corriendo en {api_url}")
        else:
            st.info(f"🔧 Iniciando entrenamiento para {selected_ucp}... Esto puede tardar varios minutos.")
            
            with st.spinner("Entrenando modelo (esto puede tardar 5-10 minutos)..."):
                response = call_retrain_api(api_url, selected_ucp)
            
            if "error" in response:
                st.error(f"❌ {response['error']}")
            else:
                st.success(f"✅ {response.get('message', 'Modelo entrenado exitosamente')}")
                
                # Mostrar métricas si están disponibles
                if "metricas" in response and isinstance(response["metricas"], dict):
                    metrics = response["metricas"]
                    st.subheader("📊 Métricas del Modelo Entrenado")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'val_metrics' in metrics:
                            mape = metrics['val_metrics'].get('mape', 0)
                            st.metric("MAPE", f"{mape:.2f}%")
                    
                    with col2:
                        if 'val_metrics' in metrics:
                            rmape = metrics['val_metrics'].get('rmape', 0)
                            st.metric("rMAPE", f"{rmape:.4f}")
                    
                    with col3:
                        if 'val_metrics' in metrics:
                            r2 = metrics['val_metrics'].get('r2', 0)
                            st.metric("R²", f"{r2:.4f}")
                    
                    # Mostrar métricas completas en expander
                    with st.expander("📄 Ver métricas completas"):
                        st.json(metrics)
                elif "datos_entrenamiento" in response:
                    st.info(f"📊 Datos de entrenamiento: {response['datos_entrenamiento']}")
                
                # Limpiar cache para que se recargue el modelo
                st.cache_data.clear()
                st.info("💡 El modelo ha sido actualizado. Puedes generar nuevas predicciones ahora.")
    
    # Acción: Predecir
    if predict_button:
        if not api_available:
            st.error(f"❌ No se pudo conectar a la API. Verifica que el servidor esté corriendo en {api_url}")
        else:
            st.info(f"🔮 Generando predicción para {selected_ucp}...")
            st.info(f"📅 Usando datos históricos hasta: {end_date.strftime('%Y-%m-%d')}")
            
            with st.spinner(f"Prediciendo {n_days} días (esto puede tardar 1-2 minutos)..."):
                # Convertir end_date a string para la API
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                # Mostrar información del offset si está aplicado
                if offset_scalar != 1.0:
                    cambio_pct = (offset_scalar - 1.0) * 100
                    if cambio_pct > 0:
                        st.info(f"🔧 Aplicando offset: +{cambio_pct:.1f}% (scalar: {offset_scalar:.2f})")
                    else:
                        st.info(f"🔧 Aplicando offset: {cambio_pct:.1f}% (scalar: {offset_scalar:.2f})")
                
                response = call_predict_api(
                    api_url=api_url,
                    ucp=selected_ucp,
                    n_days=n_days,
                    end_date=end_date_str,
                    force_retrain=force_retrain,
                    offset_scalar=offset_scalar if offset_scalar != 1.0 else None
                )
            
            if "error" in response:
                st.error(f"❌ Error en predicción: {response['error']}")
                st.session_state.predictions_df = None
                st.session_state.api_response = None
            else:
                st.success("✅ Predicción generada exitosamente!")
                
                # Convertir respuesta a DataFrame
                predictions_df = convert_predictions_to_dataframe(response)
                
                if predictions_df is None or len(predictions_df) == 0:
                    st.warning("⚠️ No se recibieron predicciones en la respuesta de la API")
                    st.json(response)  # Mostrar respuesta completa para debugging
                else:
                    st.session_state.predictions_df = predictions_df
                    st.session_state.api_response = response
                    st.session_state.last_ucp = selected_ucp
                    st.session_state.last_n_days = n_days
    
    # ========================================================================
    # VISUALIZACIÓN DE RESULTADOS
    # ========================================================================
    
    if st.session_state.predictions_df is not None:
        predictions_df = st.session_state.predictions_df
        api_response = st.session_state.api_response
        
        # Asegurar que la columna de fecha esté en formato datetime
        if 'fecha' in predictions_df.columns:
            predictions_df['fecha'] = pd.to_datetime(predictions_df['fecha'])
        
        # Determinar columna de demanda
        if 'demanda_predicha' in predictions_df.columns:
            demanda_col = 'demanda_predicha'
        elif 'demanda_total' in predictions_df.columns:
            demanda_col = 'demanda_total'
        elif 'TOTAL' in predictions_df.columns:
            demanda_col = 'TOTAL'
        else:
            demanda_col = None
        
        if demanda_col is None:
            st.error("❌ No se encontró columna de demanda en las predicciones")
        else:
            # Calcular rango de fechas de predicciones
            fecha_inicio_pred = predictions_df['fecha'].min()
            fecha_fin_pred = predictions_df['fecha'].max()
            
            # Cargar datos históricos para comparación visual
            df_historico = load_historical_data_for_comparison(selected_ucp)
            
            # Extraer datos históricos del rango de fechas de predicciones
            historical_df = pd.DataFrame()
            if df_historico is not None and len(df_historico) > 0:
                historical_df = get_historical_total(df_historico, fecha_inicio_pred, fecha_fin_pred)
            
            # Mostrar metadata de la API si está disponible
            if api_response and "metadata" in api_response:
                metadata = api_response["metadata"]
                st.markdown("---")
                st.subheader("📊 Información de la Predicción")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if "fecha_inicio" in metadata:
                        st.metric("Fecha Inicio", metadata["fecha_inicio"])
                
                with col2:
                    if "fecha_fin" in metadata:
                        st.metric("Fecha Fin", metadata["fecha_fin"])
                
                with col3:
                    if "dias_predichos" in metadata:
                        st.metric("Días Predichos", metadata["dias_predichos"])
                
                with col4:
                    if "modelo_entrenado" in metadata:
                        status = "✅ Sí" if metadata["modelo_entrenado"] else "❌ No"
                        st.metric("Modelo Entrenado", status)
                
                # Mostrar métricas del modelo si están disponibles
                if "metricas_modelo" in metadata:
                    st.markdown("---")
                    st.subheader("📈 Métricas del Modelo")
                    metrics = metadata["metricas_modelo"]
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if "mape" in metrics:
                            st.metric("MAPE", f"{metrics['mape']:.2f}%")
                    
                    with col2:
                        if "rmape" in metrics:
                            st.metric("rMAPE", f"{metrics['rmape']:.4f}")
                    
                    with col3:
                        if "r2" in metrics:
                            st.metric("R²", f"{metrics['r2']:.4f}")
            
            # Mostrar métricas principales de predicciones
            st.markdown("---")
            st.subheader("📊 Métricas de Predicción")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Demanda Promedio",
                    f"{predictions_df[demanda_col].mean():,.0f} MWh"
                )
            
            with col2:
                st.metric(
                    "Demanda Mínima",
                    f"{predictions_df[demanda_col].min():,.0f} MWh"
                )
            
            with col3:
                st.metric(
                    "Demanda Máxima",
                    f"{predictions_df[demanda_col].max():,.0f} MWh"
                )
            
            with col4:
                st.metric(
                    "Total Días",
                    f"{len(predictions_df)}"
                )
            
            # Mostrar recomendación de reentrenamiento si está disponible
            if api_response and "should_retrain" in api_response:
                st.markdown("---")
                if api_response["should_retrain"]:
                    st.warning(f"⚠️ **Recomendación:** {api_response.get('reason', 'Se recomienda reentrenar el modelo')}")
                else:
                    st.success(f"✅ {api_response.get('reason', 'El modelo está funcionando correctamente')}")
            
            # Mostrar eventos futuros si están disponibles
            if api_response and "events" in api_response and api_response["events"]:
                st.markdown("---")
                st.subheader("📅 Eventos Futuros Relevantes")
                events = api_response["events"]
                for fecha, evento in events.items():
                    st.info(f"**{fecha}**: {evento}")
            
            # Gráfica principal
            st.markdown("---")
            st.subheader("📈 Predicciones vs Históricos")
            
            # Preparar datos para gráfica
            pred_for_plot = pd.DataFrame({
                'fecha': predictions_df['fecha'],
                'demanda_total': predictions_df[demanda_col]
            })
            
            # Crear gráfica
            fig = plot_predictions_vs_historical(
                pred_for_plot,
                historical_df,
                title=f"Predicciones vs Históricos - {selected_ucp}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ========================================================================
            # GRÁFICA HORARIA
            # ========================================================================
            st.markdown("---")
            st.subheader("⏰ Predicciones Horarias vs Históricos")
            
            # Convertir predicciones a formato horario
            hourly_predictions_df = convert_predictions_to_hourly_dataframe(api_response)
            
            # Extraer datos históricos horarios si están disponibles
            hourly_historical_df = pd.DataFrame()
            if df_historico is not None and len(df_historico) > 0:
                hourly_historical_df = get_historical_hourly_data(df_historico, fecha_inicio_pred, fecha_fin_pred)
            
            # Crear gráfica horaria
            if hourly_predictions_df is not None and len(hourly_predictions_df) > 0:
                fig_hourly = plot_hourly_predictions_vs_historical(
                    hourly_predictions_df,
                    hourly_historical_df,
                    title=f"Predicciones Horarias vs Históricos - {selected_ucp}"
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
                
                # Mostrar información sobre los datos horarios
                if len(hourly_historical_df) > 0:
                    st.info(f"📊 Mostrando {len(hourly_predictions_df)} horas de predicciones y {len(hourly_historical_df)} horas de datos históricos")
                else:
                    st.info(f"📊 Mostrando {len(hourly_predictions_df)} horas de predicciones (no hay datos históricos horarios disponibles para este período)")
            else:
                st.warning("⚠️ No se pudieron extraer datos horarios de las predicciones")
            
            # Mostrar tabla de comparación si hay datos históricos
            if len(historical_df) > 0:
                st.markdown("---")
                st.subheader("📋 Comparación Detallada")
                
                # Unir predicciones con históricos
                comparison_df = pred_for_plot.copy()
                comparison_df = comparison_df.merge(
                    historical_df[['fecha', 'demanda_total']],
                    on='fecha',
                    how='left',
                    suffixes=('_pred', '_real')
                )
                
                # Calcular errores
                comparison_df['error_abs'] = (
                    comparison_df['demanda_total_pred'] - comparison_df['demanda_total_real']
                ).abs()
                comparison_df['error_pct'] = (
                    (comparison_df['error_abs'] / comparison_df['demanda_total_real']) * 100
                ).round(2)
                
                # Renombrar columnas para visualización
                comparison_df = comparison_df.rename(columns={
                    'fecha': 'Fecha',
                    'demanda_total_pred': 'Predicción (MWh)',
                    'demanda_total_real': 'Real (MWh)',
                    'error_abs': 'Error Absoluto (MWh)',
                    'error_pct': 'Error (%)'
                })
                
                # Mostrar tabla
                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Métricas de error
                if comparison_df['Error (%)'].notna().any():
                    avg_error = comparison_df['Error (%)'].mean()
                    max_error = comparison_df['Error (%)'].max()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Error Promedio (%)", f"{avg_error:.2f}%")
                    with col2:
                        st.metric("Error Máximo (%)", f"{max_error:.2f}%")
            
            # Mostrar datos completos en expander
            with st.expander("📄 Ver datos completos de predicción"):
                st.dataframe(predictions_df, use_container_width=True)
            
            # Mostrar respuesta completa de la API en expander (para debugging)
            with st.expander("🔍 Ver respuesta completa de la API"):
                st.json(api_response)
    
    else:
        # Mensaje inicial
        st.info("👈 Configura los parámetros en la barra lateral y haz clic en '🔮 Predecir' para generar una predicción")
        
        # Mostrar información del sistema
        st.markdown("---")
        st.subheader("ℹ️ Información del Sistema")
        
        if api_available:
            st.success("✅ API conectada y disponible")
        else:
            st.error(f"❌ No se pudo conectar a la API en {api_url}")
            st.info("💡 Para iniciar la API, ejecuta: `uvicorn src.api.main:app --reload`")
    
    # ========================================================================
    # SECCIÓN: CURVAS BASE
    # ========================================================================
    st.markdown("---")
    st.markdown("---")
    st.subheader("📊 Curvas Base de Demanda")
    st.markdown("Obtén curvas base de demanda horaria para un rango de fechas específico")
    
    # Formulario para curvas base
    col1, col2, col3 = st.columns(3)
    
    with col1:
        base_curve_ucp = st.selectbox(
            "UCP para Curva Base",
            options=ucp_options,
            index=0,
            key="base_curve_ucp"
        )
    
    with col2:
        base_curve_fecha_inicio = st.date_input(
            "Fecha Inicio",
            value=datetime.now().date(),
            key="base_curve_fecha_inicio"
        )
    
    with col3:
        base_curve_fecha_fin = st.date_input(
            "Fecha Fin",
            value=datetime.now().date() + timedelta(days=30),
            key="base_curve_fecha_fin"
        )
    
    base_curve_button = st.button("📊 Obtener Curvas Base", type="primary", use_container_width=True)
    
    # Inicializar estado de sesión para curvas base
    if 'base_curves_df' not in st.session_state:
        st.session_state.base_curves_df = None
    if 'base_curves_response' not in st.session_state:
        st.session_state.base_curves_response = None
    if 'base_curves_historical_df' not in st.session_state:
        st.session_state.base_curves_historical_df = None
    
    # Procesar solicitud de curvas base
    if base_curve_button:
        if not api_available:
            st.error(f"❌ No se pudo conectar a la API. Verifica que el servidor esté corriendo en {api_url}")
        else:
            if base_curve_fecha_inicio > base_curve_fecha_fin:
                st.error("❌ La fecha de inicio debe ser anterior o igual a la fecha de fin")
            else:
                st.info(f"📊 Obteniendo curvas base para {base_curve_ucp} desde {base_curve_fecha_inicio} hasta {base_curve_fecha_fin}...")
                
                with st.spinner("Consultando endpoint de curvas base..."):
                    fecha_inicio_str = base_curve_fecha_inicio.strftime('%Y-%m-%d')
                    fecha_fin_str = base_curve_fecha_fin.strftime('%Y-%m-%d')
                    
                    response = call_base_curve_api(
                        api_url=api_url,
                        ucp=base_curve_ucp,
                        fecha_inicio=fecha_inicio_str,
                        fecha_fin=fecha_fin_str
                    )
                
                if "error" in response:
                    st.error(f"❌ Error obteniendo curvas base: {response['error']}")
                    st.session_state.base_curves_df = None
                    st.session_state.base_curves_response = None
                else:
                    st.success("✅ Curvas base obtenidas exitosamente!")
                    
                    # Convertir respuesta a DataFrame
                    base_curves_df = convert_base_curves_to_hourly_dataframe(response)
                    
                    if base_curves_df is None or len(base_curves_df) == 0:
                        st.warning("⚠️ No se recibieron curvas base en la respuesta de la API")
                        st.json(response)  # Mostrar respuesta completa para debugging
                    else:
                        st.session_state.base_curves_df = base_curves_df
                        st.session_state.base_curves_response = response
                        
                        # Intentar cargar datos históricos para comparación
                        df_historico = load_historical_data_for_comparison(base_curve_ucp)
                        if df_historico is not None and len(df_historico) > 0:
                            fecha_inicio_dt = pd.to_datetime(base_curve_fecha_inicio)
                            fecha_fin_dt = pd.to_datetime(base_curve_fecha_fin)
                            historical_df = get_historical_hourly_data(df_historico, fecha_inicio_dt, fecha_fin_dt)
                            st.session_state.base_curves_historical_df = historical_df
                        else:
                            st.session_state.base_curves_historical_df = pd.DataFrame()
    
    # Mostrar gráfica de curvas base si hay datos
    if st.session_state.base_curves_df is not None and len(st.session_state.base_curves_df) > 0:
        base_curves_df = st.session_state.base_curves_df
        historical_df = st.session_state.base_curves_historical_df if st.session_state.base_curves_historical_df is not None else pd.DataFrame()
        
        # Calcular estadísticas
        st.markdown("---")
        st.subheader("📈 Visualización de Curvas Base")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Demanda Promedio",
                f"{base_curves_df['demanda_horaria'].mean():,.0f} MW"
            )
        
        with col2:
            st.metric(
                "Demanda Mínima",
                f"{base_curves_df['demanda_horaria'].min():,.0f} MW"
            )
        
        with col3:
            st.metric(
                "Demanda Máxima",
                f"{base_curves_df['demanda_horaria'].max():,.0f} MW"
            )
        
        with col4:
            total_horas = len(base_curves_df)
            total_dias = len(set(base_curves_df['fecha_hora'].dt.date))
            st.metric(
                "Total Horas",
                f"{total_horas} ({total_dias} días)"
            )
        
        # Crear gráfica
        fig_base = plot_base_curves_vs_historical(
            base_curves_df,
            historical_df,
            title=f"Curvas Base vs Históricos - {base_curve_ucp}"
        )
        
        st.plotly_chart(fig_base, use_container_width=True)
        
        # Información sobre datos históricos
        if historical_df is not None and len(historical_df) > 0:
            st.info(f"📊 Mostrando {len(base_curves_df)} horas de curvas base y {len(historical_df)} horas de datos históricos")
            
            # Mostrar tabla de comparación si hay datos históricos
            st.markdown("---")
            st.subheader("📋 Comparación Detallada")
            
            # Unir curvas base con históricos
            comparison_df = base_curves_df[['fecha_hora', 'demanda_horaria']].copy()
            comparison_df = comparison_df.rename(columns={'demanda_horaria': 'curva_base'})
            
            comparison_df = comparison_df.merge(
                historical_df[['fecha_hora', 'demanda_horaria']],
                on='fecha_hora',
                how='left',
                suffixes=('', '_real')
            )
            comparison_df = comparison_df.rename(columns={'demanda_horaria': 'historico_real'})
            
            # Calcular errores
            comparison_df['error_abs'] = (
                comparison_df['curva_base'] - comparison_df['historico_real']
            ).abs()
            comparison_df['error_pct'] = (
                (comparison_df['error_abs'] / comparison_df['historico_real']) * 100
            ).round(2)
            
            # Renombrar columnas para visualización
            comparison_df = comparison_df.rename(columns={
                'fecha_hora': 'Fecha y Hora',
                'curva_base': 'Curva Base (MW)',
                'historico_real': 'Histórico Real (MW)',
                'error_abs': 'Error Absoluto (MW)',
                'error_pct': 'Error (%)'
            })
            
            # Mostrar tabla
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Métricas de error
            if comparison_df['Error (%)'].notna().any():
                avg_error = comparison_df['Error (%)'].mean()
                max_error = comparison_df['Error (%)'].max()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Error Promedio (%)", f"{avg_error:.2f}%")
                with col2:
                    st.metric("Error Máximo (%)", f"{max_error:.2f}%")
        else:
            st.info(f"📊 Mostrando {len(base_curves_df)} horas de curvas base (no hay datos históricos disponibles para este período)")
        
        # Mostrar datos completos en expander
        with st.expander("📄 Ver datos completos de curvas base"):
            st.dataframe(base_curves_df, use_container_width=True)
        
        # Mostrar respuesta completa de la API en expander (para debugging)
        with st.expander("🔍 Ver respuesta completa de la API"):
            st.json(st.session_state.base_curves_response)


if __name__ == "__main__":
    main()
