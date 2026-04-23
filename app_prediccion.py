"""
Aplicación Streamlit para Predicción de Demanda Energética EPM
Llama al endpoint /predict y visualiza histórico + predicciones
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Configuración de la página
st.set_page_config(
    page_title="EPM - Predicción Demanda Energética",
    page_icon="⚡",
    layout="wide"
)

# Título
st.title("⚡ Sistema de Predicción de Demanda Energética EPM")
st.markdown("---")

# Sidebar con parámetros
st.sidebar.header("Parámetros de Predicción")

# URL del API
api_url = st.sidebar.text_input(
    "URL del API",
    value="http://localhost:8001",
    help="URL base del servidor FastAPI"
)

# Parámetros del request
power_data_path = st.sidebar.text_input(
    "Ruta datos de potencia",
    value="data/raw/datos.csv"
)

weather_data_path = st.sidebar.text_input(
    "Ruta datos climáticos",
    value="data/raw/clima.csv"
)

start_date = st.sidebar.date_input(
    "Fecha inicio datos históricos",
    value=datetime(2017, 1, 1),
    help="Fecha desde la cual usar datos históricos"
)

n_days = st.sidebar.slider(
    "Días a predecir",
    min_value=1,
    max_value=90,
    value=30,
    help="Número de días futuros a predecir"
)

# NUEVO: Modo de validación
modo_validacion = st.sidebar.checkbox(
    "🔍 Modo Validación",
    value=True,
    help="Predice sobre los últimos N días históricos para comparar con datos reales"
)

if modo_validacion:
    st.sidebar.info(
        "**Modo Validación activado:**\n"
        "El sistema excluirá los últimos días del entrenamiento "
        "y los predecirá para comparar con la realidad."
    )

force_retrain = st.sidebar.checkbox(
    "Forzar reentrenamiento",
    value=False,
    help="Entrenar nuevos modelos aunque ya existan"
)

# Botón de predicción
predict_button = st.sidebar.button("🚀 Generar Predicción", type="primary")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Nota:** El servidor API debe estar ejecutándose en el puerto 8000.\n\n"
    "Para iniciarlo: `uvicorn src.api.main:app --reload`"
)

# Función para cargar datos históricos
@st.cache_data
def load_historical_data(filepath: str):
    """Carga datos históricos de demanda"""
    try:
        df = pd.read_csv(filepath)
        if 'FECHA' in df.columns:
            df['FECHA'] = pd.to_datetime(df['FECHA'])
        return df
    except Exception as e:
        st.error(f"Error cargando datos históricos: {str(e)}")
        return None

# Función para llamar al endpoint
def call_predict_endpoint(payload: dict):
    """Llama al endpoint /predict"""
    try:
        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            timeout=300  # 5 minutos timeout
        )

        if response.status_code == 200:
            return response.json(), None
        else:
            error_detail = response.json().get('detail', 'Error desconocido')
            return None, f"Error {response.status_code}: {error_detail}"

    except requests.exceptions.ConnectionError:
        return None, "No se pudo conectar al servidor. Verifica que esté corriendo en el puerto 8000."
    except requests.exceptions.Timeout:
        return None, "La petición tardó demasiado (timeout de 5 minutos)."
    except Exception as e:
        return None, f"Error inesperado: {str(e)}"

# Main content
if predict_button:
    # Cargar datos históricos primero para calcular fechas
    df_historico_full = load_historical_data(power_data_path)

    if df_historico_full is None:
        st.error("No se pudieron cargar los datos históricos")
        st.stop()

    # Usar fecha fija para validación
    end_date_str = None
    if modo_validacion:
        end_date_str = "2025-03-01"
        st.info(
            f"📅 **Modo Validación:**\n"
            f"- Usando end_date fija: {end_date_str}\n"
            f"- Prediciendo {n_days} días desde esa fecha.\n"
            f"- Esto permite comparar predicciones vs datos reales"
        )

    # Construir payload
    payload = {
        "power_data_path": power_data_path,
        "weather_data_path": weather_data_path,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "n_days": n_days,
        "force_retrain": force_retrain,
        "ucp": "Atlantico" 
    }

    # Agregar end_date solo si está en modo validación
    if end_date_str:
        payload["end_date"] = end_date_str

    # Mostrar payload
    with st.expander("Ver JSON del Request"):
        st.json(payload)

    # Llamar al endpoint
    with st.spinner(f"Generando predicción para {n_days} días... Esto puede tardar unos minutos."):
        result, error = call_predict_endpoint(payload)

    if error:
        st.error(error)
    else:
        st.success(f"Predicción generada exitosamente: {len(result['predictions'])} días")

        # Convertir predicciones a DataFrame
        predictions_list = []
        for pred in result['predictions']:
            predictions_list.append({
                'fecha': pd.to_datetime(pred['fecha']),
                'demanda_total': pred['demanda_total'],
                'dia_semana': pred['dia_semana'],
                'is_festivo': pred['is_festivo'],
                'is_weekend': pred['is_weekend']
            })

        df_predicciones = pd.DataFrame(predictions_list)

        # Si está en modo validación, obtener valores reales para comparar
        df_reales = None
        if modo_validacion and df_historico_full is not None:
            # Filtrar datos reales para las fechas predichas
            fechas_prediccion = df_predicciones['fecha'].tolist()
            df_reales = df_historico_full[
                pd.to_datetime(df_historico_full['FECHA']).isin(fechas_prediccion)
            ].copy()
            df_reales['FECHA'] = pd.to_datetime(df_reales['FECHA'])
            df_reales = df_reales.sort_values('FECHA')

            # Calcular métricas de error
            df_comparacion = df_predicciones.merge(
                df_reales[['FECHA', 'TOTAL']],
                left_on='fecha',
                right_on='FECHA',
                how='inner'
            )

            if len(df_comparacion) > 0:
                # Calcular errores
                df_comparacion['error'] = df_comparacion['demanda_total'] - df_comparacion['TOTAL']
                df_comparacion['error_abs'] = df_comparacion['error'].abs()
                df_comparacion['error_pct'] = (df_comparacion['error'] / df_comparacion['TOTAL'] * 100)
                df_comparacion['error_pct_abs'] = df_comparacion['error_pct'].abs()

                mae = df_comparacion['error_abs'].mean()
                mape = df_comparacion['error_pct_abs'].mean()
                rmse = (df_comparacion['error'] ** 2).mean() ** 0.5

        # Métricas generales
        if modo_validacion and df_reales is not None and len(df_comparacion) > 0:
            st.success(f"✅ Modo Validación: Comparando {len(df_comparacion)} días de predicciones vs datos reales")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "MAE (Error Absoluto Medio)",
                    f"{mae:,.0f} MWh",
                    help="Mean Absolute Error - Promedio del error absoluto"
                )

            with col2:
                st.metric(
                    "MAPE",
                    f"{mape:.2f}%",
                    help="Mean Absolute Percentage Error - Error porcentual promedio"
                )

            with col3:
                st.metric(
                    "RMSE",
                    f"{rmse:,.0f} MWh",
                    help="Root Mean Squared Error - Raíz del error cuadrático medio"
                )

            with col4:
                st.metric(
                    "Días Comparados",
                    len(df_comparacion)
                )
        else:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Días Predichos",
                    len(df_predicciones)
                )

            with col2:
                st.metric(
                    "Demanda Promedio",
                    f"{df_predicciones['demanda_total'].mean():,.0f} MWh"
                )

            with col3:
                st.metric(
                    "Demanda Mínima",
                    f"{df_predicciones['demanda_total'].min():,.0f} MWh"
                )

            with col4:
                st.metric(
                    "Demanda Máxima",
                    f"{df_predicciones['demanda_total'].max():,.0f} MWh"
                )

        st.markdown("---")

        # Gráfico principal: Histórico + Predicciones (+ Reales si modo validación)
        st.subheader("📊 Demanda Histórica y Predicciones")

        # Filtrar últimos 90 días de histórico para mejor visualización
        df_historico_reciente = df_historico_full.tail(90).copy()

        # Crear figura
        fig = go.Figure()

        # Histórico (excluyendo el período de validación si aplica)
        if modo_validacion and df_reales is not None:
            # Mostrar histórico solo hasta la fecha de corte
            fecha_corte_dt = pd.to_datetime(end_date_str)
            df_historico_plot = df_historico_reciente[
                pd.to_datetime(df_historico_reciente['FECHA']) <= fecha_corte_dt
            ]
        else:
            df_historico_plot = df_historico_reciente

        fig.add_trace(go.Scatter(
            x=df_historico_plot['FECHA'],
            y=df_historico_plot['TOTAL'],
            mode='lines',
            name='Histórico (Entrenamiento)',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Demanda: %{y:,.0f} MWh<extra></extra>'
        ))

        # Predicciones
        fig.add_trace(go.Scatter(
            x=df_predicciones['fecha'],
            y=df_predicciones['demanda_total'],
            mode='lines+markers',
            name='Predicción',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=6),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Predicción: %{y:,.0f} MWh<extra></extra>'
        ))

        # Valores reales (solo en modo validación)
        if modo_validacion and df_reales is not None and len(df_reales) > 0:
            fig.add_trace(go.Scatter(
                x=df_reales['FECHA'],
                y=df_reales['TOTAL'],
                mode='lines+markers',
                name='Valores Reales',
                line=dict(color='#2ca02c', width=2),
                marker=dict(size=8, symbol='circle'),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Real: %{y:,.0f} MWh<extra></extra>'
            ))

            # Marcar festivos en predicciones
            df_festivos = df_predicciones[df_predicciones['is_festivo'] == True]
            if len(df_festivos) > 0:
                fig.add_trace(go.Scatter(
                    x=df_festivos['fecha'],
                    y=df_festivos['demanda_total'],
                    mode='markers',
                    name='Festivos',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='star',
                        line=dict(color='darkred', width=1)
                    ),
                    hovertemplate='<b>%{x|%Y-%m-%d} - FESTIVO</b><br>Demanda: %{y:,.0f} MWh<extra></extra>'
                ))

            # Layout
            fig.update_layout(
                title=f"Demanda Energética: Últimos 90 días + Próximos {n_days} días",
                xaxis_title="Fecha",
                yaxis_title="Demanda Total (MWh)",
                hovermode='x unified',
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

        st.plotly_chart(fig, use_container_width=True)

        # Análisis de errores (solo en modo validación)
        if modo_validacion and df_reales is not None and len(df_comparacion) > 0:
            st.markdown("---")
            st.subheader("📉 Análisis de Errores de Predicción")

            col1, col2 = st.columns(2)

            with col1:
                # Gráfico de error absoluto por día
                fig_error = go.Figure()

                fig_error.add_trace(go.Bar(
                    x=df_comparacion['fecha'],
                    y=df_comparacion['error'],
                    name='Error (Predicción - Real)',
                    marker_color=['red' if x < 0 else 'green' for x in df_comparacion['error']],
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Error: %{y:,.0f} MWh<extra></extra>'
                ))

                fig_error.add_hline(y=0, line_dash="dash", line_color="gray")

                fig_error.update_layout(
                    title="Error por Día (Predicción - Real)",
                    xaxis_title="Fecha",
                    yaxis_title="Error (MWh)",
                    height=400
                )

                st.plotly_chart(fig_error, use_container_width=True)

            with col2:
                # Gráfico de error porcentual
                fig_error_pct = go.Figure()

                fig_error_pct.add_trace(go.Scatter(
                    x=df_comparacion['fecha'],
                    y=df_comparacion['error_pct_abs'],
                    mode='lines+markers',
                    name='Error Porcentual Absoluto',
                    line=dict(color='purple', width=2),
                    marker=dict(size=6),
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Error: %{y:.2f}%<extra></extra>'
                ))

                fig_error_pct.add_hline(
                    y=mape,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"MAPE Promedio: {mape:.2f}%"
                )

                fig_error_pct.update_layout(
                    title="Error Porcentual Absoluto por Día",
                    xaxis_title="Fecha",
                    yaxis_title="Error Absoluto (%)",
                    height=400
                )

                st.plotly_chart(fig_error_pct, use_container_width=True)

            # Distribución de errores
            col1, col2 = st.columns(2)

            with col1:
                # Histograma de errores
                fig_hist = go.Figure()

                fig_hist.add_trace(go.Histogram(
                    x=df_comparacion['error'],
                    nbinsx=20,
                    name='Distribución de Error',
                    marker_color='steelblue'
                ))

                fig_hist.update_layout(
                    title="Distribución de Errores",
                    xaxis_title="Error (MWh)",
                    yaxis_title="Frecuencia",
                    height=400
                )

                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                # Scatter: Real vs Predicho
                fig_scatter = go.Figure()

                fig_scatter.add_trace(go.Scatter(
                    x=df_comparacion['TOTAL'],
                    y=df_comparacion['demanda_total'],
                    mode='markers',
                    marker=dict(size=10, color='blue', opacity=0.6),
                    name='Predicciones',
                    hovertemplate='<b>Real: %{x:,.0f}</b><br>Predicho: %{y:,.0f}<extra></extra>'
                ))

                # Línea de predicción perfecta
                min_val = min(df_comparacion['TOTAL'].min(), df_comparacion['demanda_total'].min())
                max_val = max(df_comparacion['TOTAL'].max(), df_comparacion['demanda_total'].max())

                fig_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Predicción Perfecta',
                    showlegend=True
                ))

                fig_scatter.update_layout(
                    title="Real vs Predicho",
                    xaxis_title="Demanda Real (MWh)",
                    yaxis_title="Demanda Predicha (MWh)",
                    height=400
                )

                st.plotly_chart(fig_scatter, use_container_width=True)

        # Tabla de predicciones
        st.markdown("---")
        st.subheader("📋 Tabla de Predicciones Detalladas")

        # Formatear para mostrar
        if modo_validacion and df_reales is not None and len(df_comparacion) > 0:
            # Tabla con comparación
            df_display = df_comparacion.copy()
            df_display['fecha'] = df_display['fecha'].dt.strftime('%Y-%m-%d')
            df_display['Predicción (MWh)'] = df_display['demanda_total'].round(2)
            df_display['Real (MWh)'] = df_display['TOTAL'].round(2)
            df_display['Error (MWh)'] = df_display['error'].round(2)
            df_display['Error (%)'] = df_display['error_pct'].round(2)
            df_display['Festivo'] = df_display['is_festivo'].apply(lambda x: '🎉' if x else '')
            df_display['Fin de Semana'] = df_display['is_weekend'].apply(lambda x: '🏖️' if x else '')

            df_display = df_display.rename(columns={
                'fecha': 'Fecha',
                'dia_semana': 'Día de la Semana'
            })

            st.dataframe(
                df_display[['Fecha', 'Día de la Semana', 'Predicción (MWh)', 'Real (MWh)', 'Error (MWh)', 'Error (%)', 'Festivo', 'Fin de Semana']],
                use_container_width=True,
                height=400
            )
        else:
            # Tabla solo con predicciones
            df_display = df_predicciones.copy()
            df_display['fecha'] = df_display['fecha'].dt.strftime('%Y-%m-%d')
            df_display['demanda_total'] = df_display['demanda_total'].round(2)
            df_display['Festivo'] = df_display['is_festivo'].apply(lambda x: '🎉' if x else '')
            df_display['Fin de Semana'] = df_display['is_weekend'].apply(lambda x: '🏖️' if x else '')

            df_display = df_display.rename(columns={
                'fecha': 'Fecha',
                'demanda_total': 'Demanda Total (MWh)',
                'dia_semana': 'Día de la Semana'
            })

            st.dataframe(
                df_display[['Fecha', 'Día de la Semana', 'Demanda Total (MWh)', 'Festivo', 'Fin de Semana']],
                use_container_width=True,
                height=400
            )

            # Análisis por día de la semana
            st.subheader("📈 Análisis por Día de la Semana")

            col1, col2 = st.columns(2)

            with col1:
                # Promedio por día de la semana
                avg_by_day = df_predicciones.groupby('dia_semana')['demanda_total'].mean().round(2)

                # Ordenar días de la semana
                dias_orden = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                avg_by_day = avg_by_day.reindex([d for d in dias_orden if d in avg_by_day.index])

                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=avg_by_day.index,
                        y=avg_by_day.values,
                        marker_color='#2ca02c',
                        text=avg_by_day.values,
                        texttemplate='%{text:,.0f}',
                        textposition='outside'
                    )
                ])

                fig_bar.update_layout(
                    title="Demanda Promedio por Día de la Semana",
                    xaxis_title="Día",
                    yaxis_title="Demanda Promedio (MWh)",
                    height=400
                )

                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                # Distribución
                fig_box = go.Figure()

                for dia in dias_orden:
                    datos_dia = df_predicciones[df_predicciones['dia_semana'] == dia]['demanda_total']
                    if len(datos_dia) > 0:
                        fig_box.add_trace(go.Box(
                            y=datos_dia,
                            name=dia,
                            boxmean='sd'
                        ))

                fig_box.update_layout(
                    title="Distribución de Demanda por Día",
                    yaxis_title="Demanda (MWh)",
                    height=400,
                    showlegend=False
                )

                st.plotly_chart(fig_box, use_container_width=True)

            # Descargar resultados
            st.markdown("---")
            st.subheader("💾 Descargar Resultados")

            col1, col2 = st.columns(2)

            with col1:
                # CSV de predicciones
                csv = df_display.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Predicciones (CSV)",
                    data=csv,
                    file_name=f"predicciones_epm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col2:
                # JSON completo
                json_str = json.dumps(result, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Descargar Respuesta Completa (JSON)",
                    data=json_str,
                    file_name=f"predicciones_completas_epm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

else:
    # Pantalla inicial
    st.info("👈 Configura los parámetros en el panel lateral y haz clic en **Generar Predicción**")

    # Mostrar ejemplo de uso
    st.subheader("ℹ️ Cómo usar esta aplicación")

    st.markdown("""
    1. **Asegúrate de que el servidor API esté corriendo:**
       ```bash
       uvicorn src.api.main:app --reload --port 8000
       ```

    2. **Configura los parámetros en el panel lateral:**
       - Rutas a los archivos CSV de datos
       - Fecha de inicio para datos históricos
       - Número de días a predecir (1-90)
       - Si quieres forzar reentrenamiento de modelos

    3. **Haz clic en "Generar Predicción"**

    4. **Visualiza los resultados:**
       - Gráfico de histórico + predicciones
       - Tabla detallada con todas las predicciones
       - Análisis por día de la semana
       - Descarga los resultados en CSV o JSON

    ---

    ### Valores por defecto:
    - **Datos de potencia:** `data/raw/datos.csv`
    - **Datos climáticos:** `data/raw/clima.csv`
    - **Fecha inicio:** 2017-01-01
    - **Días a predecir:** 30
    - **Reentrenamiento:** No
    """)

    # Mostrar estado del API
    st.markdown("---")
    st.subheader("🔌 Estado del API")

    try:
        response = requests.get(f"{api_url}/health", timeout=2)
        if response.status_code == 200:
            st.success(f"✅ API conectada correctamente en {api_url}")
            health_data = response.json()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Status", health_data.get('status', 'N/A'))
            with col2:
                st.metric("Timestamp", health_data.get('timestamp', 'N/A'))
        else:
            st.warning(f"⚠️ API responde pero con código {response.status_code}")
    except:
        st.error(f"❌ No se pudo conectar al API en {api_url}")
        st.info("Verifica que el servidor esté corriendo: `uvicorn src.api.main:app --reload --port 8000`")
