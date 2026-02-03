"""
Aplicación Streamlit para demostración de cálculos FDA/FDP
Sistema de Factores de Demanda Eléctrica - EPM
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Factores EPM",
    page_icon="⚡",
    layout="wide"
)

# URL base de la API
API_BASE_URL = "http://localhost:8000"

# Título principal
st.title("⚡ Sistema de Factores de Demanda Eléctrica")
st.markdown("### Cálculo de FDA (Factor de Demanda Ajustada) y FDP (Factor de Potencia)")

# Sidebar para configuración
st.sidebar.header("Configuración")

# Selector de operación
operacion = st.sidebar.selectbox(
    "Selecciona la operación",
    ["FDA - Factor de Demanda Ajustada",
     "FDP - Factor de Potencia",
     "FDA + FDP Combinado",
     "Clustering - Agrupación de Medidas"]
)

# Parámetros comunes
st.sidebar.subheader("Parámetros de consulta")

# Fechas
fecha_inicial = st.sidebar.date_input(
    "Fecha inicial",
    value=datetime(2025, 1, 1),
    min_value=datetime(2024, 12, 15),
    max_value=datetime(2025, 12, 22)
)

fecha_final = st.sidebar.date_input(
    "Fecha final",
    value=datetime(2025, 1, 31),
    min_value=datetime(2024, 12, 15),
    max_value=datetime(2025, 12, 22)
)

# Lista de barras disponibles
barras_disponibles = [
    "POBLADO1", "S_JERON1", "PTO_INM1", "GUADALU4", "HISPANI1",
    "AMAGA__1", "ANCONSU1", "ANTIOQ41", "BARBOSA1", "BELEN__1"
]

barra = st.sidebar.selectbox("Barra eléctrica", barras_disponibles)
mc = st.sidebar.selectbox("MC (Mercado/Centro)", ["Antioquia"], index=0)

# Parámetros específicos para clustering
if "Clustering" in operacion:
    flujo_tipo = st.sidebar.selectbox("Tipo de flujo", ["A", "R"])
    tipo_dia_clustering = st.sidebar.selectbox(
        "Tipo de día (opcional)",
        ["", "ORDINARIO", "SABADO", "FESTIVO"]
    )

# Botón para ejecutar
ejecutar = st.sidebar.button("🔄 Ejecutar Cálculo", type="primary")

# Función para formatear resultados FDA/FDP
def mostrar_resultados_fda_fdp(resultados, titulo):
    st.subheader(titulo)

    for tipo_dia, datos in resultados.items():
        if datos.get("n_registros", 0) == 0:
            continue

        st.markdown(f"#### 📅 {tipo_dia}")
        st.metric("Número de registros", datos["n_registros"])

        if "suma_total" in datos:
            st.metric("Suma total de periodos", f"{datos['suma_total']:.2f}")

        if "ajuste_aplicado" in datos:
            st.metric("Ajuste aplicado", f"{datos['ajuste_aplicado']:.5f}")

        # Convertir factores a DataFrame
        factores_dict = datos.get("factores", {})
        if factores_dict:
            df = pd.DataFrame(factores_dict).T

            # Gráfico de líneas por periodo
            periodos_cols = [col for col in df.columns if col.startswith('p') or col.startswith('fdp_p')]

            if periodos_cols:
                fig = go.Figure()

                for idx, row in df.iterrows():
                    fecha = row.get('fecha', f'Registro {idx}')
                    valores = [row[col] for col in periodos_cols if col in row]

                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(valores) + 1)),
                        y=valores,
                        mode='lines+markers',
                        name=fecha
                    ))

                fig.update_layout(
                    title=f"Evolución de factores por periodo - {tipo_dia}",
                    xaxis_title="Periodo (hora)",
                    yaxis_title="Factor",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

            # Tabla de datos
            st.dataframe(df, use_container_width=True)

# Función para mostrar resultados de clustering
def mostrar_resultados_clustering(data):
    st.subheader("Resultados de Clustering")

    if not data:
        st.warning("No se encontraron datos para los parámetros especificados")
        return

    df = pd.DataFrame(data)

    st.metric("Total de registros", len(df))

    # Expandir periodos a columnas
    periodos_data = []
    for _, row in df.iterrows():
        row_data = {
            'barra': row['barra'],
            'fecha': row['fecha']
        }
        row_data.update(row['periodos'])
        periodos_data.append(row_data)

    df_expandido = pd.DataFrame(periodos_data)

    # Gráfico
    periodos_cols = [f'p{i}' for i in range(1, 25)]

    fig = go.Figure()

    for _, row in df_expandido.iterrows():
        valores = [row[col] for col in periodos_cols]
        fecha = row['fecha']

        fig.add_trace(go.Scatter(
            x=list(range(1, 25)),
            y=valores,
            mode='lines+markers',
            name=fecha
        ))

    fig.update_layout(
        title="Medidas agrupadas por fecha y periodo",
        xaxis_title="Periodo (hora)",
        yaxis_title="Valor de medida",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Tabla
    st.dataframe(df_expandido, use_container_width=True)

# Ejecución principal
if ejecutar:
    with st.spinner("Ejecutando cálculo..."):
        try:
            # Preparar payload
            fecha_ini_str = fecha_inicial.strftime("%Y-%m-%d")
            fecha_fin_str = fecha_final.strftime("%Y-%m-%d")

            if "FDA - Factor" in operacion:
                # Endpoint FDA
                payload = {
                    "fecha_inicial": fecha_ini_str,
                    "fecha_final": fecha_fin_str,
                    "mc": mc,
                    "barra": barra,
                    "codigo_rpm": []
                }

                response = requests.post(
                    f"{API_BASE_URL}/factores/calculos/fda",
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    resultado = response.json()
                    st.success("✅ Cálculo completado exitosamente")

                    # Información general
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Barra", resultado["barra"])
                    col2.metric("MC", resultado["mc"])
                    col3.metric("Rango de fechas", f"{fecha_ini_str} a {fecha_fin_str}")

                    mostrar_resultados_fda_fdp(resultado["resultados"], "Resultados FDA")
                else:
                    st.error(f"Error en la API: {response.status_code} - {response.text}")

            elif "FDP - Factor" in operacion:
                # Endpoint FDP
                payload = {
                    "fecha_inicial": fecha_ini_str,
                    "fecha_final": fecha_fin_str,
                    "mc": mc,
                    "barra": barra,
                    "codigo_rpm": []
                }

                response = requests.post(
                    f"{API_BASE_URL}/factores/calculos/fdp",
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    resultado = response.json()
                    st.success("✅ Cálculo completado exitosamente")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Barra", resultado["barra"])
                    col2.metric("MC", resultado["mc"])
                    col3.metric("Rango de fechas", f"{fecha_ini_str} a {fecha_fin_str}")

                    mostrar_resultados_fda_fdp(resultado["resultados"], "Resultados FDP")
                else:
                    st.error(f"Error en la API: {response.status_code} - {response.text}")

            elif "FDA + FDP" in operacion:
                # Endpoint combinado
                payload = {
                    "fecha_inicial": fecha_ini_str,
                    "fecha_final": fecha_fin_str,
                    "mc": mc,
                    "barra": barra,
                    "codigo_rpm": []
                }

                response = requests.post(
                    f"{API_BASE_URL}/factores/calculos/fda-fdp",
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    resultado = response.json()
                    st.success("✅ Cálculo completado exitosamente")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Barra", resultado["barra"])
                    col2.metric("MC", resultado["mc"])
                    col3.metric("Rango de fechas", f"{fecha_ini_str} a {fecha_fin_str}")

                    # Mostrar FDA
                    mostrar_resultados_fda_fdp(resultado["fda"], "📊 Resultados FDA")

                    st.markdown("---")

                    # Mostrar FDP
                    mostrar_resultados_fda_fdp(resultado["fdp"], "⚡ Resultados FDP")
                else:
                    st.error(f"Error en la API: {response.status_code} - {response.text}")

            elif "Clustering" in operacion:
                # Endpoint clustering
                payload = {
                    "fecha_inicial": fecha_ini_str,
                    "fecha_final": fecha_fin_str,
                    "mc": mc,
                    "barra": barra,
                    "flujo_tipo": flujo_tipo,
                    "tipo_dia": tipo_dia_clustering
                }

                response = requests.post(
                    f"{API_BASE_URL}/factores/calculos/clustering",
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    resultado = response.json()
                    st.success("✅ Cálculo completado exitosamente")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Barra", barra)
                    col2.metric("MC", mc)
                    col3.metric("Flujo", flujo_tipo)
                    col4.metric("Tipo día", tipo_dia_clustering if tipo_dia_clustering else "Todos")

                    mostrar_resultados_clustering(resultado["data"])
                else:
                    st.error(f"Error en la API: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ No se pudo conectar a la API. Asegúrate de que el servidor FastAPI esté corriendo en http://localhost:8000")
        except Exception as e:
            st.error(f"❌ Error inesperado: {str(e)}")

# Información adicional
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Información")
st.sidebar.markdown("""
**FDA (Factor de Demanda Ajustada)**:
Normaliza factores para que sumen 1.0 por periodo

**FDP (Factor de Potencia)**:
Mide eficiencia eléctrica (0.94-1.0)

**Clustering**:
Agrupa medidas por barra y aplica factores
""")

# Footer
st.markdown("---")
st.markdown("Sistema de Factores EPM - Refactorización FastAPI + Streamlit")
