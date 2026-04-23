"""
Streamlit para visualizar Curvas Típicas
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Curvas Típicas", page_icon="📈", layout="wide")

API_BASE_URL = "http://localhost:8001"

st.title("📈 Visualización de Curvas Típicas")

# Sidebar - Parámetros
st.sidebar.header("Parámetros")

fecha_inicial = st.sidebar.date_input(
    "Fecha inicial",
    value=datetime(2025, 9, 1)
)

fecha_final = st.sidebar.date_input(
    "Fecha final",
    value=datetime(2025, 11, 30)
)

mc = st.sidebar.text_input("MC (Mercado/Centro)", value="Antioquia")

tipo_dia = st.sidebar.selectbox(
    "Tipo de día",
    ["ORDINARIO", "SABADO", "FESTIVO"]
)

flujo_tipo = st.sidebar.selectbox(
    "Tipo de flujo",
    ["A", "R"],
    format_func=lambda x: "A - Activa" if x == "A" else "R - Reactiva"
)

n_max = st.sidebar.number_input(
    "N máximo de curvas",
    min_value=1,
    max_value=100,
    value=9
)

barra = st.sidebar.text_input("Barra (opcional, vacío = todas del MC)", value="AMAGA__1")

ejecutar = st.sidebar.button("🔍 Buscar Curvas Típicas", type="primary")


def graficar_curvas(curvas, titulo, color_palette=None):
    """Genera gráfico Plotly para un conjunto de curvas."""
    fig = go.Figure()
    periodos = list(range(1, 25))

    colors = color_palette or [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    for i, curva in enumerate(curvas):
        barra_nombre = curva["barra"]
        fecha = curva["fecha"]
        valores = [curva["periodos"][f"p{p}"] for p in periodos]
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=periodos,
            y=valores,
            mode='lines+markers',
            name=f"{barra_nombre} ({fecha})",
            line=dict(color=color),
            hovertemplate=(
                f"<b>{barra_nombre}</b><br>"
                f"Fecha: {fecha}<br>"
                "Periodo: %{x}<br>"
                "Valor: %{y:.2f}<br>"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title=titulo,
        xaxis_title="Periodo (hora)",
        yaxis_title="Valor",
        height=450,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def curvas_a_dataframe(curvas):
    """Convierte lista de curvas a DataFrame."""
    periodos = list(range(1, 25))
    tabla_data = []
    for curva in curvas:
        row = {
            "Barra": curva["barra"],
            "Fecha": curva["fecha"]
        }
        for p in periodos:
            row[f"P{p}"] = round(curva["periodos"][f"p{p}"], 2)
        tabla_data.append(row)
    return pd.DataFrame(tabla_data)


# Ejecución
if ejecutar:
    with st.spinner("Consultando curvas típicas..."):
        try:
            payload = {
                "fecha_inicial": fecha_inicial.strftime("%Y-%m-%d"),
                "fecha_final": fecha_final.strftime("%Y-%m-%d"),
                "mc": mc,
                "tipo_dia": tipo_dia,
                "flujo_tipo": flujo_tipo,
                "n_max": n_max,
                "barra": barra if barra else None,
                "incluir_originales": True
            }

            response = requests.post(
                f"{API_BASE_URL}/factores/calculos/curvas-tipicas",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                resultado = response.json()

                if resultado["ok"] and resultado["n"] > 0:
                    st.success(f"Se encontraron {resultado['n']} curvas típicas")

                    # Métricas
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Curvas encontradas", resultado["n"])
                    col2.metric("Tipo día", tipo_dia)
                    col3.metric("Flujo", "Activa" if flujo_tipo == "A" else "Reactiva")
                    col4.metric("MC", mc)

                    curvas_cluster = resultado["data"]
                    curvas_originales = resultado.get("data_originales", [])

                    # Debug: verificar si hay diferencias
                    st.subheader("🔍 Debug: Verificación de datos")

                    if curvas_originales:
                        # Crear lookup para originales
                        orig_lookup = {(c["barra"], c["fecha"]): c["periodos"] for c in curvas_originales}

                        diferencias = []
                        for curva in curvas_cluster:
                            key = (curva["barra"], curva["fecha"])
                            if key in orig_lookup:
                                orig_periodos = orig_lookup[key]
                                for p in range(1, 25):
                                    val_cluster = curva["periodos"][f"p{p}"]
                                    val_orig = orig_periodos[f"p{p}"]
                                    if abs(val_cluster - val_orig) > 0.00001:
                                        diferencias.append({
                                            "barra": curva["barra"],
                                            "fecha": curva["fecha"],
                                            "periodo": f"p{p}",
                                            "clusterizado": val_cluster,
                                            "original": val_orig,
                                            "factor_implícito": round(val_cluster / val_orig, 5) if val_orig != 0 else "N/A"
                                        })

                        if diferencias:
                            st.success(f"Se encontraron {len(diferencias)} diferencias entre valores")
                            df_diff = pd.DataFrame(diferencias[:20])  # Mostrar primeras 20
                            st.dataframe(df_diff, use_container_width=True, hide_index=True)
                        else:
                            st.warning("⚠️ Los valores clusterizados y originales son IGUALES. Los factores probablemente son 1.0")
                    else:
                        st.error("No se recibieron curvas originales del servidor")

                    # Gráficas lado a lado
                    st.subheader("📊 Comparación: Clusterizadas vs Originales")

                    col_izq, col_der = st.columns(2)

                    with col_izq:
                        fig_cluster = graficar_curvas(
                            curvas_cluster,
                            "Curvas Clusterizadas (valor × factor)"
                        )
                        st.plotly_chart(fig_cluster, use_container_width=True)

                    with col_der:
                        if curvas_originales:
                            fig_orig = graficar_curvas(
                                curvas_originales,
                                "Curvas Originales (valores BD)"
                            )
                            st.plotly_chart(fig_orig, use_container_width=True)
                        else:
                            st.warning("No se encontraron curvas originales")

                    # Tablas de datos
                    st.subheader("📋 Datos de las curvas")

                    tab1, tab2 = st.tabs(["Clusterizadas", "Originales"])

                    with tab1:
                        df_cluster = curvas_a_dataframe(curvas_cluster)
                        st.dataframe(df_cluster, use_container_width=True, hide_index=True)

                    with tab2:
                        if curvas_originales:
                            df_orig = curvas_a_dataframe(curvas_originales)
                            st.dataframe(df_orig, use_container_width=True, hide_index=True)
                        else:
                            st.info("No hay datos originales disponibles")

                    # JSON raw
                    with st.expander("🔧 Ver JSON de respuesta"):
                        st.json(resultado)

                else:
                    st.warning("No se encontraron curvas típicas para los parámetros especificados")

            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("No se pudo conectar a la API. Asegúrate de que FastAPI esté corriendo en http://localhost:8001")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Info
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Valores mostrados:**

**Clusterizadas:**
- `valor_BD × factor`
- Agrupados por barra+fecha

**Originales:**
- Valores directos de BD
- Agrupados por barra+fecha
- Sin multiplicar por factor
""")
