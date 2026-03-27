"""
Servicio de cálculos FDA/FDP y Clustering.

Este módulo implementa los algoritmos de:
- Curvas típicas: clustering en el histórico para detectar patrones y devolver las N curvas más típicas (forma y nivel).
- FDA (Factor de Demanda Ajustada): Normalización sobre las curvas típicas seleccionadas (suma 1.0).
- FDP (Factor de Demanda Pronóstico): Cos(Atan(Q/P)) sobre las curvas típicas seleccionadas.
- Clustering (agregación): Aplicación de factores multiplicadores a medidas por barra+fecha.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from app.services import factores_service


# =============================================================================
# CONSTANTES
# =============================================================================

PRECISION_DECIMALES = 5
PERIODOS_COLUMNAS = [f'p{i}' for i in range(1, 25)]
TIPOS_DIA = ["ORDINARIO", "SABADO", "FESTIVO"]


# =============================================================================
# FUNCIONES AUXILIARES - CLUSTERING
# =============================================================================

def _agrupar_medidas_clusterizadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa medidas por barra+fecha y suma periodos.

    Args:
        df: DataFrame con columnas: barra, fecha, p1-p24

    Returns:
        DataFrame agrupado con periodos sumados y redondeados a 5 decimales
    """
    return df.groupby(['barra', 'fecha'])[PERIODOS_COLUMNAS].sum().round(PRECISION_DECIMALES)


def _curvas_a_matriz(curvas: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Convierte lista de curvas {barra, fecha, periodos} en matriz (n x 24) y lista de (barra, fecha).
    """
    if not curvas:
        return np.array([]).reshape(0, 24), []
    filas = []
    keys = []
    for c in curvas:
        p = c.get("periodos") or c
        if isinstance(p, dict):
            filas.append([float(p.get(f"p{i}", 0)) for i in range(1, 25)])
        else:
            filas.append([float(x) for x in p[:24]])
        keys.append((c["barra"], c["fecha"]))
    return np.array(filas), keys


def _filtrar_outliers_iqr(curvas: List[Dict[str, Any]], factor_iqr: float = 1.5) -> List[Dict[str, Any]]:
    """
    Filtra curvas que tienen valores outliers usando el método IQR.

    Una curva se descarta si tiene AL MENOS UN período fuera de los límites:
    - lower = Q1 - factor_iqr * IQR
    - upper = Q3 + factor_iqr * IQR

    Args:
        curvas: Lista de curvas con periodos p1-p24
        factor_iqr: Multiplicador del IQR (default 1.5)

    Returns:
        Lista de curvas sin outliers
    """
    if not curvas or len(curvas) < 4:
        return curvas

    X, keys = _curvas_a_matriz(curvas)
    n_curvas, n_periodos = X.shape

    # Calcular Q1, Q3, IQR para cada período
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1

    lower = q1 - factor_iqr * iqr
    upper = q3 + factor_iqr * iqr

    # Identificar curvas válidas (ningún período fuera de límites)
    curvas_validas = []
    for i in range(n_curvas):
        es_outlier = False
        for j in range(n_periodos):
            if X[i, j] < lower[j] or X[i, j] > upper[j]:
                es_outlier = True
                break
        if not es_outlier:
            curvas_validas.append(curvas[i])

    return curvas_validas


def _seleccionar_curvas_tipicas(
    curvas: List[Dict[str, Any]], n_max: int
) -> List[Dict[str, Any]]:
    """
    De una lista de curvas (barra, fecha, periodos), devuelve hasta n_max más típicas
    por forma y nivel: normaliza L2, mide centralidad (menor distancia media = más típica).
    Si hay menos de n_max curvas, devuelve todas las encontradas.

    IMPORTANTE: Primero filtra outliers usando IQR antes de calcular tipicidad.
    """
    if not curvas:
        return []

    # Paso 1: Filtrar outliers por IQR
    curvas_filtradas = _filtrar_outliers_iqr(curvas)

    if not curvas_filtradas:
        return []
    if len(curvas_filtradas) <= n_max:
        return curvas_filtradas

    X, keys = _curvas_a_matriz(curvas_filtradas)
    # Normalizar por L2 para que forma y nivel relativo cuenten
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms

    # Distancia euclidiana entre todas las filas (menor distancia media = más típica/central)
    n = len(Xn)
    mean_dists = np.zeros(n)
    for i in range(n):
        d = np.array([np.linalg.norm(Xn[i] - Xn[j]) for j in range(n) if j != i])
        mean_dists[i] = float(d.mean()) if len(d) else 0.0

    # Más típicas = menor distancia media (más centrales)
    indices = np.argsort(mean_dists)[:n_max]
    return [curvas_filtradas[i] for i in indices]


# =============================================================================
# FUNCIONES AUXILIARES - FDA
# =============================================================================

def _calcular_fda_normalizado(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica algoritmo FDA: normaliza valores para que cada período sume exactamente 1.0.

    Algoritmo:
    1. Normalizar: dividir cada valor por la suma de su período (para que sume 1.0)
    2. Calcular el ajuste fino necesario (diferencia con 1.0 por errores de redondeo)
    3. Aplicar ajuste al valor máximo de cada período
    4. Resultado: cada período suma exactamente 1.0

    Args:
        df: DataFrame con periodos p1-p24 (valores absolutos de potencia)

    Returns:
        DataFrame con factores FDA normalizados (cada período suma 1.0)
    """
    # Paso 1: Normalizar dividiendo por la suma de cada período
    sumas_por_periodo = df[PERIODOS_COLUMNAS].sum()

    # Evitar división por cero
    sumas_por_periodo = sumas_por_periodo.replace(0, 1)

    # Normalizar: cada valor / suma del período
    df_normalizado = df[PERIODOS_COLUMNAS].div(sumas_por_periodo)

    # Paso 2: Calcular ajuste fino (por errores de redondeo después de normalizar)
    sumas_normalizadas = df_normalizado.sum()
    ajustes_por_periodo = 1.0 - sumas_normalizadas

    # Paso 3: Encontrar máximos por período (en valores normalizados)
    maximos_por_periodo = df_normalizado.max()
    maximos_ajustados = maximos_por_periodo + ajustes_por_periodo

    # Paso 4: Aplicar ajuste solo al máximo de cada período
    def aplicar_ajuste_row(row):
        return pd.Series([
            round(maximos_ajustados[col], PRECISION_DECIMALES) if abs(row[col] - maximos_por_periodo[col]) < 1e-9
            else round(row[col], PRECISION_DECIMALES)
            for col in PERIODOS_COLUMNAS
        ], index=PERIODOS_COLUMNAS)

    df_ajustado = df_normalizado.apply(aplicar_ajuste_row, axis=1)

    return df_ajustado.round(PRECISION_DECIMALES)


# =============================================================================
# FUNCIONES AUXILIARES - FDP
# =============================================================================

def _calcular_fdp_vectorizado(df_a: pd.DataFrame, df_r: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula FDP para todos los periodos usando numpy vectorizado.

    FDP = Cos(Atan(Potencia_Reactiva / Potencia_Activa))

    Casos especiales:
    - Si P=0 y Q=0: FDP = 1.0
    - Si P=0 y Q≠0: FDP = 0.0

    Args:
        df_a: DataFrame con medidas activas (tipo A); debe tener fecha y opcionalmente barra
        df_r: DataFrame con medidas reactivas (tipo R)

    Returns:
        DataFrame con columnas fdp_p1 a fdp_p24 (y fecha, barra si aplica)
    """
    on_cols = ['barra', 'fecha'] if 'barra' in df_a.columns and 'barra' in df_r.columns else ['fecha']
    cols_a = on_cols + PERIODOS_COLUMNAS
    cols_r = on_cols + PERIODOS_COLUMNAS
    df_merged = pd.merge(
        df_a[cols_a],
        df_r[cols_r],
        on=on_cols,
        suffixes=('_a', '_r')
    )

    # Calcular FDP para cada periodo usando numpy vectorizado
    for i in range(1, 25):
        col_a = f'p{i}_a'
        col_r = f'p{i}_r'
        col_fdp = f'fdp_p{i}'

        P = df_merged[col_a].values
        Q = df_merged[col_r].values

        # Vectorizado con numpy.where para manejar división por cero
        df_merged[col_fdp] = np.where(
            P == 0,
            np.where(Q == 0, 1.0, 0.0),  # Casos especiales
            np.cos(np.arctan(Q / P))      # Cálculo normal
        )

    # Seleccionar solo columnas FDP y redondear
    cols_fdp = [f'fdp_p{i}' for i in range(1, 25)]
    out_cols = on_cols + cols_fdp
    return df_merged[out_cols].round(PRECISION_DECIMALES)


# =============================================================================
# FUNCIONES AUXILIARES - CONVERSIÓN
# =============================================================================

def _df_to_response(df: pd.DataFrame, tipo_dia: str, ajuste: float = None) -> Dict[str, Any]:
    """
    Convierte DataFrame a estructura de respuesta JSON.

    Args:
        df: DataFrame con resultados
        tipo_dia: ORDINARIO, SABADO o FESTIVO
        ajuste: Ajuste aplicado (opcional, solo para FDA)

    Returns:
        Diccionario con estructura de respuesta
    """
    response = {
        "tipo_dia": tipo_dia,
        "n_registros": len(df),
        "factores": df.to_dict('index'),
        "suma_total": round(df[PERIODOS_COLUMNAS].sum().sum(), PRECISION_DECIMALES)
    }

    if ajuste is not None:
        response["ajuste_aplicado"] = round(ajuste, PRECISION_DECIMALES)

    return response


# =============================================================================
# FUNCIONES PRINCIPALES - CLUSTERING
# =============================================================================

def aplicar_clustering(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    barra: str,
    flujo_tipo: str,
    tipo_dia: str = "",
    dsn: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Aplica factores multiplicadores a medidas y agrupa por barra+fecha.

    Este es el "clustering" que multiplica cada medida por su factor correspondiente
    y luego agrupa sumando los periodos.

    Args:
        fecha_inicial: Formato YYYY-MM-DD
        fecha_final: Formato YYYY-MM-DD
        mc: Código de mercado/centro
        barra: Nombre de la barra
        flujo_tipo: "A" (Activa) o "R" (Reactiva)
        tipo_dia: ORDINARIO, SABADO, FESTIVO o vacío para todos

    Returns:
        Lista de medidas clusterizadas por fecha
    """
    # Obtener códigos RPM de la barra
    codigos = factores_service.consultar_barra_nombre(barra, dsn=dsn)
    if not codigos:
        return []

    codigo_rpm = [row['codigo_rpm'] for row in codigos]

    # Obtener factores
    factores = factores_service.consultar_barra_factor_nombre(barra, flujo_tipo, codigo_rpm, dsn=dsn)
    if not factores:
        return []

    # Obtener medidas
    flujos = [f['flujo'] for f in factores]
    medidas = factores_service.consultar_medidas_calcular_completo(
        fecha_inicial, fecha_final, mc, flujos, tipo_dia, codigo_rpm, barra, False, dsn=dsn
    )

    if not medidas:
        return []

    # Crear DataFrame y aplicar factores
    df = pd.DataFrame(medidas)

    # Crear diccionario de factores para lookup
    factor_map = {(f['codigo_rpm'], f['flujo']): float(f['factor']) for f in factores}

    # Multiplicar periodos por factores
    # Las columnas de la consulta son: mep1, mep2, ..., mep24
    for i in range(1, 25):
        col_medida = f'mep{i}'
        col_resultado = f'p{i}'
        if col_medida in df.columns:
            df[col_resultado] = df.apply(
                lambda row: round(
                    float(row[col_medida]) * factor_map.get((row['mecodigo_rpm'], row['meflujo']), 1.0),
                    PRECISION_DECIMALES
                ),
                axis=1
            )

    # Renombrar columnas para uniformidad
    df = df.rename(columns={'babarra': 'barra', 'mefecha': 'fecha'})

    # Agrupar por barra+fecha
    df_agrupado = _agrupar_medidas_clusterizadas(df)
    df_agrupado = df_agrupado.reset_index()

    # Convertir a lista de dicts
    resultado = []
    for _, row in df_agrupado.iterrows():
        periodos_dict = {f'p{i}': row[f'p{i}'] for i in range(1, 25)}
        resultado.append({
            'barra': row['barra'],
            'fecha': str(row['fecha']),
            'periodos': periodos_dict
        })

    return resultado


def obtener_curvas_tipicas(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    tipo_dia: str,
    flujo_tipo: str,
    n_max: int,
    barra: Optional[str] = None,
    dsn: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Obtiene las N curvas más típicas del histórico (por forma y nivel).

    1. Obtiene todas las curvas clusterizadas en el rango/MC/tipo_dia (una barra o todas las del MC).
    2. Normaliza por L2 y mide centralidad (distancia media a las demás).
    3. Devuelve hasta n_max más típicas; si el cluster solo encuentra menos, devuelve esas.

    Returns:
        Lista de {barra, fecha, periodos} con las curvas más típicas.
    """
    if flujo_tipo not in ("A", "R"):
        return []

    if barra:
        barras_a_usar = [{"barra": barra}]
    else:
        barras_a_usar = factores_service.consultar_barras_index_xmc(mc, dsn=dsn)
        if not barras_a_usar:
            return []

    curvas_todas = []
    for b in barras_a_usar:
        nombre_barra = b.get("barra")
        if not nombre_barra:
            continue
        medidas = aplicar_clustering(
            fecha_inicial, fecha_final, mc, nombre_barra, flujo_tipo, tipo_dia, dsn=dsn
        )
        for m in medidas:
            curvas_todas.append({
                "barra": m["barra"],
                "fecha": m["fecha"],
                "periodos": m["periodos"],
            })

    return _seleccionar_curvas_tipicas(curvas_todas, n_max)


# =============================================================================
# FUNCIONES PRINCIPALES - FDA
# =============================================================================

def _filtrar_medidas_por_curvas_tipicas(
    medidas: List[Dict[str, Any]],
    curvas_tipicas: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filtra medidas para conservar solo las (barra, fecha) que están en curvas_tipicas."""
    if not curvas_tipicas:
        return []
    set_ref = {(c["barra"], str(c["fecha"])) for c in curvas_tipicas}
    return [
        m for m in medidas
        if (m["barra"], str(m["fecha"])) in set_ref
    ]


def _obtener_medidas_clusterizadas_para_curvas_tipicas(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    tipo_dia: str,
    curvas_tipicas: List[Dict[str, Any]],
    flujo_tipo: str,
    dsn: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Obtiene medidas clusterizadas solo para las (barra, fecha) indicadas en curvas_tipicas.
    """
    if not curvas_tipicas:
        return []
    barras_unicas = list({c["barra"] for c in curvas_tipicas})
    todas = []
    for barra in barras_unicas:
        medidas = aplicar_clustering(
            fecha_inicial, fecha_final, mc, barra, flujo_tipo, tipo_dia, dsn=dsn
        )
        todas.extend(medidas)
    return _filtrar_medidas_por_curvas_tipicas(todas, curvas_tipicas)


def calcular_fda_para_tipo_dia(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    tipo_dia: str,
    curvas_tipicas: List[Dict[str, Any]],
    dsn: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calcula FDA (Factor de Demanda Ajustada) solo sobre las curvas típicas indicadas.

    El algoritmo FDA normaliza los factores para que la suma sea exactamente 1.0,
    aplicando el ajuste únicamente al valor máximo de cada periodo.

    Args:
        fecha_inicial: Formato YYYY-MM-DD
        fecha_final: Formato YYYY-MM-DD
        mc: Código de mercado/centro
        tipo_dia: ORDINARIO, SABADO o FESTIVO
        curvas_tipicas: Lista de {barra, fecha} (salida de curvas-tipicas). FDA se calcula solo sobre estas.
        dsn: URL de conexión a BD alternativa (opcional)

    Returns:
        Diccionario con factores FDA normalizados
    """
    medidas_clusterizadas = _obtener_medidas_clusterizadas_para_curvas_tipicas(
        fecha_inicial, fecha_final, mc, tipo_dia, curvas_tipicas, "A", dsn=dsn
    )

    if not medidas_clusterizadas:
        return {
            "tipo_dia": tipo_dia,
            "n_registros": 0,
            "factores": {},
            "suma_total": 0.0,
            "ajuste_aplicado": 0.0
        }

    # Convertir a DataFrame
    df_list = []
    for medida in medidas_clusterizadas:
        row_data = {'barra': medida['barra'], 'fecha': medida['fecha']}
        row_data.update(medida['periodos'])
        df_list.append(row_data)

    df = pd.DataFrame(df_list)

    # Aplicar normalización FDA
    df_normalizado = _calcular_fda_normalizado(df)

    # Calcular ajuste real aplicado (diferencia entre suma normalizada y 1.0)
    # Este valor debería ser muy cercano a 0 después de la normalización
    sumas_finales = df_normalizado[PERIODOS_COLUMNAS].sum()
    ajustes_reales = (1.0 - sumas_finales).abs()
    ajuste_promedio = ajustes_reales.mean()  # Promedio de ajustes por período

    # Agregar barra y fecha de vuelta
    df_normalizado['barra'] = df['barra'].values
    df_normalizado['fecha'] = df['fecha'].values

    # Convertir a respuesta
    return _df_to_response(df_normalizado, tipo_dia, ajuste_promedio)


# =============================================================================
# FUNCIONES PRINCIPALES - FDP
# =============================================================================

def calcular_fdp_para_tipo_dia(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    tipo_dia: str,
    curvas_tipicas: List[Dict[str, Any]],
    dsn: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calcula FDP (Factor de Demanda Pronóstico) solo sobre las curvas típicas indicadas.

    FDP = Cos(Atan(Potencia_Reactiva / Potencia_Activa))

    Requiere medidas tanto de tipo A (activa) como R (reactiva) para esas curvas.

    Args:
        fecha_inicial: Formato YYYY-MM-DD
        fecha_final: Formato YYYY-MM-DD
        mc: Código de mercado/centro
        tipo_dia: ORDINARIO, SABADO o FESTIVO
        curvas_tipicas: Lista de {barra, fecha}. FDP se calcula solo sobre estas.
        dsn: URL de conexión a BD alternativa (opcional)

    Returns:
        Diccionario con factores FDP calculados
    """
    medidas_a = _obtener_medidas_clusterizadas_para_curvas_tipicas(
        fecha_inicial, fecha_final, mc, tipo_dia, curvas_tipicas, "A", dsn=dsn
    )
    medidas_r = _obtener_medidas_clusterizadas_para_curvas_tipicas(
        fecha_inicial, fecha_final, mc, tipo_dia, curvas_tipicas, "R", dsn=dsn
    )

    if not medidas_a or not medidas_r:
        return {
            "tipo_dia": tipo_dia,
            "n_registros": 0,
            "factores": {}
        }

    # Convertir a DataFrames (incluir barra para merge correcto con varias barras)
    df_a_list = []
    for m in medidas_a:
        row = {'barra': m['barra'], 'fecha': m['fecha']}
        row.update(m['periodos'])
        df_a_list.append(row)
    df_a = pd.DataFrame(df_a_list)

    df_r_list = []
    for m in medidas_r:
        row = {'barra': m['barra'], 'fecha': m['fecha']}
        row.update(m['periodos'])
        df_r_list.append(row)
    df_r = pd.DataFrame(df_r_list)

    # Calcular FDP vectorizado
    df_fdp = _calcular_fdp_vectorizado(df_a, df_r)

    # Convertir columnas fdp_p* a p*
    for i in range(1, 25):
        df_fdp[f'p{i}'] = df_fdp[f'fdp_p{i}']
        df_fdp = df_fdp.drop(columns=[f'fdp_p{i}'])

    # Convertir a respuesta
    return _df_to_response(df_fdp, tipo_dia)
