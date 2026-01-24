"""
Pipeline de Predicción para los Próximos 30 Días
================================================

Este script genera predicciones automáticas para los próximos 30 días usando:
1. Datos históricos hasta ayer
2. Pronóstico del clima (30 días)
3. Calendario de festivos
4. Feature engineering automático (reutiliza pipeline Semana 1)
5. Predicción recursiva día por día

Uso:
    python predict_next_30_days.py

Output:
    - predictions/predictions_next_30_days.csv
    - predictions/predictions_summary.json
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sys

# Importar pipeline de feature engineering
try:
    from ..pipeline.feature_engineering import FeatureEngineer
except ImportError:
    # Fallback para ejecución directa
    sys.path.append(str(Path(__file__).parent.parent))
    from pipeline.feature_engineering import FeatureEngineer

# Importar sistema de desagregación horaria
try:
    from .hourly import HourlyDisaggregationEngine
except ImportError:
    # Si no está disponible, será None
    HourlyDisaggregationEngine = None

# Importar cliente de festivos API
try:
    from .festivos_api import FestivosAPIClient
except ImportError:
    FestivosAPIClient = None

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ForecastPipeline:
    """
    Pipeline completo para predicción de próximos 30 días
    
    IMPORTANTE: El caché de festivos es solo por instancia (en memoria).
    Cada ejecución nueva crea una nueva instancia, por lo que siempre se llama
    a la API al menos una vez, garantizando que los cambios en la API se reflejen.
    """

    def __init__(self,
                 model_path: str = 'models/registry/champion_model.joblib',
                 historical_data_path: str = 'data/features/data_with_features_latest.csv',
                 festivos_path: str = None,  # Deprecated: ya no se usa, se mantiene por compatibilidad
                 enable_hourly_disaggregation: bool = True,
                 raw_climate_path: str = 'data/raw/clima_new.csv',
                 ucp: str = None):
        """
        Inicializa el pipeline

        Args:
            model_path: Ruta al modelo entrenado
            historical_data_path: Ruta a datos históricos con features
            festivos_path: DEPRECATED - Ya no se usa, los festivos se obtienen desde la API
            enable_hourly_disaggregation: Si True, habilita desagregación horaria automática
            raw_climate_path: Ruta a datos climáticos RAW (para obtener datos reales más allá del entrenamiento)
            ucp: Nombre del UCP (ej: 'Antioquia', 'Atlantico', 'Oriente') - REQUERIDO para obtener festivos desde API
        """
        self.model_path = model_path
        self.historical_data_path = historical_data_path
        self.festivos_path = festivos_path  # Mantenido por compatibilidad, pero no se usa
        self.enable_hourly_disaggregation = enable_hourly_disaggregation
        self.raw_climate_path = raw_climate_path
        self.ucp = ucp or 'Antioquia'  # Default a Antioquia si no se especifica
        
        # Inicializar cliente de festivos API
        if FestivosAPIClient is None:
            logger.warning("⚠ FestivosAPIClient no disponible. Los festivos no funcionarán correctamente.")
            self.festivos_client = None
        else:
            self.festivos_client = FestivosAPIClient()

        # Cargar modelo
        logger.info(f"Cargando modelo desde {model_path}")
        model_dict = joblib.load(model_path)
        self.model = model_dict['model'] if isinstance(model_dict, dict) else model_dict
        self.feature_names = model_dict.get('feature_names', None)

        # Cargar datos históricos
        logger.info(f"Cargando datos históricos desde {historical_data_path}")
        self.df_historico = pd.read_csv(historical_data_path)

        # Normalizar nombres de columnas (FECHA -> fecha, TOTAL -> demanda_total)
        if 'FECHA' in self.df_historico.columns:
            self.df_historico.rename(columns={'FECHA': 'fecha'}, inplace=True)
        elif 'fecha' not in self.df_historico.columns:
            # Si no hay columna de fecha, crear una basándose en year, month, day
            if all(col in self.df_historico.columns for col in ['year', 'month', 'day']):
                self.df_historico['fecha'] = pd.to_datetime(
                    self.df_historico[['year', 'month', 'day']].rename(columns={
                        'year': 'year', 'month': 'month', 'day': 'day'
                    })
                )
            else:
                # Último recurso: crear fechas basadas en el índice
                # Asumiendo que los datos empiezan en 2017-01-01
                logger.warning("No se encontró columna de fecha, creando fechas desde índice")
                self.df_historico['fecha'] = pd.date_range(start='2017-01-01', periods=len(self.df_historico), freq='D')

        if 'TOTAL' in self.df_historico.columns:
            self.df_historico.rename(columns={'TOTAL': 'demanda_total'}, inplace=True)

        if not pd.api.types.is_datetime64_any_dtype(self.df_historico['fecha']):
            self.df_historico['fecha'] = pd.to_datetime(self.df_historico['fecha'])

        # Cargar datos climáticos RAW (para obtener datos más allá del entrenamiento)
        self.df_climate_raw = None
        if Path(self.raw_climate_path).exists():
            try:
                logger.info(f"Cargando datos climáticos RAW desde {self.raw_climate_path}")
                from ..pipeline.connectors import WeatherDataConnector
                weather_connector = WeatherDataConnector({'path': self.raw_climate_path})
                # CRÍTICO: NO pasar filtros de fecha para obtener TODOS los datos disponibles
                self.df_climate_raw = weather_connector.read_data(start_date=None, end_date=None)
                if 'FECHA' in self.df_climate_raw.columns:
                    self.df_climate_raw.rename(columns={'FECHA': 'fecha'}, inplace=True)
                if not pd.api.types.is_datetime64_any_dtype(self.df_climate_raw['fecha']):
                    self.df_climate_raw['fecha'] = pd.to_datetime(self.df_climate_raw['fecha'])
                logger.info(f"✅ Datos climaticos RAW cargados: {len(self.df_climate_raw)} dias ({self.df_climate_raw['fecha'].min()} a {self.df_climate_raw['fecha'].max()})")
                logger.info(f"   Columnas disponibles: {list(self.df_climate_raw.columns)[:15]}...")
                logger.info(f"   Shape: {self.df_climate_raw.shape}")
            except Exception as e:
                logger.warning(f"⚠️ ALERTA No se pudieron cargar datos climaticos RAW: {e}")
                logger.warning(f"   Se usarán promedios históricos como fallback")
                self.df_climate_raw = None

        # Inicializar conjunto de festivos (se carga bajo demanda en predict_next_n_days)
        self.festivos = set()  # Set para búsquedas rápidas
        self.festivos_loaded_range = None  # Rango de fechas para el cual se cargaron los festivos

        # Feature engineer
        self.feature_engineer = FeatureEngineer()

        # Sistema de desagregación horaria
        if self.enable_hourly_disaggregation and HourlyDisaggregationEngine is not None:
            logger.info(f"Inicializando sistema de desagregación horaria{' para ' + self.ucp if self.ucp else ''}...")
            try:
                # Cargar modelos desde directorio específico del UCP si está disponible
                if self.ucp:
                    models_dir = f'models/{self.ucp}'
                    self.hourly_engine = HourlyDisaggregationEngine(
                        auto_load=True,
                        models_dir=models_dir,
                        ucp=self.ucp,
                        historical_data_path=self.historical_data_path  # Pasar ruta de datos para auto-reentrenamiento
                    )
                    logger.info(f"✓ Sistema de desagregación horaria cargado para {self.ucp}")
                else:
                    self.hourly_engine = HourlyDisaggregationEngine(
                        auto_load=True, 
                        ucp=self.ucp,
                        historical_data_path=self.historical_data_path  # Pasar ruta de datos para auto-reentrenamiento
                    )
                    logger.info("✓ Sistema de desagregación horaria cargado")
            except Exception as e:
                logger.warning(f"⚠ No se pudo cargar sistema de desagregación: {e}")
                logger.warning("  Se usarán placeholders para distribución horaria")
                self.hourly_engine = None
        else:
            if self.enable_hourly_disaggregation:
                logger.warning("⚠ HourlyDisaggregationEngine no disponible. Se usarán placeholders.")
            self.hourly_engine = None

        logger.info("✓ Pipeline inicializado correctamente")

    def _load_festivos_for_range(self, start_date: datetime, end_date: datetime) -> None:
        """
        Carga festivos desde la API para un rango de fechas específico.
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
        """
        if self.festivos_client is None:
            logger.error("⚠ Cliente de festivos API no disponible. No se pueden cargar festivos.")
            return
        
        # Verificar si ya tenemos los festivos cargados para este rango
        if (self.festivos_loaded_range and 
            start_date >= self.festivos_loaded_range[0] and 
            end_date <= self.festivos_loaded_range[1]):
            logger.debug(f"Festivos ya cargados para el rango solicitado")
            return
        
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            logger.info(f"Cargando festivos desde API para {self.ucp} ({start_str} a {end_str})...")
            festivos_list = self.festivos_client.get_festivos(start_str, end_str, self.ucp)
            
            # Agregar al set existente (o reemplazar si es un rango completamente nuevo)
            if self.festivos_loaded_range is None:
                self.festivos = set(festivos_list)
            else:
                # Extender el set con nuevos festivos
                self.festivos.update(festivos_list)
            
            # Actualizar el rango cargado (extender si es necesario)
            if self.festivos_loaded_range is None:
                self.festivos_loaded_range = (start_date, end_date)
            else:
                self.festivos_loaded_range = (
                    min(self.festivos_loaded_range[0], start_date),
                    max(self.festivos_loaded_range[1], end_date)
                )
            
            logger.info(f"✓ Cargados {len(festivos_list)} festivos (total en caché: {len(self.festivos)})")
            
        except Exception as e:
            logger.error(f"❌ Error al cargar festivos desde API: {e}")
            logger.warning("  Continuando sin festivos (se asumirá que no hay festivos)")

    def is_festivo(self, fecha: datetime) -> bool:
        """
        Verifica si una fecha es festivo.
        
        Nota: Los festivos deberían estar pre-cargados con _load_festivos_for_range().
        Si no están, intenta cargarlos automáticamente (pero esto es ineficiente).
        """
        fecha_str = fecha.strftime('%Y-%m-%d')
        
        # Verificar si ya está en el set (caso más común - optimizado)
        if fecha_str in self.festivos:
            return True
        
        # Si no está en el caché, verificar si el rango cargado incluye esta fecha
        # Si no, cargar el año completo (solo como fallback)
        if self.festivos_client is not None:
            # Verificar si la fecha está fuera del rango cargado
            if (self.festivos_loaded_range is None or 
                fecha < self.festivos_loaded_range[0] or 
                fecha > self.festivos_loaded_range[1]):
                # Cargar el año de la fecha como fallback
                year = fecha.year
                start_date = datetime(year, 1, 1)
                end_date = datetime(year, 12, 31)
                self._load_festivos_for_range(start_date, end_date)
            
            # Verificar de nuevo después de cargar (o si ya estaba en rango pero no en set)
            return fecha_str in self.festivos
        
        # Si no hay cliente disponible, retornar False
        return False

    def _get_placeholder_hourly(self, total_daily: float) -> dict:
        """
        Genera distribución horaria usando placeholders simples.

        Se usa como fallback cuando el sistema de clustering no está disponible.

        Args:
            total_daily: Demanda total del día

        Returns:
            Dict con P1-P24
        """
        # Perfil aproximado basado en patrones típicos de demanda
        # Horas pico: 6-9am y 6-9pm
        hourly_distribution = [
            0.038,  # P1  (00:00-01:00) - madrugada
            0.036,  # P2  (01:00-02:00)
            0.034,  # P3  (02:00-03:00)
            0.035,  # P4  (03:00-04:00)
            0.037,  # P5  (04:00-05:00)
            0.040,  # P6  (05:00-06:00) - empieza a subir
            0.042,  # P7  (06:00-07:00) - pico mañana
            0.044,  # P8  (07:00-08:00)
            0.045,  # P9  (08:00-09:00)
            0.044,  # P10 (09:00-10:00)
            0.043,  # P11 (10:00-11:00)
            0.042,  # P12 (11:00-12:00)
            0.041,  # P13 (12:00-13:00)
            0.040,  # P14 (13:00-14:00)
            0.041,  # P15 (14:00-15:00)
            0.042,  # P16 (15:00-16:00)
            0.043,  # P17 (16:00-17:00)
            0.045,  # P18 (17:00-18:00) - empieza pico tarde
            0.047,  # P19 (18:00-19:00) - pico tarde
            0.048,  # P20 (19:00-20:00)
            0.046,  # P21 (20:00-21:00)
            0.044,  # P22 (21:00-22:00)
            0.041,  # P23 (22:00-23:00)
            0.039,  # P24 (23:00-00:00)
        ]

        # Normalizar para que sume exactamente 1.0
        total_dist = sum(hourly_distribution)
        hourly_distribution = [x / total_dist for x in hourly_distribution]

        return {f'P{i}': total_daily * hourly_distribution[i-1] for i in range(1, 25)}

    def get_real_climate_data(self, start_date: datetime, days: int = 30) -> pd.DataFrame:
        """
        Intenta obtener datos climáticos REALES, primero de RAW luego de histórico

        Args:
            start_date: Fecha inicial
            days: Número de días

        Returns:
            DataFrame con datos reales o None si no existen
        """
        end_date = start_date + timedelta(days=days - 1)

        # PRIMERO: Intentar desde datos climáticos RAW (tienen datos más recientes)
        if self.df_climate_raw is not None:
            df_climate_raw_filtered = self.df_climate_raw[
                (self.df_climate_raw['fecha'] >= start_date) &
                (self.df_climate_raw['fecha'] <= end_date)
            ].copy()

            logger.info(f"   Buscando datos climáticos RAW: {start_date.date()} a {end_date.date()}")
            logger.info(f"   Registros encontrados en RAW: {len(df_climate_raw_filtered)}/{days} días")

            # CAMBIO: Usar datos disponibles aunque sean menos de 'days' (complementaremos con promedios)
            if len(df_climate_raw_filtered) > 0:
                # Tenemos al menos algunos datos en RAW
                # CAMBIO: Buscar columnas de forma más flexible
                all_cols = df_climate_raw_filtered.columns.tolist()
                
                # Filtrar columnas climáticas (API EPM: temp, humidity, wind_speed, rain)
                # Excluir columnas con _lag, _x_ (son transformaciones)
                clima_cols = [col for col in all_cols if
                             any(keyword in col.lower() for keyword in
                                 ['temp', 'humidity', 'wind_speed', 'rain']) and
                             '_lag' not in col and 
                             '_x_' not in col and
                             col != 'fecha']

                logger.info(f"   Columnas climáticas disponibles en RAW: {len(clima_cols)}")
                if len(clima_cols) > 0:
                    logger.info(f"   Primeras columnas: {clima_cols[:5]}")

                if len(clima_cols) > 0:
                    df_result = df_climate_raw_filtered[['fecha'] + clima_cols].copy()
                    if len(df_result) >= days:
                        logger.info(f"✅ Usando datos climáticos REALES de archivo RAW para {len(df_result)} días (completos)")
                    else:
                        logger.info(f"✅ Usando datos climáticos REALES de archivo RAW para {len(df_result)}/{days} días (parciales, se complementarán con promedios)")
                    return df_result
                else:
                    logger.warning(f"⚠️ No se encontraron columnas climáticas válidas en RAW")
                    logger.warning(f"   Columnas disponibles: {all_cols[:10]}...")

        # SEGUNDO: Buscar en histórico (features file)
        # Verificar si tenemos columnas climáticas en el histórico (incluyendo lag)
        # Buscar columnas base (sin _lag1d) primero - SOLO variables API EPM
        climate_cols_base = [col for col in self.df_historico.columns if
                            any(x in col for x in ['temp_mean', 'temp_min', 'temp_max', 'temp_std',
                                                   'humidity_mean', 'humidity_min', 'humidity_max',
                                                   'wind_speed_mean', 'wind_speed_max',
                                                   'rain_mean', 'rain_sum'])
                            and '_lag' not in col and '_x_' not in col]

        # Si no hay columnas base, buscar lag
        climate_cols_lag = [col for col in self.df_historico.columns if
                           col.endswith('_lag1d') and
                           any(x in col for x in ['temp_', 'humidity_', 'wind_speed_', 'rain_'])]

        if len(climate_cols_base) == 0 and len(climate_cols_lag) == 0:
            logger.warning(f"⚠ No se encontraron datos climáticos en el histórico")
            return None

        # Filtrar datos históricos para ese rango
        df_climate = self.df_historico[
            (self.df_historico['fecha'] >= start_date) &
            (self.df_historico['fecha'] <= end_date)
        ].copy()

        if len(df_climate) < days:
            # No hay suficientes datos reales
            logger.warning(f"⚠ Solo se encontraron {len(df_climate)}/{days} días de datos climáticos reales (necesarios: {days})")
            logger.warning(f"  Rango disponible: {self.df_historico['fecha'].min()} a {self.df_historico['fecha'].max()}")
            logger.warning(f"  Rango solicitado: {start_date} a {end_date}")
            return None

        # Usar columnas base si existen, si no usar lag (y renombrar quitando _lag1d)
        if len(climate_cols_base) > 0:
            df_result = df_climate[['fecha'] + climate_cols_base].copy()
            logger.info(f"✅ Usando datos climáticos REALES de features file para {len(df_result)} días (columnas base)")
        else:
            df_result = df_climate[['fecha'] + climate_cols_lag].copy()
            # Renombrar quitando _lag1d
            rename_map = {col: col.replace('_lag1d', '') for col in climate_cols_lag}
            df_result = df_result.rename(columns=rename_map)
            logger.info(f"✅ Usando datos climáticos REALES de features file para {len(df_result)} días (columnas lag)")

        return df_result

    def generate_climate_forecast(self, start_date: datetime, days: int = 30) -> pd.DataFrame:
        """
        Genera pronóstico del clima para los próximos N días

        PRIORIDAD:
        1. Intenta usar datos climáticos REALES del histórico si existen
        2. Si hay datos parciales, los complementa con promedios históricos
        3. Si no hay datos, usa PROMEDIOS HISTÓRICOS como fallback

        Args:
            start_date: Fecha inicial
            days: Número de días a pronosticar

        Returns:
            DataFrame con pronóstico del clima
        """
        logger.info(f"Generando pronóstico del clima para {days} días...")

        # PRIMERO: Intentar usar datos reales
        real_data = self.get_real_climate_data(start_date, days)
        if real_data is not None:
            # Verificar si tenemos todos los días o solo parciales
            if len(real_data) >= days:
                return real_data
            else:
                # Complementar datos parciales con promedios históricos
                logger.info(f"   Complementando {len(real_data)} días reales con promedios históricos para {days - len(real_data)} días faltantes")
                return self._complement_with_historical_averages(real_data, start_date, days)

        # FALLBACK: Usar promedios históricos completos
        logger.warning("⚠️  Usando promedios históricos (fallback). Integrar API de clima para producción.")

        # Calcular promedios históricos por mes/día del año
        climate_stats = self._calculate_historical_climate_stats()

        forecasts = []
        for day in range(days):
            fecha = start_date + timedelta(days=day)
            month = fecha.month
            dayofyear = fecha.timetuple().tm_yday

            # Usar estadísticas del mismo mes
            stats = climate_stats[month]

            # Añadir variación estocástica pequeña
            np.random.seed(int(fecha.timestamp()))

            forecasts.append({
                'fecha': fecha,
                'temp_mean': stats['temp_mean'] + np.random.normal(0, 1),
                'temp_min': stats['temp_min'] + np.random.normal(0, 0.5),
                'temp_max': stats['temp_max'] + np.random.normal(0, 0.5),
                'temp_std': stats['temp_std'],
                'humidity_mean': stats['humidity_mean'] + np.random.normal(0, 2),
                'humidity_min': stats['humidity_min'],
                'humidity_max': stats['humidity_max'],
                'wind_speed_mean': stats.get('wind_speed_mean', 2.0),
                'wind_speed_max': stats.get('wind_speed_max', 5.0),
                'rain_mean': stats.get('rain_mean', 0.0),
                'rain_sum': stats.get('rain_sum', 0.0)
            })

        return pd.DataFrame(forecasts)

    def _complement_with_historical_averages(self, real_data: pd.DataFrame, start_date: datetime, days: int) -> pd.DataFrame:
        """
        Complementa datos climáticos reales parciales con promedios históricos para días faltantes
        
        Args:
            real_data: DataFrame con datos reales (puede tener menos de 'days' registros)
            start_date: Fecha inicial del período completo
            days: Número total de días necesarios
            
        Returns:
            DataFrame completo con datos reales + promedios históricos
        """
        # Obtener fechas que ya tenemos en datos reales
        real_dates = set(real_data['fecha'].dt.date)
        
        # Calcular promedios históricos
        climate_stats = self._calculate_historical_climate_stats()
        
        # Crear lista completa de fechas
        all_dates = [start_date + timedelta(days=day) for day in range(days)]
        
        # Construir DataFrame completo
        complete_forecasts = []
        for fecha in all_dates:
            fecha_date = fecha.date()
            
            # Si tenemos datos reales para esta fecha, usarlos
            if fecha_date in real_dates:
                real_row = real_data[real_data['fecha'].dt.date == fecha_date].iloc[0]
                complete_forecasts.append(real_row.to_dict())
            else:
                # Usar promedios históricos para días faltantes
                month = fecha.month
                stats = climate_stats[month]
                
                np.random.seed(int(fecha.timestamp()))
                
                complete_forecasts.append({
                    'fecha': fecha,
                    'temp_mean': stats['temp_mean'] + np.random.normal(0, 1),
                    'temp_min': stats['temp_min'] + np.random.normal(0, 0.5),
                    'temp_max': stats['temp_max'] + np.random.normal(0, 0.5),
                    'temp_std': stats['temp_std'],
                    'humidity_mean': stats['humidity_mean'] + np.random.normal(0, 2),
                    'humidity_min': stats['humidity_min'],
                    'humidity_max': stats['humidity_max'],
                    'wind_speed_mean': stats.get('wind_speed_mean', 2.0),
                    'wind_speed_max': stats.get('wind_speed_max', 5.0),
                    'rain_mean': stats.get('rain_mean', 0.0),
                    'rain_sum': stats.get('rain_sum', 0.0)
                })
        
        result_df = pd.DataFrame(complete_forecasts)
        
        # Asegurar que las columnas de fecha sean datetime
        if not pd.api.types.is_datetime64_any_dtype(result_df['fecha']):
            result_df['fecha'] = pd.to_datetime(result_df['fecha'])
        
        logger.info(f"✅ Pronóstico completo: {len(real_data)} días reales + {days - len(real_data)} días con promedios históricos")
        
        return result_df

    def _calculate_historical_climate_stats(self) -> dict:
        """Calcula estadísticas climáticas históricas por mes"""
        # Verificar si tenemos datos climáticos en el histórico
        climate_cols = [col for col in self.df_historico.columns if 'temp' in col or 'humidity' in col or 'wind_speed' in col or 'rain' in col]

        if not climate_cols:
            logger.warning("No hay datos climáticos en el histórico. Usando valores por defecto.")
            return self._get_default_climate_stats()

        # Calcular por mes
        self.df_historico['month'] = pd.to_datetime(self.df_historico['fecha']).dt.month
        stats = {}

        for month in range(1, 13):
            month_data = self.df_historico[self.df_historico['month'] == month]

            if len(month_data) == 0:
                stats[month] = self._get_default_climate_stats()[month]
                continue

            # Extraer valores actuales (quitando _lag1d del nombre si existe)
            def get_base_value(df, prefix):
                """Obtiene valor base de columna con lag"""
                lag_col = f'{prefix}_lag1d'
                if lag_col in df.columns:
                    return df[lag_col].mean()
                elif prefix in df.columns:
                    return df[prefix].mean()
                return None

            stats[month] = {
                'temp_mean': get_base_value(month_data, 'temp_mean') or 22.0,
                'temp_min': get_base_value(month_data, 'temp_min') or 16.0,
                'temp_max': get_base_value(month_data, 'temp_max') or 28.0,
                'temp_std': 2.5,
                'humidity_mean': get_base_value(month_data, 'humidity_mean') or 70.0,
                'humidity_min': get_base_value(month_data, 'humidity_min') or 50.0,
                'humidity_max': get_base_value(month_data, 'humidity_max') or 90.0,
                'wind_speed_mean': get_base_value(month_data, 'wind_speed_mean') or 2.0,
                'wind_speed_max': get_base_value(month_data, 'wind_speed_max') or 5.0,
                'rain_mean': get_base_value(month_data, 'rain_mean') or 0.0,
                'rain_sum': get_base_value(month_data, 'rain_sum') or 0.0
            }

        return stats

    def _get_default_climate_stats(self) -> dict:
        """Estadísticas climáticas por defecto para Medellín"""
        # Medellín/Antioquia: Clima templado, poca variación anual (API EPM)
        base_stats = {
            'temp_mean': 22.0, 'temp_min': 16.0, 'temp_max': 28.0, 'temp_std': 2.5,
            'humidity_mean': 70.0, 'humidity_min': 50.0, 'humidity_max': 90.0,
            'wind_speed_mean': 2.0, 'wind_speed_max': 5.0,
            'rain_mean': 0.5, 'rain_sum': 2.0
        }

        # Ajustes leves por mes (temporada de lluvias: Abril-Mayo, Octubre-Noviembre)
        adjustments = {
            1: 0, 2: 0, 3: 0.5, 4: 1, 5: 1, 6: 0.5,
            7: 0, 8: 0, 9: 0.5, 10: 1, 11: 1, 12: 0
        }

        stats = {}
        for month in range(1, 13):
            adj = adjustments[month]
            stats[month] = {
                'temp_mean': base_stats['temp_mean'] - adj,
                'temp_min': base_stats['temp_min'] - adj,
                'temp_max': base_stats['temp_max'] - adj * 0.5,
                'temp_std': base_stats['temp_std'],
                'humidity_mean': base_stats['humidity_mean'] + adj * 3,
                'humidity_min': base_stats['humidity_min'],
                'humidity_max': base_stats['humidity_max'],
                'wind_speed_mean': base_stats['wind_speed_mean'] + adj * 0.5,
                'wind_speed_max': base_stats['wind_speed_max'] + adj,
                'rain_mean': base_stats['rain_mean'] + adj * 2,
                'rain_sum': base_stats['rain_sum'] + adj * 5
            }

        return stats

    def build_features_for_date(self,
                                fecha: datetime,
                                climate_forecast: dict,
                                df_temp: pd.DataFrame,
                                ultimo_historico: datetime = None) -> dict:
        """
        Construye todas las features necesarias para una fecha específica

        Args:
            fecha: Fecha a predecir
            climate_forecast: Pronóstico del clima para esa fecha
            df_temp: DataFrame temporal con histórico + predicciones previas
            ultimo_historico: Ultimo dia con datos reales (para filtrar predicciones en rolling stats)

        Returns:
            Diccionario con todas las features
        """
        features = {}

        # ========================================
        # A. FEATURES TEMPORALES (del calendario)
        # ========================================
        features['year'] = fecha.year
        features['month'] = fecha.month
        features['day'] = fecha.day
        features['dayofweek'] = fecha.dayofweek
        features['dayofyear'] = fecha.timetuple().tm_yday
        features['week'] = fecha.isocalendar()[1]
        features['quarter'] = (fecha.month - 1) // 3 + 1
        features['is_weekend'] = int(fecha.dayofweek >= 5)
        features['is_saturday'] = int(fecha.dayofweek == 5)
        features['is_sunday'] = int(fecha.dayofweek == 6)
        features['is_month_start'] = int(fecha.day == 1)
        features['is_month_end'] = int(fecha.day == pd.Timestamp(fecha).days_in_month)
        features['is_quarter_start'] = int(fecha.month in [1, 4, 7, 10] and fecha.day == 1)
        features['is_quarter_end'] = int(fecha.month in [3, 6, 9, 12] and fecha.day == pd.Timestamp(fecha).days_in_month)
        features['is_festivo'] = int(self.is_festivo(fecha))
        features['is_rainy_season'] = int(fecha.month in [4, 5, 10, 11])
        features['is_january'] = int(fecha.month == 1)
        features['is_december'] = int(fecha.month == 12)
        features['week_of_month'] = (fecha.day - 1) // 7 + 1

        # ========================================
        # NUEVAS FEATURES: Contexto de Época Especial
        # ========================================
        # Feature 1: Flag de época navideña (dic 23 - ene 6)
        features['es_epoca_navidena'] = int(
            (fecha.month == 12 and fecha.day >= 23) or
            (fecha.month == 1 and fecha.day <= 6)
        )

        # Feature 2: Flag de post-época especial (ene 7-15)
        features['es_post_epoca_especial'] = int(
            fecha.month == 1 and 7 <= fecha.day <= 15
        )

        # Feature 3: Días desde último festivo (buscar en df_temp)
        # Buscar festivos anteriores en df_temp
        if 'is_festivo' in df_temp.columns:
            festivos_anteriores = df_temp[
                (df_temp['fecha'] < pd.Timestamp(fecha)) &
                (df_temp['is_festivo'] == 1)
            ]
            if len(festivos_anteriores) > 0:
                ultimo_festivo = festivos_anteriores['fecha'].max()
                dias_desde_festivo = (pd.Timestamp(fecha) - ultimo_festivo).days
                features['dias_desde_ultimo_festivo'] = min(dias_desde_festivo, 30)  # Saturar en 30
            else:
                # Si no hay festivos anteriores, usar 30
                features['dias_desde_ultimo_festivo'] = 30
        else:
            # Si no existe columna is_festivo, usar 30
            features['dias_desde_ultimo_festivo'] = 30

        # Feature 4: Flag de temporada alta
        features['es_temporada_alta'] = int(
            not ((fecha.month == 12 and fecha.day >= 23) or
                 (fecha.month == 1 and fecha.day <= 6))
        )

        # Estacionales (sin/cos)
        features['dayofweek_sin'] = np.sin(2 * np.pi * fecha.dayofweek / 7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * fecha.dayofweek / 7)
        features['month_sin'] = np.sin(2 * np.pi * (fecha.month - 1) / 12)
        features['month_cos'] = np.cos(2 * np.pi * (fecha.month - 1) / 12)
        features['dayofyear_sin'] = np.sin(2 * np.pi * features['dayofyear'] / 365)
        features['dayofyear_cos'] = np.cos(2 * np.pi * features['dayofyear'] / 365)

        # ========================================
        # B. FEATURES CLIMÁTICAS (SOLO API EPM: temp, humidity, wind_speed, rain)
        # ========================================
        # IMPORTANTE: Los nombres deben coincidir EXACTAMENTE con feature_engineering.py
        features['temp_lag1d'] = climate_forecast.get('temp_mean', climate_forecast.get('temp', 22.0))
        features['humidity_lag1d'] = climate_forecast.get('humidity_mean', climate_forecast.get('humidity', 70.0))
        features['wind_speed_lag1d'] = climate_forecast.get('wind_speed_mean', 2.0)
        features['rain_lag1d'] = climate_forecast.get('rain_sum', 0.0)

        # Feature derivada: día lluvioso (> 1mm de lluvia)
        features['is_rainy_day'] = int(features['rain_lag1d'] > 1.0)

        # ========================================
        # C. FEATURES DE LAG (demanda histórica)
        # ========================================
        # IMPORTANTE: Usar FECHAS absolutas en lugar de índices relativos
        # para evitar desalineación cuando hay gaps en los datos

        # Calcular fechas de los lags
        fecha_lag_1d = fecha - timedelta(days=1)
        fecha_lag_7d = fecha - timedelta(days=7)
        fecha_lag_14d = fecha - timedelta(days=14)

        # Buscar valores por FECHA en lugar de índice
        def get_value_by_date(df, target_date, column='demanda_total'):
            """Obtiene valor de una columna buscando por fecha exacta"""
            mask = df['fecha'].dt.date == target_date.date()
            if mask.any():
                return df.loc[mask, column].iloc[-1]  # Tomar último si hay duplicados
            else:
                # Fallback: buscar el día más cercano anterior
                df_antes = df[df['fecha'].dt.date < target_date.date()]
                if len(df_antes) > 0:
                    return df_antes.iloc[-1][column]
                else:
                    # Si no hay datos anteriores, usar el primer valor disponible
                    return df.iloc[0][column] if len(df) > 0 else 0

        # Lag 1 día
        features['total_lag_1d'] = get_value_by_date(df_temp, fecha_lag_1d, 'demanda_total')
        features['p8_lag_1d'] = get_value_by_date(df_temp, fecha_lag_1d, 'P8') if 'P8' in df_temp.columns else features['total_lag_1d'] * 0.042
        features['p12_lag_1d'] = get_value_by_date(df_temp, fecha_lag_1d, 'P12') if 'P12' in df_temp.columns else features['total_lag_1d'] * 0.046
        features['p18_lag_1d'] = get_value_by_date(df_temp, fecha_lag_1d, 'P18') if 'P18' in df_temp.columns else features['total_lag_1d'] * 0.048
        features['p20_lag_1d'] = get_value_by_date(df_temp, fecha_lag_1d, 'P20') if 'P20' in df_temp.columns else features['total_lag_1d'] * 0.045

        # Lag 7 días
        features['total_lag_7d'] = get_value_by_date(df_temp, fecha_lag_7d, 'demanda_total')
        features['p8_lag_7d'] = get_value_by_date(df_temp, fecha_lag_7d, 'P8') if 'P8' in df_temp.columns else features['total_lag_7d'] * 0.042
        features['p12_lag_7d'] = get_value_by_date(df_temp, fecha_lag_7d, 'P12') if 'P12' in df_temp.columns else features['total_lag_7d'] * 0.046
        features['p18_lag_7d'] = get_value_by_date(df_temp, fecha_lag_7d, 'P18') if 'P18' in df_temp.columns else features['total_lag_7d'] * 0.048
        features['p20_lag_7d'] = get_value_by_date(df_temp, fecha_lag_7d, 'P20') if 'P20' in df_temp.columns else features['total_lag_7d'] * 0.045

        # Lag 14 días
        features['total_lag_14d'] = get_value_by_date(df_temp, fecha_lag_14d, 'demanda_total')

        # ========================================
        # C.2. LAGS HISTÓRICOS ANUALES (mismo día del año en años anteriores)
        # ========================================
        # Útiles para festivos especiales y patrones estacionales específicos
        # Maneja años bisiestos correctamente usando pd.DateOffset
        from pandas import DateOffset
        
        for years_back in [1, 2, 3]:  # 1 año, 2 años, 3 años
            fecha_lag_anual = fecha - DateOffset(years=years_back)
            col_name = f'total_lag_{years_back}y'
            
            # Intentar buscar fecha exacta
            features[col_name] = get_value_by_date(df_temp, fecha_lag_anual, 'demanda_total')
            
            # Si no se encuentra (primeros años o falta de datos), usar fallback
            # Fallback: promedio de días del mismo mes-día en años anteriores disponibles
            if features[col_name] == 0:
                # Buscar días del mismo mes-día en años anteriores disponibles
                same_month_day = df_temp[
                    (df_temp['fecha'].dt.month == fecha.month) &
                    (df_temp['fecha'].dt.day == fecha.day) &
                    (df_temp['fecha'].dt.date < fecha.date())
                ]
                
                if len(same_month_day) > 0:
                    # Usar promedio de años anteriores del mismo día
                    features[col_name] = same_month_day['demanda_total'].mean()
                else:
                    # Último fallback: usar lag de 365 días (aproximado)
                    fecha_approx = fecha - timedelta(days=365 * years_back)
                    features[col_name] = get_value_by_date(df_temp, fecha_approx, 'demanda_total')
                    if features[col_name] == 0:
                        features[col_name] = features['total_lag_1d']  # Fallback final

        # ========================================
        # D. ROLLING STATISTICS
        # ========================================
        # IMPORTANTE: Usar FECHAS para definir ventanas, no índices

        def get_rolling_values(df, target_date, days_back, column='demanda_total', ultimo_historico=None):
            """
            Obtiene valores de una ventana de tiempo basada en fechas, excluyendo época navideña.

            CRITICO: Solo usa datos historicos REALES, NO predicciones.
            FILTRO: Excluye época navideña (dic 23 - ene 6) para evitar contaminación por días atípicos.
            EXPANSIÓN: Si no hay suficientes datos, expande progresivamente hacia atrás (14, 28, 56, 112, 224 días)
                      MANTENIENDO el filtro de Navidad, buscando datos pre-Navidad.

            Args:
                df: DataFrame con datos (historicos + predicciones)
                target_date: Fecha objetivo
                days_back: Dias hacia atras para la ventana inicial
                column: Columna a extraer
                ultimo_historico: Ultimo dia con datos reales (no predicciones)

            Returns:
                np.array: Valores históricos encontrados (excluyendo época navideña)
            """
            # FILTRO CRITICO: Solo datos historicos reales (nunca usar predicciones)
            if ultimo_historico is not None:
                # Usar ultimos N dias HISTORICOS disponibles (antes de ultimo_historico)
                fecha_fin = ultimo_historico  # Ultimo dia con datos reales
                fecha_inicio = fecha_fin - timedelta(days=days_back - 1)  # N dias hacia atras

                mask = (df['fecha'].dt.date >= fecha_inicio.date()) & (df['fecha'].dt.date <= fecha_fin.date())
            else:
                # Modo normal (sin filtro, para compatibilidad)
                fecha_inicio = target_date - timedelta(days=days_back)
                fecha_fin = target_date - timedelta(days=1)
                mask = (df['fecha'].dt.date >= fecha_inicio.date()) & (df['fecha'].dt.date <= fecha_fin.date())

            # NUEVO FILTRO: Excluir época navideña (dic 23 - ene 6) de rolling statistics
            # Esto previene que períodos de demanda anormalmente baja contaminen las predicciones
            epoca_navidena_mask = (
                ((df['fecha'].dt.month == 12) & (df['fecha'].dt.day >= 23)) |
                ((df['fecha'].dt.month == 1) & (df['fecha'].dt.day <= 6))
            )
            mask = mask & ~epoca_navidena_mask  # Excluir días de época navideña

            valores = df.loc[mask, column].values

            # Si después de excluir época navideña no quedan suficientes valores,
            # expandir la ventana hacia atrás MANTENIENDO el filtro (buscar datos pre-Navidad)
            max_expansiones = 5  # Prevenir loops infinitos (224 días máximo)
            expansion_actual = 0

            while len(valores) < max(3, days_back // 2) and expansion_actual < max_expansiones:
                # Expandir ventana progresivamente hacia atrás
                days_back_extended = days_back * (2 ** (expansion_actual + 1))  # 14, 28, 56, 112, 224 días
                fecha_inicio_extended = fecha_fin - timedelta(days=days_back_extended - 1)

                mask_extended = (df['fecha'].dt.date >= fecha_inicio_extended.date()) & (df['fecha'].dt.date <= fecha_fin.date())
                mask_extended = mask_extended & ~epoca_navidena_mask  # MANTENER filtro de Navidad

                valores = df.loc[mask_extended, column].values
                expansion_actual += 1

                if len(valores) >= max(3, days_back // 2):
                    logger.info(f"   ✓ Expandido a {days_back_extended} días para evitar época navideña ({len(valores)} valores encontrados)")
                    break

            # Si después de todas las expansiones NO hay datos (dataset muy corto),
            # usar promedio histórico completo como fallback
            if len(valores) < 1:
                logger.warning(f"   ⚠️ No hay datos históricos suficientes después de filtrar época navideña.")
                # Usar todos los datos históricos disponibles (sin importar época navideña) como último recurso
                mask_fallback = (df['fecha'].dt.date <= fecha_fin.date())
                valores_fallback = df.loc[mask_fallback, column].values
                if len(valores_fallback) > 0:
                    # Tomar promedio de datos históricos completos
                    valores = np.array([np.mean(valores_fallback)])
                    logger.info(f"   → Usando promedio histórico: {valores[0]:.2f}")
                else:
                    # Caso extremo: no hay datos históricos en absoluto
                    valores = np.array([0])
                    logger.warning(f"   → No hay datos históricos. Usando 0 como fallback.")

            return valores

        # Últimos 7 días (SOLO DATOS HISTORICOS REALES)
        ultimos_7 = get_rolling_values(df_temp, fecha, 7, ultimo_historico=ultimo_historico)
        if len(ultimos_7) > 0:
            features['total_rolling_mean_7d'] = np.mean(ultimos_7)
            features['total_rolling_std_7d'] = np.std(ultimos_7) if len(ultimos_7) > 1 else 0
            features['total_rolling_min_7d'] = np.min(ultimos_7)
            features['total_rolling_max_7d'] = np.max(ultimos_7)
        else:
            features['total_rolling_mean_7d'] = features['total_lag_1d']
            features['total_rolling_std_7d'] = 0
            features['total_rolling_min_7d'] = features['total_lag_1d']
            features['total_rolling_max_7d'] = features['total_lag_1d']

        # Últimos 14 días (SOLO DATOS HISTORICOS REALES)
        ultimos_14 = get_rolling_values(df_temp, fecha, 14, ultimo_historico=ultimo_historico)
        if len(ultimos_14) > 0:
            features['total_rolling_mean_14d'] = np.mean(ultimos_14)
            features['total_rolling_std_14d'] = np.std(ultimos_14) if len(ultimos_14) > 1 else 0
            features['total_rolling_min_14d'] = np.min(ultimos_14)
            features['total_rolling_max_14d'] = np.max(ultimos_14)
        else:
            features['total_rolling_mean_14d'] = features['total_rolling_mean_7d']
            features['total_rolling_std_14d'] = features['total_rolling_std_7d']
            features['total_rolling_min_14d'] = features['total_rolling_min_7d']
            features['total_rolling_max_14d'] = features['total_rolling_max_7d']

        # Últimos 28 días (SOLO DATOS HISTORICOS REALES)
        ultimos_28 = get_rolling_values(df_temp, fecha, 28, ultimo_historico=ultimo_historico)
        if len(ultimos_28) > 0:
            features['total_rolling_mean_28d'] = np.mean(ultimos_28)
            features['total_rolling_std_28d'] = np.std(ultimos_28) if len(ultimos_28) > 1 else 0
            features['total_rolling_min_28d'] = np.min(ultimos_28)
            features['total_rolling_max_28d'] = np.max(ultimos_28)
        else:
            features['total_rolling_mean_28d'] = features['total_rolling_mean_14d']
            features['total_rolling_std_28d'] = features['total_rolling_std_14d']
            features['total_rolling_min_28d'] = features['total_rolling_min_14d']
            features['total_rolling_max_28d'] = features['total_rolling_max_14d']

        # Log rolling stats para validación (especialmente útil para debugging enero post-Navidad)
        if fecha.month == 1 and fecha.day <= 12:
            logger.info(f"   🔍 Rolling stats para {fecha.strftime('%Y-%m-%d')} (post-Navidad):")
            logger.info(f"      7d:  {len(ultimos_7)} valores, mean={features['total_rolling_mean_7d']:.2f} MWh")
            logger.info(f"      14d: {len(ultimos_14)} valores, mean={features['total_rolling_mean_14d']:.2f} MWh")
            logger.info(f"      28d: {len(ultimos_28)} valores, mean={features['total_rolling_mean_28d']:.2f} MWh")

        # ========================================
        # E. FEATURES DE CAMBIO
        # ========================================
        features['total_day_change'] = features['total_lag_1d'] - features['total_lag_7d']
        if features['total_lag_7d'] != 0:
            features['total_day_change_pct'] = (features['total_day_change'] / features['total_lag_7d']) * 100
        else:
            features['total_day_change_pct'] = 0

        # ========================================
        # F. FEATURES DE INTERACCIÓN
        # ========================================
        # IMPORTANTE: Usar nombres coherentes con las features climáticas simplificadas
        features['temp_x_is_weekend'] = features['temp_lag1d'] * features['is_weekend']
        features['temp_x_is_festivo'] = features['temp_lag1d'] * features['is_festivo']
        features['humidity_x_is_weekend'] = features['humidity_lag1d'] * features['is_weekend']
        features['dayofweek_x_festivo'] = features['dayofweek'] * features['is_festivo']
        features['month_x_festivo'] = features['month'] * features['is_festivo']
        features['weekend_x_month'] = features['is_weekend'] * features['month']

        return features

    def predict_next_n_days(self, n_days: int = 30) -> pd.DataFrame:
        """
        Predice los próximos N días de forma recursiva

        Args:
            n_days: Número de días a predecir

        Returns:
            DataFrame con predicciones
        """
        logger.info("="*80)
        logger.info(f"INICIANDO PREDICCIÓN DE PRÓXIMOS {n_days} DÍAS")
        logger.info("="*80)

        # Fecha inicial (mañana)
        ultimo_dia_historico = self.df_historico['fecha'].max()
        primer_dia_prediccion = ultimo_dia_historico + timedelta(days=1)
        
        # Fecha final del período a predecir
        ultimo_dia_prediccion = primer_dia_prediccion + timedelta(days=n_days - 1)
        
        # Cargar festivos para el rango completo de predicción (con margen de seguridad)
        # Agregar 30 días antes y después para cubrir lags y márgenes
        start_festivos = primer_dia_prediccion - timedelta(days=30)
        end_festivos = ultimo_dia_prediccion + timedelta(days=30)
        self._load_festivos_for_range(start_festivos, end_festivos)
        
        # Pre-cargar festivos en el CalendarClassifier del hourly_engine para evitar llamadas repetidas
        if self.hourly_engine is not None and hasattr(self.hourly_engine, 'calendar_classifier'):
            # Obtener años únicos del rango (más eficiente que bucle)
            years_needed = set(range(start_festivos.year, end_festivos.year + 1))
            
            # Pre-cargar todos los años necesarios
            if hasattr(self.hourly_engine.calendar_classifier, 'preload_years'):
                logger.info(f"Pre-cargando festivos para años {sorted(years_needed)} en CalendarClassifier...")
                self.hourly_engine.calendar_classifier.preload_years(list(years_needed))
                logger.info(f"✓ Festivos pre-cargados en CalendarClassifier (total: {len(self.hourly_engine.calendar_classifier.festivos_cache)} fechas)")

        logger.info(f"Último día con datos históricos: {ultimo_dia_historico.strftime('%Y-%m-%d')}")
        logger.info(f"Primera fecha a predecir: {primer_dia_prediccion.strftime('%Y-%m-%d')}")

        # Generar pronóstico del clima
        climate_forecast_df = self.generate_climate_forecast(primer_dia_prediccion, n_days)

        # DataFrame temporal (histórico + predicciones que vamos generando)
        df_temp = self.df_historico.copy()

        # Lista para guardar predicciones
        predictions = []

        # Loop día por día
        for day_idx in range(n_days):
            fecha = primer_dia_prediccion + timedelta(days=day_idx)

            logger.info(f"\n{'─'*80}")
            logger.info(f"📅 Día {day_idx + 1}/{n_days}: {fecha.strftime('%Y-%m-%d %A')}")

            # Obtener pronóstico del clima para este día
            # Normalizar la fecha a solo la parte de fecha (sin hora) para comparación robusta
            fecha_normalized = pd.Timestamp(fecha.date())
            climate_forecast_df['fecha_normalized'] = pd.to_datetime(climate_forecast_df['fecha']).dt.normalize()

            climate_rows = climate_forecast_df[climate_forecast_df['fecha_normalized'] == fecha_normalized]

            if len(climate_rows) == 0:
                logger.error(f"❌ No se encontró pronóstico climático para {fecha.date()}")
                logger.error(f"   Fechas disponibles en forecast: {climate_forecast_df['fecha'].head().tolist()}")
                logger.error(f"   Rango: {climate_forecast_df['fecha'].min()} a {climate_forecast_df['fecha'].max()}")
                raise ValueError(f"Falta pronóstico climático para {fecha.date()}")

            climate = climate_rows.iloc[0].to_dict()
            # Remover columna temporal
            if 'fecha_normalized' in climate:
                del climate['fecha_normalized']

            # Construir features (pasando ultimo_dia_historico para filtrar rolling stats)
            features = self.build_features_for_date(fecha, climate, df_temp, ultimo_dia_historico)

            # Ordenar features según el modelo (si tenemos feature_names)
            if self.feature_names:
                X_pred = pd.DataFrame([features])[self.feature_names]
            else:
                X_pred = pd.DataFrame([features])

            # Predecir
            demanda_pred_original = self.model.predict(X_pred)[0]
            demanda_pred = demanda_pred_original
            
            # ========================================
            # AJUSTE POST-PREDICCIÓN PARA FESTIVOS ESPECIALES Y TEMPORADA NAVIDEÑA
            # ========================================
            # Usa valores históricos del año anterior para corregir:
            # 1. Festivos especiales (8 dic, 25 dic, 1 ene)
            # 2. Temporada navideña (23 dic - 6 ene) - SOLO para festivos y fines de semana
            # IMPORTANTE: NO aplicar ajuste a días hábiles en temporada navideña
            lag_1y = features.get('total_lag_1y', 0)
            aplicar_ajuste = False
            weight_historical = 0.60  # Por defecto: ajuste moderado
            
            # Verificar si está en temporada navideña (23 dic - 6 ene)
            es_temporada_navideña = False
            if fecha.month == 12 and fecha.day >= 23:
                es_temporada_navideña = True
            elif fecha.month == 1 and fecha.day <= 6:
                es_temporada_navideña = True
            
            # Días muy especiales (incluye 24 dic y 2 ene aunque no sean festivos oficiales)
            month_day = f"{fecha.month:02d}-{fecha.day:02d}"
            very_special_holidays = ['12-24', '12-25', '12-08', '01-01', '01-02']  # Nochebuena, Navidad, Inmaculada, Año Nuevo, 2 de enero
            
            # Verificar si es día muy especial (festivo oficial o 24 de diciembre)
            if month_day in very_special_holidays:
                aplicar_ajuste = True
                weight_historical = 0.70  # Ajuste más fuerte para festivos muy especiales
            # Verificar si es festivo especial (otros festivos)
            elif features['is_festivo']:
                if not es_temporada_navideña:
                    # Otros festivos (fuera de temporada navideña): ajuste moderado
                    aplicar_ajuste = True
                    weight_historical = 0.60
                elif es_temporada_navideña:
                    # Festivos dentro de temporada navideña: ajuste moderado
                    aplicar_ajuste = True
                    weight_historical = 0.60
            
            # Para temporada navideña: aplicar ajuste SOLO a fines de semana (no días hábiles)
            if es_temporada_navideña and not features['is_festivo']:
                if features['is_weekend']:
                    # Fines de semana en temporada navideña: aplicar ajuste
                    aplicar_ajuste = True
                    weight_historical = 0.60
                # Días hábiles en temporada navideña: NO aplicar ajuste (mantener predicción del modelo)
            
            # Aplicar ajuste si corresponde y tenemos datos históricos
            if aplicar_ajuste and lag_1y > 0:
                # Aplicar promedio ponderado
                demanda_pred = (weight_historical * lag_1y) + ((1 - weight_historical) * demanda_pred_original)
                
                # Determinar tipo de ajuste para logging
                if month_day in very_special_holidays:
                    tipo_ajuste = "día muy especial (Nochebuena/Navidad/Año Nuevo/2 de enero/Inmaculada)"
                elif es_temporada_navideña:
                    tipo_ajuste = "temporada navideña"
                else:
                    tipo_ajuste = "festivo especial"
                
                logger.info(f"   🔧 Ajuste post-predicción aplicado ({tipo_ajuste})")
                logger.info(f"      - Valor histórico (1 año): {lag_1y:,.2f} MW")
                logger.info(f"      - Predicción modelo original: {demanda_pred_original:,.2f} MW")
                logger.info(f"      - Predicción final (ponderada {int(weight_historical*100)}% histórico): {demanda_pred:,.2f} MW")
            
            # Log adicional para debugging
            logger.info(f"   Demanda predicha: {demanda_pred:,.2f} MW")
            logger.info(f"   Features clave: is_weekend={features['is_weekend']}, is_festivo={features['is_festivo']}")
            logger.info(f"   Lags: lag_1d={features['total_lag_1d']:.2f}, lag_7d={features['total_lag_7d']:.2f}")
            if features.get('total_lag_1y', 0) > 0:
                logger.info(f"   Lag histórico (1 año): {features['total_lag_1y']:.2f}")

            # Desagregación horaria (si está habilitada)
            hourly_breakdown = {}
            senda_breakdown = {}
            cluster_id = None
            metodo_desagregacion = 'placeholder'

            if self.hourly_engine is not None:
                try:
                    hourly_result = self.hourly_engine.predict_hourly(fecha, demanda_pred, validate=True, return_senda=True)
                    hourly_breakdown = {f'P{i}': hourly_result['hourly'][i-1] for i in range(1, 25)}

                    # Capturar senda de referencia si está disponible
                    if 'senda_referencia' in hourly_result:
                        senda_breakdown = {f'senda_P{i}': hourly_result['senda_referencia'][i-1] for i in range(1, 25)}

                    # Capturar cluster_id si está disponible
                    if 'cluster_id' in hourly_result:
                        cluster_id = hourly_result['cluster_id']

                    metodo_desagregacion = hourly_result['method']
                    logger.info(f"   ✓ Desagregación horaria: método={metodo_desagregacion}, cluster={cluster_id}")
                except Exception as e:
                    logger.warning(f"   ⚠ Error en desagregación horaria: {e}")
                    logger.warning(f"   Usando placeholders")
                    hourly_breakdown = self._get_placeholder_hourly(demanda_pred)
                    metodo_desagregacion = 'placeholder'
            else:
                # Placeholders si no hay desagregación
                hourly_breakdown = self._get_placeholder_hourly(demanda_pred)
                metodo_desagregacion = 'placeholder'

            # Guardar predicción
            prediction_record = {
                'fecha': fecha,
                'demanda_predicha': demanda_pred,
                'is_festivo': features['is_festivo'],
                'is_weekend': features['is_weekend'],
                'dayofweek': features['dayofweek'],
                'temp_mean': climate['temp_mean'],
                'metodo_desagregacion': metodo_desagregacion,
                'cluster_id': cluster_id,
                **hourly_breakdown,  # Agregar P1-P24
                **senda_breakdown   # Agregar senda_P1-senda_P24
            }
            predictions.append(prediction_record)

            # Agregar predicción al df_temp para siguiente iteración
            # CRÍTICO: INCLUIR hourly_breakdown para que los lags (p8_lag_1d, p12_lag_1d, etc.)
            # usen valores REALES del clustering en lugar de multiplicadores fijos incorrectos
            new_row = {
                'fecha': fecha,
                'demanda_total': demanda_pred,
                **hourly_breakdown  # ← CRÍTICO: Agregar P1-P24 del clustering
            }
            df_temp = pd.concat([df_temp, pd.DataFrame([new_row])], ignore_index=True)

        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Predicción completada: {n_days} días procesados")
        logger.info(f"{'='*80}\n")

        return pd.DataFrame(predictions)

    def save_predictions(self, predictions_df: pd.DataFrame, output_dir: str = 'predictions'):
        """Guarda predicciones en CSV y resumen en JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Guardar CSV
        csv_path = output_path / 'predictions_next_30_days.csv'
        predictions_df.to_csv(csv_path, index=False)
        logger.info(f"✓ Predicciones guardadas en: {csv_path}")

        # Generar resumen
        summary = {
            'fecha_generacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_dias_predichos': len(predictions_df),
            'fecha_inicio': predictions_df['fecha'].min().strftime('%Y-%m-%d'),
            'fecha_fin': predictions_df['fecha'].max().strftime('%Y-%m-%d'),
            'demanda_promedio': float(predictions_df['demanda_predicha'].mean()),
            'demanda_min': float(predictions_df['demanda_predicha'].min()),
            'demanda_max': float(predictions_df['demanda_predicha'].max()),
            'dias_laborables': int((predictions_df['is_weekend'] == 0).sum()),
            'dias_fin_de_semana': int((predictions_df['is_weekend'] == 1).sum()),
            'dias_festivos': int(predictions_df['is_festivo'].sum())
        }

        json_path = output_path / 'predictions_summary.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Resumen guardado en: {json_path}")

        # Mostrar resumen
        logger.info("\n" + "="*80)
        logger.info("📊 RESUMEN DE PREDICCIONES")
        logger.info("="*80)
        logger.info(f"Período: {summary['fecha_inicio']} a {summary['fecha_fin']}")
        logger.info(f"Demanda promedio: {summary['demanda_promedio']:,.2f} MW")
        logger.info(f"Demanda mínima: {summary['demanda_min']:,.2f} MW")
        logger.info(f"Demanda máxima: {summary['demanda_max']:,.2f} MW")
        logger.info(f"Días laborables: {summary['dias_laborables']}")
        logger.info(f"Días fin de semana: {summary['dias_fin_de_semana']}")
        logger.info(f"Días festivos: {summary['dias_festivos']}")
        logger.info("="*80 + "\n")


def main():
    """Función principal"""
    logger.info("🚀 Iniciando pipeline de predicción EPM")

    # Inicializar pipeline
    pipeline = ForecastPipeline(
        model_path='models/trained/xgboost_20251120_161937.joblib',
        historical_data_path='data/features/data_with_features_latest.csv',
        festivos_path='data/calendario_festivos.json'
    )

    # Predecir próximos 30 días
    predictions = pipeline.predict_next_n_days(n_days=30)

    # Guardar resultados
    pipeline.save_predictions(predictions)

    logger.info("✅ Pipeline completado exitosamente")


if __name__ == '__main__':
    main()
