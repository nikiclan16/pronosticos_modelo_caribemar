"""
Módulo para obtener festivos desde la API de pronosticos.jmdatalabs.co

Reemplaza la librería 'holidays' y los archivos JSON hardcodeados.
"""

import requests
from typing import List, Set, Optional
from datetime import datetime, date
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class FestivosAPIClient:
    """
    Cliente para obtener festivos desde la API de pronosticos.jmdatalabs.co
    
    IMPORTANTE: Este cliente NO tiene caché persistente. Cada instancia nueva
    hace llamadas frescas a la API, garantizando que los cambios en la API se
    reflejen en cada ejecución. El caché solo existe a nivel de instancia
    (ForecastPipeline, CalendarClassifier) durante la ejecución actual.
    """
    
    BASE_URL = "https://pronosticos.jmdatalabs.co/api/v1/admin/configuracion-interna/listarFestivos"
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Inicializa el cliente de festivos.
        
        Args:
            base_url: URL base del endpoint (opcional, usa default si no se proporciona)
        """
        self.base_url = base_url or self.BASE_URL
    
    def get_festivos(
        self, 
        start_date: str, 
        end_date: str, 
        ucp: str
    ) -> List[str]:
        """
        Obtiene lista de festivos desde la API para un rango de fechas y UCP.
        
        Args:
            start_date: Fecha inicio en formato YYYY-MM-DD
            end_date: Fecha fin en formato YYYY-MM-DD
            ucp: Nombre del UCP (ej: 'Antioquia', 'Atlantico')
        
        Returns:
            Lista de fechas en formato 'YYYY-MM-DD'
        
        Raises:
            requests.RequestException: Si hay error en la petición HTTP
            ValueError: Si la respuesta no es exitosa
        """
        url = f"{self.base_url}/{start_date}/{end_date}/{ucp}"
        
        try:
            logger.debug(f"Solicitando festivos desde API: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('success', False):
                error_msg = data.get('message', 'Error desconocido en la API')
                logger.error(f"API retornó success=False: {error_msg}")
                raise ValueError(f"Error en API de festivos: {error_msg}")
            
            # Extraer fechas de la respuesta
            festivos_list = data.get('data', [])
            fechas = [item['fecha'] for item in festivos_list if 'fecha' in item]
            
            logger.info(f"✓ Obtenidos {len(fechas)} festivos desde API para {ucp} ({start_date} a {end_date})")
            return fechas
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout al obtener festivos desde API: {url}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en petición HTTP a API de festivos: {e}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Error procesando respuesta de API de festivos: {e}")
            raise ValueError(f"Error procesando respuesta de API: {e}")
    
    def is_festivo(
        self, 
        fecha: date, 
        ucp: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> bool:
        """
        Verifica si una fecha es festivo para un UCP específico.
        
        Args:
            fecha: Fecha a verificar (date o datetime)
            ucp: Nombre del UCP
            start_date: Fecha inicio para consulta (opcional, usa año de la fecha si no se proporciona)
            end_date: Fecha fin para consulta (opcional, usa año de la fecha si no se proporciona)
        
        Returns:
            True si es festivo, False en caso contrario
        """
        # Convertir a date si es datetime
        if isinstance(fecha, datetime):
            fecha = fecha.date()
        
        # Determinar rango de fechas si no se proporciona
        if start_date is None or end_date is None:
            year = fecha.year
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
        
        # Obtener festivos para el rango
        festivos = self.get_festivos(start_date, end_date, ucp)
        
        # Verificar si la fecha está en la lista
        fecha_str = fecha.strftime('%Y-%m-%d')
        return fecha_str in festivos
    
    def get_festivos_set(
        self, 
        start_date: str, 
        end_date: str, 
        ucp: str
    ) -> Set[str]:
        """
        Obtiene un set de festivos (más eficiente para búsquedas múltiples).
        
        Args:
            start_date: Fecha inicio en formato YYYY-MM-DD
            end_date: Fecha fin en formato YYYY-MM-DD
            ucp: Nombre del UCP
        
        Returns:
            Set de fechas en formato 'YYYY-MM-DD'
        """
        festivos_list = self.get_festivos(start_date, end_date, ucp)
        return set(festivos_list)


# NOTA: El caché es solo por instancia (en memoria), no persistente.
# Cada ejecución nueva crea nuevas instancias, por lo que siempre se llama a la API
# al menos una vez por año necesario. Esto garantiza que los cambios en la API
# se reflejen en cada ejecución, mientras que dentro de la misma ejecución
# se evitan llamadas repetidas mediante caché en memoria.

