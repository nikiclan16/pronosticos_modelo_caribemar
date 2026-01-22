# FastAPI Factores

Backend en FastAPI para el modulo de Factores (barras, agrupaciones, medidas y utilidades) usando la misma base de datos PostgreSQL.

## Requisitos

- Python 3.10+
- PostgreSQL accesible desde el host

## Configuracion

Definir `DATABASE_URL` con el DSN de PostgreSQL:

```bash
export DATABASE_URL="postgresql://usuario:password@host:puerto/base"
```

## Instalacion

```bash
pip install -r requirements.txt
```

## Ejecutar

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Notas

- Las consultas y operaciones se basan en `App_Code/Factores.cs` y los WebMethods de `modulofactores.aspx.cs`.
- El endpoint `/factores/medidas/faltantes` reemplaza la logica de `modalinicio`.
- El endpoint `/factores/medidas/marcar` reemplaza `ActualizarMedidas` (solo actualiza en BD).

## Endpoints principales

- Barras: `/factores/barras`, `/factores/barras/por-mc/{mc}`
- Agrupaciones: `/factores/agrupaciones`, `/factores/agrupaciones/por-barra/{id}`
- Medidas: `/factores/medidas`, `/factores/medidas/completo`, `/factores/medidas/calcular-completo`
- Rangos: `/factores/rangos`
- Utilidades: `/factores/tipo-dia/{nombre}`, `/factores/festivos`

Consulta el OpenAPI en `/docs` para ver todos los parametros.
