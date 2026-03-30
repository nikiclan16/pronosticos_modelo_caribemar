# Flujo: Curvas Típicas → FDA → FDP

Este documento describe el flujo completo para obtener curvas típicas y calcular los factores FDA y FDP.

## Resumen del Flujo

```
┌─────────────────────────────────────────────────────────────┐
│  1. POST /curvas-tipicas                                    │
│     → Dame las 8 más típicas de enero ORDINARIO             │
│     ← Devuelve 4 (solo encontró 4 típicas)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. POST /fda  (o /fda-fdp para ambos)                      │
│     → Calcula FDA sobre esas 4 curvas                       │
│     ← Factores normalizados (suma = 1.0 por período)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. POST /fdp                                               │
│     → Calcula FDP sobre esas 4 curvas                       │
│     ← Factor de potencia por período                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Paso 1: Obtener Curvas Típicas

Selecciona las N curvas más típicas del período de análisis por forma (patrón horario).

### Endpoint

```
POST /factores/calculos/curvas-tipicas
```

### Request

```json
{
  "fecha_inicial": "2024-01-01",
  "fecha_final": "2024-01-31",
  "mc": "MC01",
  "tipo_dia": "ORDINARIO",
  "flujo_tipo": "A",
  "n_max": 8,
  "barra": null
}
```

| Campo             | Tipo        | Descripción                                   |
| ----------------- | ----------- | ---------------------------------------------- |
| `fecha_inicial` | string      | Fecha inicio del rango (YYYY-MM-DD)            |
| `fecha_final`   | string      | Fecha fin del rango (YYYY-MM-DD)               |
| `mc`            | string      | Código del mercado/centro                     |
| `tipo_dia`      | string      | `ORDINARIO`, `SABADO` o `FESTIVO`        |
| `flujo_tipo`    | string      | `A` (Activa) o `R` (Reactiva)              |
| `n_max`         | int         | Máximo de curvas típicas a devolver (1-100)  |
| `barra`         | string/null | Barra específica o `null` para todas del MC |

### Response

```json
{
  "ok": true,
  "n": 4,
  "data": [
    {
      "barra": "BARRA_SUR",
      "fecha": "2024-01-15",
      "periodos": {"p1": 120.5, "p2": 115.3, "p3": 110.2, "...": "...", "p24": 98.2}
    },
    {
      "barra": "BARRA_NORTE",
      "fecha": "2024-01-08",
      "periodos": {"p1": 85.1, "p2": 90.4, "p3": 88.7, "...": "...", "p24": 72.0}
    },
    {
      "barra": "BARRA_SUR",
      "fecha": "2024-01-22",
      "periodos": {"p1": 118.0, "p2": 112.7, "p3": 108.5, "...": "...", "p24": 95.5}
    },
    {
      "barra": "BARRA_ESTE",
      "fecha": "2024-01-10",
      "periodos": {"p1": 200.3, "p2": 195.8, "p3": 190.2, "...": "...", "p24": 180.1}
    }
  ]
}
```

> **Nota:** Si pides 8 curvas pero solo hay 4 típicas, devuelve 4.

### Algoritmo de Selección

1. Obtiene todas las curvas clusterizadas del rango/MC/tipo_día
2. Normaliza por L2 (compara **forma**, no nivel absoluto)
3. Calcula centralidad (menor distancia media = más típica)
4. Devuelve hasta `n_max` más típicas

---

## Paso 2: Calcular FDA (Factor de Demanda Ajustada)

Normaliza los factores para que cada período sume exactamente 1.0.

### Endpoint

```
POST /factores/calculos/fda
```

### Request

Usa `barra` y `fecha` de la respuesta de curvas-tipicas:

```json
{
  "fecha_inicial": "2024-01-01",
  "fecha_final": "2024-01-31",
  "mc": "MC01",
  "tipo_dia": "ORDINARIO",
  "curvas_tipicas": [
    {"barra": "BARRA_SUR", "fecha": "2024-01-15"},
    {"barra": "BARRA_NORTE", "fecha": "2024-01-08"},
    {"barra": "BARRA_SUR", "fecha": "2024-01-22"},
    {"barra": "BARRA_ESTE", "fecha": "2024-01-10"}
  ]
}
```

### Response

```json
{
  "ok": true,
  "mc": "MC01",
  "tipo_dia": "ORDINARIO",
  "fecha_inicial": "2024-01-01",
  "fecha_final": "2024-01-31",
  "resultado": {
    "tipo_dia": "ORDINARIO",
    "n_registros": 4,
    "factores": {
      "0": {"barra": "BARRA_SUR", "fecha": "2024-01-15", "p1": 0.23, "p2": 0.22, "...": "...", "p24": 0.19},
      "1": {"barra": "BARRA_NORTE", "fecha": "2024-01-08", "p1": 0.16, "p2": 0.17, "...": "...", "p24": 0.14},
      "2": {"barra": "BARRA_SUR", "fecha": "2024-01-22", "p1": 0.22, "p2": 0.21, "...": "...", "p24": 0.18},
      "3": {"barra": "BARRA_ESTE", "fecha": "2024-01-10", "p1": 0.39, "p2": 0.40, "...": "...", "p24": 0.49}
    },
    "suma_total": 1.0,
    "ajuste_aplicado": 0.00012
  }
}
```

> **Importante:** Cada período (p1-p24) suma exactamente **1.0** entre todas las curvas.

### Algoritmo FDA

1. Suma todos los valores de cada período
2. Encuentra el máximo de cada período
3. Calcula ajuste: `1.0 - suma_período`
4. Aplica ajuste solo al valor máximo de cada período
5. Resultado: cada período suma 1.0

---

## Paso 3: Calcular FDP (Factor de Demanda Pronóstico)

Calcula el factor de potencia combinando potencia activa y reactiva.

### Endpoint

```
POST /factores/calculos/fdp
```

### Request

Mismo formato que FDA:

```json
{
  "fecha_inicial": "2024-01-01",
  "fecha_final": "2024-01-31",
  "mc": "MC01",
  "tipo_dia": "ORDINARIO",
  "curvas_tipicas": [
    {"barra": "BARRA_SUR", "fecha": "2024-01-15"},
    {"barra": "BARRA_NORTE", "fecha": "2024-01-08"},
    {"barra": "BARRA_SUR", "fecha": "2024-01-22"},
    {"barra": "BARRA_ESTE", "fecha": "2024-01-10"}
  ]
}
```

### Response

```json
{
  "ok": true,
  "mc": "MC01",
  "tipo_dia": "ORDINARIO",
  "fecha_inicial": "2024-01-01",
  "fecha_final": "2024-01-31",
  "resultado": {
    "tipo_dia": "ORDINARIO",
    "n_registros": 4,
    "factores": {
      "0": {"barra": "BARRA_SUR", "fecha": "2024-01-15", "p1": 0.95, "p2": 0.94, "...": "...", "p24": 0.97},
      "1": {"barra": "BARRA_NORTE", "fecha": "2024-01-08", "p1": 0.92, "p2": 0.91, "...": "...", "p24": 0.93},
      "2": {"barra": "BARRA_SUR", "fecha": "2024-01-22", "p1": 0.96, "p2": 0.95, "...": "...", "p24": 0.98},
      "3": {"barra": "BARRA_ESTE", "fecha": "2024-01-10", "p1": 0.88, "p2": 0.87, "...": "...", "p24": 0.90}
    }
  }
}
```

### Fórmula FDP

```
FDP = cos(atan(Q / P))

Donde:
  P = Potencia Activa (medidas tipo "A")
  Q = Potencia Reactiva (medidas tipo "R")
```

**Casos especiales:**

- Si P=0 y Q=0: FDP = 1.0
- Si P=0 y Q≠0: FDP = 0.0

> **Nota:** Requiere que existan medidas tanto tipo A como tipo R para las curvas seleccionadas.

---

## Alternativa: FDA + FDP en una sola llamada

### Endpoint

```
POST /factores/calculos/fda-fdp
```

### Request

Mismo formato que FDA/FDP.

### Response

```json
{
  "ok": true,
  "mc": "MC01",
  "tipo_dia": "ORDINARIO",
  "fecha_inicial": "2024-01-01",
  "fecha_final": "2024-01-31",
  "fda": {
    "tipo_dia": "ORDINARIO",
    "n_registros": 4,
    "factores": { "...": "..." },
    "suma_total": 1.0,
    "ajuste_aplicado": 0.00012
  },
  "fdp": {
    "tipo_dia": "ORDINARIO",
    "n_registros": 4,
    "factores": { "...": "..." }
  }
}
```

---

## Ejemplo de Uso Completo (JavaScript/Frontend)

```javascript
async function calcularFactores(mc, fechaInicial, fechaFinal, tipoDia, nMax) {
  const baseUrl = '/factores/calculos';

  // Paso 1: Obtener curvas típicas
  const curvasRes = await fetch(`${baseUrl}/curvas-tipicas`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      fecha_inicial: fechaInicial,
      fecha_final: fechaFinal,
      mc: mc,
      tipo_dia: tipoDia,
      flujo_tipo: 'A',
      n_max: nMax,
      barra: null
    })
  });
  const curvas = await curvasRes.json();

  if (!curvas.ok || curvas.n === 0) {
    console.log('No se encontraron curvas típicas');
    return null;
  }

  console.log(`Encontradas ${curvas.n} curvas típicas`);

  // Paso 2: Preparar referencias para FDA/FDP
  const curvasTipicas = curvas.data.map(c => ({
    barra: c.barra,
    fecha: c.fecha
  }));

  // Paso 3: Calcular FDA y FDP juntos
  const fdaFdpRes = await fetch(`${baseUrl}/fda-fdp`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      fecha_inicial: fechaInicial,
      fecha_final: fechaFinal,
      mc: mc,
      tipo_dia: tipoDia,
      curvas_tipicas: curvasTipicas
    })
  });

  return await fdaFdpRes.json();
}

// Uso
const resultado = await calcularFactores('MC01', '2024-01-01', '2024-01-31', 'ORDINARIO', 8);
console.log('FDA:', resultado.fda);
console.log('FDP:', resultado.fdp);
```

---

## Notas Importantes

1. **Curvas típicas por forma:** El algoritmo normaliza por L2, comparando solo la forma del patrón horario, no el nivel absoluto.
2. **n_max es un máximo:** Si pides 8 curvas pero solo hay 4 típicas, devuelve 4.
3. **FDP requiere A y R:** Para calcular FDP necesitas medidas tanto de tipo Activa como Reactiva.
4. **Suma FDA = 1.0:** El algoritmo FDA garantiza que cada período sume exactamente 1.0.
