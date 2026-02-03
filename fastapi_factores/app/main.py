from fastapi import FastAPI
from dotenv import load_dotenv

from app.routes.barras import router as barras_router
from app.routes.agrupaciones import router as agrupaciones_router
from app.routes.medidas import router as medidas_router
from app.routes.rangos import router as rangos_router
from app.routes.utilidades import router as utilidades_router
from app.routes.calculos import router as calculos_router

load_dotenv()

app = FastAPI(title="FastAPI Factores", version="1.0.0")

app.include_router(barras_router)
app.include_router(agrupaciones_router)
app.include_router(medidas_router)
app.include_router(rangos_router)
app.include_router(utilidades_router)
app.include_router(calculos_router)


@app.get("/")
def root():
    return {"ok": True, "service": "fastapi-factores"}
