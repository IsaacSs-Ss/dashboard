import json
import os
import sys
from contextlib import asynccontextmanager
from math import radians, cos, sin, asin, sqrt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- 1. CEREBRO DE IA (CORREGIDO Y BLINDADO) ---
generar_analisis_brechas = None # Inicializamos vacío por seguridad
CEREBRO_ACTIVO = False

try:
    # Intentamos importar explícitamente la función
    # Esto busca en la misma carpeta (backend/app)
    from .analytics import generar_analisis_brechas
    CEREBRO_ACTIVO = True
    print("🧠 [SYSTEM] Módulo de Inteligencia (analytics.py) ACTIVADO.")

except (ImportError, AttributeError) as e:
    # Si falla el import relativo, intentamos absoluto o reportamos error
    print(f"⚠️ [DIAGNÓSTICO] Falló importación relativa: {e}")
    try:
        import analytics
        # Verificamos si la función existe dentro del módulo importado
        if hasattr(analytics, 'generar_analisis_brechas'):
            generar_analisis_brechas = analytics.generar_analisis_brechas
            CEREBRO_ACTIVO = True
            print("🧠 [SYSTEM] Módulo de Inteligencia (analytics.py) ACTIVADO (Modo Absoluto).")
        else:
            print("⚠️ [ADVERTENCIA] El archivo 'analytics.py' existe, pero falta la función 'generar_analisis_brechas'.")
            
    except (ImportError, AttributeError) as e2:
        CEREBRO_ACTIVO = False
        print(f"⚠️ [ADVERTENCIA] No se pudo cargar la IA. El análisis estará desactivado.")

# --- 2. VARIABLES GLOBALES ---
DATOS_NEGOCIOS = []

# --- 3. LIFESPAN (Gestor de vida) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- INICIO ---
    global DATOS_NEGOCIOS
    
    # Lógica robusta para encontrar el archivo JSON
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    rutas = [
        os.path.join(base_dir, "negocios_toluca_final.json"),               # En la misma carpeta
        os.path.join(base_dir, "..", "data", "negocios_toluca_final.json"), # backend/data
        os.path.join(base_dir, "..", "..", "data", "negocios_toluca_final.json"), # raiz/data
        "data/negocios_toluca_final.json"
    ]
    
    archivo_encontrado = False
    for ruta in rutas:
        if os.path.exists(ruta):
            try:
                with open(ruta, 'r', encoding='utf-8') as f:
                    DATOS_NEGOCIOS = json.load(f)
                print(f"✅ [DB] Base de datos cargada desde: {ruta}")
                print(f"📊 [DB] Total de negocios: {len(DATOS_NEGOCIOS)}")
                archivo_encontrado = True
                break
            except Exception as e:
                print(f"❌ Error leyendo {ruta}: {e}")
    
    if not archivo_encontrado:
        print("🔥 [ERROR CRÍTICO] No se encontró 'negocios_toluca_final.json'. La API iniciará vacía.")

    yield # La aplicación corre aquí

    # --- APAGADO ---
    DATOS_NEGOCIOS.clear()
    print("👋 [SYSTEM] Apagando sistema...")

# --- 4. CONFIGURACIÓN APP ---
app = FastAPI(
    title="Geo API - Inteligencia Territorial",
    version="3.0.1",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. UTILIDADES ---
def haversine(lat1, lon1, lat2, lon2):
    try:
        lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        return c * 6371
    except:
        return 9999

# --- 6. ENDPOINTS ---

@app.get("/")
def health_check():
    return {
        "estado": "online",
        "ia_activa": CEREBRO_ACTIVO,
        "negocios_memoria": len(DATOS_NEGOCIOS)
    }

@app.get("/api/negocios")
def obtener_negocios(limite: int = 2000, categoria: str = None):
    resultado = DATOS_NEGOCIOS
    if categoria and categoria.lower() != "todos":
        resultado = [n for n in resultado if categoria.lower() in str(n.get('nombre_act', '')).lower()]
    return resultado[:limite]

@app.get("/geo/hexgrid")
def analisis_inteligencia(res: int = 9, objetivo: str = "cafeteria"):
    # Verificación de seguridad antes de llamar a la función
    if not CEREBRO_ACTIVO or generar_analisis_brechas is None:
        return {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {"error": "Módulo de IA (analytics.py) no disponible o incompleto."}
        }
    
    # Llamada protegida
    try:
        return generar_analisis_brechas(DATOS_NEGOCIOS, resolucion=res, giro_objetivo=objetivo)
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        return {"error": f"Fallo interno en el análisis: {str(e)}"}