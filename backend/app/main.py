import json
import os
import sys
from contextlib import asynccontextmanager
from math import radians, cos, sin, asin, sqrt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- 1. IMPORTACIÓN DEL CEREBRO (Analytics) ---
# Esta lógica permite que funcione tanto con 'uvicorn backend...' como ejecutando el archivo directo.
CEREBRO_ACTIVO = False
generar_analisis_brechas = None

try:
    # Intento 1: Importación relativa (Estándar para paquetes)
    from .analytics import generar_analisis_brechas
    CEREBRO_ACTIVO = True
    print("🧠 [SYSTEM] Módulo de Inteligencia (analytics.py) ACTIVADO (Import relativo).")
except (ImportError, AttributeError):
    try:
        # Intento 2: Importación absoluta (Agregando ruta al sistema)
        # Esto soluciona problemas cuando Python no reconoce el paquete 'backend'
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import analytics
        
        # Verificamos explícitamente que la función exista para evitar AttributeError
        if hasattr(analytics, 'generar_analisis_brechas'):
            generar_analisis_brechas = analytics.generar_analisis_brechas
            CEREBRO_ACTIVO = True
            print("🧠 [SYSTEM] Módulo de Inteligencia (analytics.py) ACTIVADO (Import absoluto).")
        else:
            print("⚠️ [ADVERTENCIA] 'analytics.py' encontrado, pero falta la función 'generar_analisis_brechas'.")

    except (ImportError, AttributeError) as e:
        print(f"⚠️ [ADVERTENCIA] No se pudo cargar analytics.py: {e}")

if not CEREBRO_ACTIVO:
    print("⚠️ [SYSTEM] El análisis inteligente estará desactivado (No se encontró el módulo o la función).")

# --- 2. VARIABLES GLOBALES ---
DATOS_NEGOCIOS = []

# --- 3. GESTOR DE VIDA (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- AL INICIAR EL SERVIDOR ---
    global DATOS_NEGOCIOS
    print("🚀 Iniciando Geo API...")
    
    # Buscamos el archivo JSON subiendo niveles de carpetas
    base_dir = os.path.dirname(os.path.abspath(__file__)) # Carpeta backend/app
    
    rutas_posibles = [
        os.path.join(base_dir, "negocios_toluca_final.json"),               # En la misma carpeta
        os.path.join(base_dir, "..", "data", "negocios_toluca_final.json"), # Estructura estándar
        os.path.join(base_dir, "..", "..", "data", "negocios_toluca_final.json"), # Estructura anidada
        "data/negocios_toluca_final.json"                                 # Ruta simple
    ]
    
    archivo_cargado = False
    for ruta in rutas_posibles:
        if os.path.exists(ruta):
            try:
                with open(ruta, 'r', encoding='utf-8') as f:
                    DATOS_NEGOCIOS = json.load(f)
                print(f"✅ [DB] Base de datos cargada desde: {ruta}")
                print(f"📊 [DB] Registros en memoria: {len(DATOS_NEGOCIOS)}")
                archivo_cargado = True
                break
            except Exception as e:
                print(f"❌ Error leyendo {ruta}: {e}")
    
    if not archivo_cargado:
        print("🔥 [ERROR CRÍTICO] No se encontró 'negocios_toluca_final.json'. Verifica la carpeta data.")

    yield # Aquí la app está corriendo y recibiendo peticiones

    # --- AL APAGAR EL SERVIDOR ---
    DATOS_NEGOCIOS.clear()
    print("👋 [SYSTEM] Servidor apagado correctamente.")

# --- 4. CONFIGURACIÓN DE LA APP ---
app = FastAPI(
    title="Geo API - Inteligencia Territorial",
    version="3.0.0",
    description="API para Dashboard de Análisis de Brechas",
    lifespan=lifespan
)

# Configuración CORS para permitir que React (Vite) se conecte
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. UTILIDADES MATEMÁTICAS ---
def haversine(lat1, lon1, lat2, lon2):
    try:
        lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        return c * 6371 # Radio Tierra KM
    except:
        return 9999

# --- 6. ENDPOINTS (RUTAS) ---

@app.get("/")
def health_check():
    return {
        "estado": "online", 
        "ia_activa": CEREBRO_ACTIVO, 
        "negocios_cargados": len(DATOS_NEGOCIOS)
    }

@app.get("/api/negocios")
def obtener_negocios(limite: int = 2000, categoria: str = None):
    """ Retorna lista de negocios filtrada """
    resultado = DATOS_NEGOCIOS
    if categoria and categoria.lower() != "todos":
        filtro = categoria.lower()
        resultado = [n for n in resultado if filtro in str(n.get('nombre_act', '')).lower()]
    return resultado[:limite]

@app.get("/api/analisis/radio")
def analisis_radio(lat: float, lng: float, km: float = 1.0):
    """ Retorna negocios cercanos a un punto (Click en mapa) """
    resultados = []
    for negocio in DATOS_NEGOCIOS:
        # Validamos que existan coordenadas para evitar errores
        if 'latitud' in negocio and 'longitud' in negocio:
            dist = haversine(lat, lng, negocio['latitud'], negocio['longitud'])
            if dist <= km:
                item = negocio.copy()
                item['distancia_km'] = round(dist, 2)
                resultados.append(item)
    resultados.sort(key=lambda x: x['distancia_km'])
    return resultados[:50]

@app.get("/geo/hexgrid")
def analisis_inteligencia(res: int = 9, objetivo: str = "cafeteria"):
    """ 
    ENDPOINT PRINCIPAL DEL DASHBOARD
    Genera la malla de hexágonos con el índice de oportunidad.
    """
    if not CEREBRO_ACTIVO or generar_analisis_brechas is None:
        return {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {"error": "IA no disponible. Falta analytics.py o la función es incorrecta."}
        }
    
    try:
        # Ejecutar el análisis en analytics.py
        return generar_analisis_brechas(DATOS_NEGOCIOS, resolucion=res, giro_objetivo=objetivo)
    except Exception as e:
        print(f"❌ Error ejecutando análisis: {e}")
        return {"error": f"Error interno en analytics: {str(e)}"}