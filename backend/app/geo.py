from fastapi import APIRouter, Query 
from typing import Optional, Dict, Any, List 
import numpy as np
import h3
"""
    ***LIBRERÍAS***   
FastAPI es un fw para construir API's con python
 #APIrouter permite la creación de rutas agrupadas en modulos
 #Query permite definir parámetros de consulta
 #typing (Optional, Dict, Any, List) tipos para anotar y documentar lo que reciben/devuelven cada una de las funciones.
 #numpy usado para generar numeros aleatorios y algunos calculos numericos rapidos
 #h3 librería de uber para dividir el mapa en celdas hexagonales
"""


#router es el camino por donde el usuario puede pedir algo a la API
router = APIRouter()

def _color_from_io(io: float) -> str:
    # IO en [0,1] -> color HSL (verde a rojo)
    hue = int((1 - io) * 120)  # 120=verde, 0=rojo
    return f"hsl({hue}, 80%, 50%)"

@router.get("/hexgrid") #que código corre cuando alguien pide ese recurso
def hexgrid(
    minLat: float = Query(..., description="Mín lat"),
    minLon: float = Query(..., description="Mín lon"),
    maxLat: float = Query(..., description="Máx lat"),
    maxLon: float = Query(..., description="Máx lon"),
    res: int = Query(8, ge=6, le=10, description="Resolución H3 (6-10)")
) -> Dict[str, Any]:
    """
    Devuelve un FeatureCollection GeoJSON de hexágonos H3 dentro del bbox.
    Cada hex trae propiedades de ejemplo: oferta, demanda, distancia e IO.
    """


    # Polígono del bbox en formato GeoJSON (lng, lat)
    poly = {
        "type": "Polygon",
        "coordinates": [[
            [minLon, minLat],
            [minLon, maxLat],
            [maxLon, maxLat],
            [maxLon, minLat],
            [minLon, minLat],
        ]]
    }
    # Celdas H3 que cubren el bbox
    idxs = h3.polyfill(poly, res)

    rng = np.random.default_rng(42)
    features: List[Dict[str, Any]] = []
    for h in idxs:
        # Borde del hex (lng,lat) y centróide (lat,lon)
        boundary = h3.h3_to_geo_boundary(h, geo_json=True)  # [[lng,lat], ...]
        centroid_lat, centroid_lon = h3.h3_to_geo(h)

        # Señales sintéticas (ejemplo). Luego las alimentarás con datos reales (DENUE/OSM/INEGI).
        oferta = float(rng.uniform(0, 100))
        demanda = float(rng.uniform(0, 100))
        distancia = float(rng.uniform(0, 100))
        # IO simple: demanda y distancia ayudan, oferta resta (ajusta pesos al validar)
        w1, w2, w3 = 0.5, 0.3, 0.2
        io = max(0.0, min(1.0, w1*demanda + w2*distancia - w3*oferta))

        coords = boundary + [boundary[0]]  # cierra polígono
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "h3": h,
                "centroid": {"lat": centroid_lat, "lon": centroid_lon},
                "oferta": oferta,
                "demanda": demanda,
                "distancia": distancia,
                "io": io,
                "color": _color_from_io(io)
            }
        })

    return {"type": "FeatureCollection", "features": features}
