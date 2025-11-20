import os, time, csv, json, requests
from pathlib import Path

API_KEY = "AIzaSyAy3whFNyB5CyAaVZFqiVQWs9DyL9nvYiE"

# ======= Configura tu primera extracción =======
TEXT_QUERY     = "supermercado"          # Cambia: "farmacia", "banco", "tienda de abarrotes", etc.
INCLUDED_TYPE  = "supermarket"           # Opcional, ayuda a afinar (https://developers.google.com/maps/documentation/places/web-service/place-types)
REGION_CODE    = "MX"                    # País
CENTER         = {"latitude": 19.286, "longitude": -99.654}  # Toluca centro aprox.
RADIUS_METERS  = 5000                    # 5 km para costo bajo
MAX_PAGES      = 2                       # Máximo 2 páginas (~40 resultados aprox.)

# Campos mínimos para UI/costo (Field Mask)
FIELDS = ",".join([
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.types",
    "places.rating",
    "places.userRatingCount",
    "places.nationalPhoneNumber",
    "places.websiteUri",
])

# ======= Requests helpers =======
URL = "https://places.googleapis.com/v1/places:searchText"
HEADERS = {
    "X-Goog-Api-Key": API_KEY,
    "X-Goog-FieldMask": FIELDS
}

def search_page(page_token=None):
    body = {
        "textQuery": TEXT_QUERY,
        "regionCode": REGION_CODE,
        "languageCode": "es",
        "locationBias": {
            "circle": {"center": CENTER, "radius": RADIUS_METERS}
        }
    }
    if INCLUDED_TYPE:
        body["includedType"] = INCLUDED_TYPE
    if page_token:
        body["pageToken"] = page_token

    r = requests.post(URL, headers=HEADERS, json=body, timeout=30)
    r.raise_for_status()
    return r.json()

# ======= Run search with small pagination =======
seen = set()
results = []

page = 1
token = None
while page <= MAX_PAGES:
    data = search_page(token)
    for p in data.get("places", []):
        pid = p.get("id")
        if pid and pid not in seen:
            seen.add(pid)
            item = {
                "place_id": pid,
                "name": p.get("displayName", {}).get("text"),
                "address": p.get("formattedAddress"),
                "lat": (p.get("location") or {}).get("latitude"),
                "lng": (p.get("location") or {}).get("longitude"),
                "types": ",".join(p.get("types", [])),
                "rating": p.get("rating"),
                "user_ratings_total": p.get("userRatingCount"),
                "phone": p.get("nationalPhoneNumber"),
                "website": p.get("websiteUri"),
            }
            results.append(item)

    token = data.get("nextPageToken")
    if not token: break
    page += 1
    time.sleep(2)  # pequeña pausa para respetar rate limits

# ======= Export: CSV + GeoJSON =======
outdir = Path("salidas")
outdir.mkdir(exist_ok=True)

csv_path = outdir / "places_sample.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [
        "place_id","name","address","lat","lng","types","rating","user_ratings_total","phone","website"
    ])
    writer.writeheader()
    writer.writerows(results)

geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "place_id": r["place_id"],
                "name": r["name"],
                "address": r["address"],
                "types": r["types"],
                "rating": r["rating"],
                "user_ratings_total": r["user_ratings_total"],
                "phone": r["phone"],
                "website": r["website"],
            },
            "geometry": {"type": "Point", "coordinates": [r["lng"], r["lat"]]}
        }
        for r in results if r["lat"] and r["lng"]
    ]
}
gj_path = outdir / "places_sample.geojson"
with open(gj_path, "w", encoding="utf-8") as f:
    json.dump(geojson, f, ensure_ascii=False, indent=2)

print(f"Listo. Registros: {len(results)}")
print(f"- CSV:     {csv_path}")
print(f"- GeoJSON: {gj_path}")
