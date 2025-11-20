# route_matrix_all_in_one.py
# ============================================
# Extrae POI relevantes (OSM), selecciona destinos budget-aware,
# consulta Google Routes Distance Matrix (Essentials), y exporta resultados.
#
# Requisitos:
#   pip/conda: requests, pandas
#   (opcionales) xlsxwriter u openpyxl para .xlsx; matplotlib para heatmap
#   Si no están, el script sigue funcionando (omite .xlsx y/o heatmap).
# ============================================

import os, re, json, math, time
from pathlib import Path
from math import radians, cos, sin, asin, sqrt
import requests
import pandas as pd

# ====== INTENTAR CARGAR MATPLOTLIB (OPCIONAL) ======
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# =======================
# CONFIGURACIÓN PRINCIPAL
# =======================

API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY") or "TU_API_KEY_AQUI"
if (not API_KEY) or API_KEY == "TU_API_KEY_AQUI" or re.search(r"[^A-Za-z0-9\-\_]", API_KEY):
    raise SystemExit("❌ API key faltante o inválida. Exporta GOOGLE_MAPS_API_KEY en tu entorno.")

BUDGET_VALUE = 500.0
BUDGET_CURRENCY = "MXN"          # "MXN" o "USD"
EXCHANGE_RATE_MXN_PER_USD = 17.0
PRICE_PER_1000_ELEMENTS_USD = 5.0  # Essentials aprox.

# Toluca aprox (min_lat, min_lon, max_lat, max_lon)
BBOX = (19.20, -99.75, 19.40, -99.55)

CATS = {
    "super":   ("shop", "supermarket"),
    "pharma":  ("amenity", "pharmacy"),
    "hospital":("healthcare", "hospital"),
    "school":  ("amenity", "school"),
    "bus":     ("amenity", "bus_station"),
}

ORIGINS = [
    {"name": "O1_Conjunto", "lat": 19.287, "lon": -99.655},
    {"name": "O2_Centro",   "lat": 19.292, "lon": -99.661},
    {"name": "O3_ZonaSur",  "lat": 19.280, "lon": -99.645},
]

MAX_KM = 8.0
MAX_ELEMENTS_PER_REQUEST = 625  # Essentials

URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
FIELD_MASK = "originIndex,destinationIndex,status,condition,distanceMeters,duration,staticDuration"

OUTDIR = Path("salidas_matrix_all"); OUTDIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTDIR / "cache"; CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# Utilidades
# ============================================

def usd_from_budget(value, currency):
    return float(value) if currency.upper() == "USD" else float(value) / EXCHANGE_RATE_MXN_PER_USD

def elements_from_budget_usd(budget_usd, price_per_1000=5.0):
    return int(budget_usd / (price_per_1000/1000.0))

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

def iso_to_seconds(x):
    return float(x[:-1]) if isinstance(x, str) and x.endswith("s") else None

def summarize_cost(n_elements, usd_per_1000=5.0):
    cost_usd = (n_elements / 1000.0) * usd_per_1000
    cost_mxn = cost_usd * EXCHANGE_RATE_MXN_PER_USD
    return cost_usd, cost_mxn

# ============================================
# 1) Descargar POI desde Overpass (gratis)
# ============================================

def overpass_query(bbox, cats, cache_path: Path):
    if cache_path.exists():
        try:
            return pd.read_json(cache_path)
        except Exception:
            pass

    ors = "".join([f'  node["{k}"="{v}"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});\n'
                   for (k,v) in cats.values()])
    query = f"""[out:json][timeout:60];
(
{ors}
);
out center;"""

    r = requests.post("https://overpass-api.de/api/interpreter", data={'data': query}, timeout=120)
    r.raise_for_status()
    data = r.json().get('elements', [])

    rows = []
    for e in data:
        lat = e.get('lat') or (e.get('center') or {}).get('lat')
        lon = e.get('lon') or (e.get('center') or {}).get('lon')
        if lat is None or lon is None:
            continue
        tags = e.get('tags', {})
        key = next((k for k in ("shop","amenity","healthcare") if k in tags), None)
        val = tags.get(key, None) if key else None
        rows.append({
            "name": tags.get('name'),
            "lat": float(lat), "lon": float(lon),
            "k": key, "v": val
        })

    df = pd.DataFrame(rows)
    df.to_json(cache_path, orient="records", indent=2, force_ascii=False)
    return df

# ============================================
# 2) Selección budget-aware (nearest-k por categoría)
# ============================================

def select_pairs_budget_aware(poi_df, origins, cats, budget_value, budget_currency):
    budget_usd = usd_from_budget(budget_value, budget_currency)
    E_budget = elements_from_budget_usd(budget_usd, PRICE_PER_1000_ELEMENTS_USD)
    if E_budget <= 0:
        raise SystemExit("❌ Presupuesto insuficiente según parámetros configurados.")

    k = max(1, E_budget // (len(origins)*len(cats)))

    selected = []
    for oi, o in enumerate(origins):
        olat, olon = o["lat"], o["lon"]
        for cname, (ckey, cval) in cats.items():
            sub = poi_df[(poi_df["k"] == ckey) & (poi_df["v"] == cval)].copy()
            if sub.empty:
                continue
            sub["km"] = sub.apply(lambda r: haversine_km(olat, olon, r["lat"], r["lon"]), axis=1)
            sub = sub[sub["km"] <= MAX_KM].nsmallest(k, "km")
            for _, r in sub.iterrows():
                selected.append({
                    "origin_index": oi,
                    "origin_name": o["name"],
                    "o_lat": olat, "o_lon": olon,
                    "dest_name": r["name"] or f"{cname} (OSM)",
                    "d_lat": float(r["lat"]), "d_lon": float(r["lon"]),
                    "cat": cname,
                    "cat_key": ckey, "cat_val": cval,
                    "km_straight": float(r["km"])
                })
    pairs = pd.DataFrame(selected)
    return pairs, E_budget, k

# ============================================
# 3) Llamadas a Google Distance Matrix (lotes ≤ 625)
#    (sin FutureWarning: usamos sets en vez de concat con vacíos)
# ============================================

def call_matrix_batches(pairs_df):
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": FIELD_MASK,
    }

    results = []
    by_o = pairs_df.groupby(["origin_index","origin_name","o_lat","o_lon"], sort=False)

    current_o_list = []
    current_d_set = set()     # {(lat,lon)}
    current_d_rows = []       # [ [d_lat, d_lon, dest_name, cat], ... ]

    def flush_batch():
        nonlocal current_o_list, current_d_set, current_d_rows, results
        if not current_o_list:
            return
        d_df = pd.DataFrame(current_d_rows, columns=["d_lat","d_lon","dest_name","cat"])\
                .drop_duplicates(subset=["d_lat","d_lon"])
        results.extend(_run_one_batch(headers, current_o_list, d_df))
        current_o_list = []
        current_d_set = set()
        current_d_rows = []

    for (oi,oname,olat,olon), block in by_o:
        d_this = block[["d_lat","d_lon","dest_name","cat"]].drop_duplicates()

        # ¿cabe si agregamos este origen?
        tmp_set = set(current_d_set)
        for row in d_this.itertuples(index=False):
            key = (float(row.d_lat), float(row.d_lon))
            tmp_set.add(key)
        prospective_d_count = len(tmp_set)
        prospective_elements = (len(current_o_list) + 1) * prospective_d_count

        if current_o_list and prospective_elements > MAX_ELEMENTS_PER_REQUEST:
            flush_batch()

        # agrega origen
        current_o_list.append({"location":{"latLng":{"latitude": float(olat), "longitude": float(olon)}}})

        # agrega destinos únicos
        for row in d_this.itertuples(index=False):
            key = (float(row.d_lat), float(row.d_lon))
            if key not in current_d_set:
                current_d_set.add(key)
                current_d_rows.append([float(row.d_lat), float(row.d_lon), row.dest_name, row.cat])

    flush_batch()
    df_res = pd.DataFrame(results)
    return df_res

def _run_one_batch(headers, o_list, d_df):
    body = {
        "origins":      [{"waypoint": o} for o in o_list],
        "destinations": [{"waypoint": {"location":{"latLng":{"latitude": float(r.d_lat), "longitude": float(r.d_lon)}}}}
                         for r in d_df.itertuples(index=False)],
        "travelMode": "DRIVE",
        "routingPreference": "ROUTING_PREFERENCE_UNSPECIFIED",
        "regionCode": "MX",
        "units": "METRIC",
    }

    r = requests.post(URL, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        (OUTDIR / "last_error_body.json").write_text(r.text, encoding="utf-8")
        r.raise_for_status()

    rows = r.json() if r.headers.get("Content-Type","").startswith("application/json") else json.loads(r.text)

    d_list = list(d_df.itertuples(index=False))
    batch_results = []
    for item in rows:
        oi, di = item.get("originIndex"), item.get("destinationIndex")
        if oi is None or di is None:
            continue
        o = o_list[oi]["location"]["latLng"]
        d = d_list[di]
        batch_results.append({
            "o_lat": float(o["latitude"]), "o_lon": float(o["longitude"]),
            "d_lat": float(d.d_lat), "d_lon": float(d.d_lon),
            "dest_name": d.dest_name, "cat": d.cat,
            "distance_m": item.get("distanceMeters"),
            "duration_s": iso_to_seconds(item.get("duration")),
            "static_duration_s": iso_to_seconds(item.get("staticDuration")),
            "status_code": (item.get("status") or {}).get("code", 0),
            "condition": item.get("condition")
        })
    time.sleep(0.2)
    return batch_results

# ============================================
# 4) Exports y visual (con fallback de motor Excel)
# ============================================

def export_all(pairs, df_matrix):
    df = df_matrix.copy()
    df["duration_min"] = df["duration_s"].apply(lambda x: (x/60.0) if x is not None else None)
    df["static_duration_min"] = df["static_duration_s"].apply(lambda x: (x/60.0) if x is not None else None)

    o_map = {(o["lat"], o["lon"]): o["name"] for o in ORIGINS}
    def _oname(r):
        key = (round(r["o_lat"],6), round(r["o_lon"],6))
        return o_map.get(key) or o_map.get((r["o_lat"], r["o_lon"]))
    df["origin_name"] = df.apply(_oname, axis=1)

    # CSV siempre
    df.to_csv(OUTDIR/"matrix_results.csv", index=False, encoding="utf-8")
    pairs.to_csv(OUTDIR/"pairs_selected.csv", index=False, encoding="utf-8")

    # Excel si hay motor disponible
    excel_path = OUTDIR/"route_matrix_relevant.xlsx"
    def _write_excel(engine_name):
        with pd.ExcelWriter(excel_path, engine=engine_name) as xl:
            pairs.to_excel(xl, sheet_name="pairs_selected", index=False)
            df.to_excel(xl, sheet_name="matrix_results", index=False)
            agg = df.groupby(["origin_name","cat"], dropna=False)["duration_min"].mean().reset_index()
            agg_pivot = agg.pivot(index="origin_name", columns="cat", values="duration_min")
            agg_pivot.to_excel(xl, sheet_name="avg_time_by_cat")

    wrote_xlsx = False
    try:
        _write_excel("xlsxwriter"); wrote_xlsx = True
    except ModuleNotFoundError:
        try:
            _write_excel("openpyxl"); wrote_xlsx = True
            print("ℹ️ Se usó 'openpyxl' como motor de Excel (xlsxwriter no instalado).")
        except ModuleNotFoundError:
            print("⚠️ No hay motor Excel (xlsxwriter/openpyxl). Se omitió 'route_matrix_relevant.xlsx'.")

    # Heatmap si hay matplotlib
    if HAS_MPL:
        top_dest = (df.groupby("dest_name")["duration_min"]
                      .mean().sort_values().head(30).index.tolist())
        hm = df[df["dest_name"].isin(top_dest)].copy()
        mat = hm.pivot_table(index="origin_name", columns="dest_name",
                             values="duration_min", aggfunc="mean")
        if not mat.empty:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(min(12, 1.0 + 0.45*len(mat.columns)), 2 + 0.6*len(mat.index)))
            plt.imshow(mat.values, aspect="auto")
            plt.xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right")
            plt.yticks(range(len(mat.index)), mat.index)
            plt.title("Matriz de tiempo (min) – destinos top")
            plt.colorbar(label="min")
            plt.tight_layout()
            plt.savefig(OUTDIR/"heatmap_time.png", dpi=160)
            plt.close()
    else:
        print("ℹ️ matplotlib no instalado: no se generó heatmap_time.png")

# ============================================
# MAIN
# ============================================

def main():
    print("=== Configuración ===")
    print(f"Presupuesto: {BUDGET_VALUE} {BUDGET_CURRENCY} | Tipo de cambio: {EXCHANGE_RATE_MXN_PER_USD} MXN/USD")
    budget_usd = usd_from_budget(BUDGET_VALUE, BUDGET_CURRENCY)
    print(f"≈ {budget_usd:.2f} USD  | Precio: {PRICE_PER_1000_ELEMENTS_USD} USD / 1000 elementos (Essentials)")

    print("\n→ Descargando POI OSM (cache activado)...")
    poi = overpass_query(BBOX, CATS, CACHE_DIR/"poi_osm_toluca.json")
    print(f"POI totales en bbox: {len(poi)}")
    if poi.empty:
        raise SystemExit("❌ No se obtuvieron POI desde Overpass. Ajusta BBOX o categorías.")

    print("\n→ Seleccionando destinos por presupuesto y cercanía...")
    pairs, E_budget, k = select_pairs_budget_aware(poi, ORIGINS, CATS, BUDGET_VALUE, BUDGET_CURRENCY)
    if pairs.empty:
        raise SystemExit("❌ No hay destinos seleccionados (revisa MAX_KM, categorías o BBOX).")
    print(f"Elementos presupuestados (aprox): {E_budget}")
    print(f"k por categoría y origen: {k}")
    print(f"Pares seleccionados (potenciales): {len(pairs)}")

    print("\n→ Consultando Google Distance Matrix (Essentials, lotes ≤ 625)...")
    df_matrix = call_matrix_batches(pairs)
    if df_matrix.empty:
        raise SystemExit("❌ No se recibieron resultados de la API de rutas.")

    used_elements = len(df_matrix)
    usd_cost, mxn_cost = summarize_cost(used_elements, PRICE_PER_1000_ELEMENTS_USD)

    print("\n=== Resumen de ejecución ===")
    print(f"Elementos consultados realmente: {used_elements}")
    print(f"Costo estimado: ~{usd_cost:.2f} USD  (~{mxn_cost:.2f} MXN)")
    if BUDGET_CURRENCY.upper() == "USD":
        print(f"Presupuesto: {BUDGET_VALUE:.2f} USD -> {'OK' if usd_cost <= BUDGET_VALUE else '⚠️ EXCEDIDO'}")
    else:
        print(f"Presupuesto: {BUDGET_VALUE:.2f} MXN -> {'OK' if mxn_cost <= BUDGET_VALUE else '⚠️ EXCEDIDO'}")

    print("\n→ Exportando CSV/Excel/heatmap...")
    export_all(pairs, df_matrix)

    print("\n✓ Listo. Archivos en:", OUTDIR)
    print(" - pairs_selected.csv")
    print(" - matrix_results.csv")
    print(" - (si hay motor) route_matrix_relevant.xlsx")
    print(" - (si hay matplotlib) heatmap_time.png")
    print(" - cache/poi_osm_toluca.json (POI)")

if __name__ == "__main__":
    main()
