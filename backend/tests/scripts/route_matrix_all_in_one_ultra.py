# route_matrix_all_in_one_ultra.py
# ==========================================================
# Masivo: OSM tiled + selección budget/aggressive + matrices
# Motores: google | osrm | valhalla  (elige con ROUTING_ENGINE)
# Salidas: CSV (siempre), Parquet (pyarrow), Excel (si hay motor)
# Reintentos, checkpoints, lotes grandes (50k–1M+ posibles)
# ==========================================================

import os, re, json, time, math, sys
from pathlib import Path
from math import radians, cos, sin, asin, sqrt
import random
from wsgiref import headers
import requests
import pandas as pd

# ========= Opcionales detectados =========
try:
    import pyarrow  # noqa
    HAS_PARQUET = True
except Exception:
    HAS_PARQUET = False

# ======== Config principal (puedes sobrescribir por ENV) ========
ENGINE = os.environ.get("ROUTING_ENGINE", "google").lower()  # "google" | "osrm" | "valhalla"

# Presupuesto (solo afecta a modo "budget-aware" con Google)
BUDGET_VALUE = float(os.environ.get("BUDGET_VALUE", "500"))  # 500 MXN por defecto
BUDGET_CURRENCY = os.environ.get("BUDGET_CURRENCY", "MXN")   # "USD" o "MXN"
EXCHANGE_RATE_MXN_PER_USD = float(os.environ.get("MXN_PER_USD", "17"))
PRICE_PER_1000_ELEMENTS_USD = float(os.environ.get("PRICE_PER_1000_USD", "5.0"))  # Essentials aprox

# Selección de destinos: "budget" | "aggressive"
SELECTION_MODE = os.environ.get("SELECTION_MODE", "budget").lower()
K_PER_CAT = int(os.environ.get("K_PER_CAT", "1000"))  # usado en aggressive
MAX_KM = float(os.environ.get("MAX_KM", "20"))        # radio

# Tiling OSM (más tiles = más POI sin timeouts)
OSM_TILES = int(os.environ.get("OSM_TILES", "8"))     # 8x8 = 64 tiles
OSM_SLEEP_S = float(os.environ.get("OSM_SLEEP_S", "0.8"))
OSM_TIMEOUT = int(os.environ.get("OSM_TIMEOUT", "180"))

# Google Routes
GOOGLE_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
GOOGLE_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
FIELD_MASK = (
    "originIndex,destinationIndex,status,condition,"
    "distanceMeters,duration,staticDuration,"
    "travelAdvisory,fallbackInfo,localizedValues"
)
MAX_ELEMENTS_PER_REQUEST_GOOGLE = 625  # Essentials (sin TRAFFIC_AWARE_OPTIMAL)

# OSRM / Valhalla (self-hosted)
OSRM_URL = os.environ.get("OSRM_URL", "http://localhost:5000")              # /table
VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002")      # /sources_to_targets
OSRM_MAX_COORDS = int(os.environ.get("OSRM_MAX_COORDS", "100"))             # coords por request (depende build)
VALH_MAX_SOURCES = int(os.environ.get("VALH_MAX_SOURCES", "50"))
VALH_MAX_TARGETS = int(os.environ.get("VALH_MAX_TARGETS", "200"))

# Área de estudio (Toluca aprox)
BBOX = (
    float(os.environ.get("BBOX_MINLAT", "19.20")),
    float(os.environ.get("BBOX_MINLON", "-99.75")),
    float(os.environ.get("BBOX_MAXLAT", "19.40")),
    float(os.environ.get("BBOX_MAXLON", "-99.55")),
)

# Orígenes (puedes ampliar o leer de CSV)
ORIGINS = [
    {"name": "O1_Conjunto", "lat": 19.287, "lon": -99.655},
    {"name": "O2_Centro",   "lat": 19.292, "lon": -99.661},
    {"name": "O3_ZonaSur",  "lat": 19.280, "lon": -99.645},
]

# Categorías OSM (expandidas)
BASE_CATS = {
    "super": ("shop","supermarket"),
    "pharma": ("amenity","pharmacy"),
    "hospital": ("healthcare","hospital"),
    "school": ("amenity","school"),
    "bus": ("amenity","bus_station"),
}
EXTRA_CATS = {
    "mall": ("shop","mall"),
    "convenience": ("shop","convenience"),
    "fuel": ("amenity","fuel"),
    "parking": ("amenity","parking"),
    "atm": ("amenity","atm"),
    "bank": ("amenity","bank"),
    "clinic": ("healthcare","clinic"),
    "university": ("amenity","university"),
    "college": ("amenity","college"),
    "police": ("amenity","police"),
    "fire": ("amenity","fire_station"),
    "post": ("amenity","post_office"),
    "market": ("amenity","marketplace"),
    "library": ("amenity","library"),
    # cuidado con volumen extremo:
    # "restaurant": ("amenity","restaurant"),
    # "cafe": ("amenity","cafe"),
    # "fast_food": ("amenity","fast_food"),
}
CATS = {**BASE_CATS, **EXTRA_CATS}

OSM_FIELDS = [
    "name","brand","operator","opening_hours",
    "addr:street","addr:housenumber","addr:neighbourhood","addr:city","addr:postcode",
    "phone","website","wheelchair","emergency","beds","brand:wikidata"
]

# Salidas
OUTDIR = Path("salidas_ultra"); OUTDIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTDIR/"cache"; CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUTDIR/"manifest.jsonl"  # 1 línea por lote procesado
RAW_ERR = OUTDIR/"last_error_body.json"

# ============== Utilidades ==============
def usd_from_budget(value, currency):
    return float(value) if currency.upper()=="USD" else float(value)/EXCHANGE_RATE_MXN_PER_USD

def elements_from_budget_usd(budget_usd, price_per_1000=5.0):
    return int(budget_usd/(price_per_1000/1000.0))

def haversine_km(a,b,c,d):
    R=6371.0
    dlat,dlon=radians(c-a),radians(d-b)
    x=sin(dlat/2)**2+cos(radians(a))*cos(radians(c))*sin(dlon/2)**2
    return 2*R*asin(sqrt(x))

def iso_to_seconds(x): return float(x[:-1]) if isinstance(x,str) and x.endswith("s") else None

def summarize_cost(n, usd_per_1000=5.0):
    usd=(n/1000.0)*usd_per_1000; mxn=usd*EXCHANGE_RATE_MXN_PER_USD; return usd,mxn

def backoff_sleep(attempt):
    time.sleep(min(60, (2**attempt)*0.5 + random.random()*0.3))

# ============== Overpass masivo (tiles, node/way/relation) ==============
def _tile_bbox(bbox, tiles):
    minlat,minlon,maxlat,maxlon=bbox
    dlat=(maxlat-minlat)/tiles; dlon=(maxlon-minlon)/tiles
    return [(minlat+i*dlat, minlon+j*dlon, minlat+(i+1)*dlat, minlon+(j+1)*dlon)
            for i in range(tiles) for j in range(tiles)]

def _q_for_bbox(b, cats):
    a,b_,c,d=b
    cls=[]
    for k,v in cats.values():
        for t in ("node","way","relation"):
            cls.append(f'{t}["{k}"="{v}"]({a},{b_},{c},{d});')
    return f"[out:json][timeout:{OSM_TIMEOUT}];\n(\n"+"\n".join(cls)+"\n);\nout center;"

def overpass_query_tiled(bbox, cats, tiles, cache_path:Path):
    if cache_path.exists():
        try: return pd.read_json(cache_path)
        except Exception: pass
    rows=[]
    for i,b in enumerate(_tile_bbox(bbox, max(1,tiles)), 1):
        q=_q_for_bbox(b, cats)
        for attempt in range(6):
            try:
                r=requests.post("https://overpass-api.de/api/interpreter",
                                data={'data':q}, timeout=OSM_TIMEOUT+30)
                if r.status_code==429 or r.status_code>=500:
                    raise requests.HTTPError(f"{r.status_code} {r.text[:120]}")
                r.raise_for_status()
                els=r.json().get("elements",[])
                break
            except Exception as e:
                if attempt==5: raise
                backoff_sleep(attempt)
        for e in els:
            lat=e.get('lat') or (e.get('center') or {}).get('lat')
            lon=e.get('lon') or (e.get('center') or {}).get('lon')
            if lat is None or lon is None: continue
            tags=e.get('tags',{}) or {}
            key=next((k for k in ("shop","amenity","healthcare") if k in tags), None)
            val=tags.get(key) if key else None
            rec={"tile":i,"osm_type":e.get("type"),"osm_id":e.get("id"),
                 "lat":float(lat),"lon":float(lon),"k":key,"v":val,
                 "name":tags.get("name")}
            for f in OSM_FIELDS: rec[f.replace(":","_")] = tags.get(f)
            rows.append(rec)
        time.sleep(OSM_SLEEP_S)
    df=pd.DataFrame(rows)
    if not df.empty: df=df.drop_duplicates(subset=["k","v","name","lat","lon"])
    df.to_json(cache_path, orient="records", indent=2, force_ascii=False)
    df.to_csv(OUTDIR/"poi_all.csv", index=False, encoding="utf-8")
    if HAS_PARQUET: df.to_parquet(OUTDIR/"poi_all.parquet", index=False)
    return df

# ============== Selección (budget/aggressive) ==============
def select_pairs(poi_df, origins, cats, mode="budget"):
    if mode=="budget":
        budget_usd = usd_from_budget(BUDGET_VALUE, BUDGET_CURRENCY)
        E_budget = elements_from_budget_usd(budget_usd, PRICE_PER_1000_ELEMENTS_USD)
        k = max(1, E_budget // (len(origins)*len(cats)))
    else:
        E_budget = None  # informativo
        k = K_PER_CAT

    chosen=[]
    for oi,o in enumerate(origins):
        olat,olon=o["lat"],o["lon"]
        for cname,(ckey,cval) in cats.items():
            sub=poi_df[(poi_df["k"]==ckey) & (poi_df["v"]==cval)].copy()
            if sub.empty: continue
            sub["km"]=sub.apply(lambda r:haversine_km(olat,olon,r["lat"],r["lon"]), axis=1)
            sub=sub[sub["km"]<=MAX_KM].nsmallest(k,"km")
            for _,r in sub.iterrows():
                chosen.append({
                    "origin_index":oi,"origin_name":o["name"],
                    "o_lat":olat,"o_lon":olon,
                    "dest_name":r.get("name") or f"{cname} (OSM)",
                    "d_lat":float(r["lat"]),"d_lon":float(r["lon"]),
                    "cat":cname,"k":ckey,"v":cval,
                    "km_straight":float(r["km"]),
                    "brand":r.get("brand"),"operator":r.get("operator"),
                    "opening_hours":r.get("opening_hours"),
                    "addr_street":r.get("addr_street"),
                    "addr_housenumber":r.get("addr_housenumber"),
                    "addr_neighbourhood":r.get("addr_neighbourhood"),
                    "addr_city":r.get("addr_city"),
                    "addr_postcode":r.get("addr_postcode"),
                    "phone":r.get("phone"),"website":r.get("website"),
                    "osm_type":r.get("osm_type"),"osm_id":r.get("osm_id"),
                })
    df=pd.DataFrame(chosen)
    return df, E_budget, k

# ============== Persistencia por lote ==============
def append_rows_csv(df, path:Path):
    header = not path.exists()
    df.to_csv(path, index=False, mode="a", header=header, encoding="utf-8")

def write_checkpoint(obj):
    with open(CHECKPOINT, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False)+"\n")

# ============== Motores de ruteo ==============
def run_google_matrix(pairs_df):
    # Permite forzar FieldMask mínimo si quieres aislar problemas:
    #   setx FIELD_MASK_MIN 1
    use_min_mask = os.environ.get("FIELD_MASK_MIN", "0") == "1"
    field_mask = (
        "originIndex,destinationIndex,status,condition,"
        "distanceMeters,duration,staticDuration"
    ) if use_min_mask else FIELD_MASK

    if not GOOGLE_API_KEY:
        raise SystemExit("❌ Falta GOOGLE_MAPS_API_KEY para ENGINE=google")

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": field_mask,
    }

    results_file = OUTDIR / "matrix_results.csv"
    total_rows = 0

    by_o = pairs_df.groupby(["origin_index", "origin_name", "o_lat", "o_lon"], sort=False)

    # OJO: guardamos SOLO 'location', NO anidamos 'waypoint' aquí
    current_o_list = []             # [{'location': {'latLng': {...}}}, ...]
    current_d_set = set()           # {(lat,lon)}
    current_d_rows = []             # [[d_lat, d_lon, dest_name, cat], ...]

    def flush_batch():
        nonlocal current_o_list, current_d_set, current_d_rows, total_rows

        if not current_o_list:
            return

        d_df = (
            pd.DataFrame(current_d_rows, columns=["d_lat", "d_lon", "dest_name", "cat"])
            .drop_duplicates(subset=["d_lat", "d_lon"])
        )

        elems = len(current_o_list) * len(d_df)
        print(f"→ Lote: O={len(current_o_list)} D={len(d_df)}  elems={elems}")

        body = {
            "origins": [
                {"waypoint": o}  # aquí sí envolvemos con 'waypoint'
                for o in current_o_list
            ],
            "destinations": [
                {
                    "waypoint": {
                        "location": {
                            "latLng": {
                                "latitude": float(r.d_lat),
                                "longitude": float(r.d_lon),
                            }
                        }
                    }
                }
                for r in d_df.itertuples(index=False)
            ],
            "travelMode": "DRIVE",
            "routingPreference": "ROUTING_PREFERENCE_UNSPECIFIED",
            "regionCode": "MX",
            "units": "METRIC",
        }

        # Reintentos con backoff + STREAMING NDJSON
        rows = []
        for attempt in range(6):
            try:
                r = requests.post(
                    GOOGLE_URL, headers=headers, json=body, stream=True, timeout=120
                )
                # Reintenta si hay rate limit/5xx
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code}: {r.text}")
                # Para otros 4xx, guarda error y aborta
                if r.status_code >= 400:
                    RAW_ERR.write_text(r.text, encoding="utf-8")
                    raise requests.HTTPError(f"{r.status_code}: {r.text[:400]} ...")

                # ---- Parser robusto NDJSON multi-línea ----
                rows = []
                buf = []
                depth = 0
                in_str = False
                esc = False

                for ln in r.iter_lines(decode_unicode=True):
                    if not ln:
                        continue
                    s = ln.strip()
                    if s.startswith(")]}'"):
                        s = s[4:].lstrip()
                    if s in ("[", "]"):
                        continue  # ignorar brackets del framing

                    for ch in s:
                        if in_str:
                            buf.append(ch)
                            if esc:
                                esc = False
                            elif ch == '\\':
                                esc = True
                            elif ch == '"':
                                in_str = False
                            continue

                        if ch == '"':
                            in_str = True
                            buf.append(ch)
                            continue

                        if ch == '{':
                            if depth == 0:
                                buf = []  # nuevo objeto
                            depth += 1
                            buf.append(ch)
                            continue

                        if ch == '}':
                            depth -= 1
                            buf.append(ch)
                            if depth == 0 and buf:
                                candidate = ''.join(buf).strip()
                                if candidate.endswith(','):
                                    candidate = candidate[:-1].rstrip()
                                try:
                                    rows.append(json.loads(candidate))
                                except json.JSONDecodeError:
                                    pass
                                buf = []
                            continue

                        if depth > 0:
                            buf.append(ch)

                break  # éxito, salir del loop de reintentos

            except Exception as e:
                if attempt == 5:
                    try:
                        if hasattr(e, "response") and getattr(e, "response") is not None:
                            RAW_ERR.write_text(e.response.text, encoding="utf-8")
                        else:
                            RAW_ERR.write_text(str(e), encoding="utf-8")
                    except Exception:
                        pass
                    raise
                backoff_sleep(attempt)


        # Mapear índices a coordenadas
        d_list = list(d_df.itertuples(index=False))
        out = []
        for it in rows:
            oi, di = it.get("originIndex"), it.get("destinationIndex")
            if oi is None or di is None:
                continue
            o = current_o_list[oi]["location"]["latLng"]  # <- sin 'waypoint' aquí
            d = d_list[di]
            adv = it.get("travelAdvisory") or {}
            loc = it.get("localizedValues") or {}
            fbk = it.get("fallbackInfo") or {}
            out.append(
                {
                    "engine": "google",
                    "o_lat": float(o["latitude"]),
                    "o_lon": float(o["longitude"]),
                    "d_lat": float(d.d_lat),
                    "d_lon": float(d.d_lon),
                    "dest_name": d.dest_name,
                    "cat": d.cat,
                    "distance_m": it.get("distanceMeters"),
                    "duration_s": iso_to_seconds(it.get("duration")),
                    "static_duration_s": iso_to_seconds(it.get("staticDuration")),
                    "status_code": (it.get("status") or {}).get("code", 0),
                    "condition": it.get("condition"),
                    "travelAdvisory_json": json.dumps(adv, ensure_ascii=False) if adv else None,
                    "localizedValues_json": json.dumps(loc, ensure_ascii=False) if loc else None,
                    "fallbackInfo_json": json.dumps(fbk, ensure_ascii=False) if fbk else None,
                }
            )

        df = pd.DataFrame(out)
        append_rows_csv(df, results_file)
        total_rows += len(df)
        print(f"  ↳ guardados: {len(df)} filas (acumulado {total_rows})")
        write_checkpoint(
            {
                "engine": "google",
                "batch_size": len(df),
                "rows_cum": total_rows,
                "origins_in_batch": len(current_o_list),
                "dest_unique_in_batch": len(d_df),
            }
        )

        # reset del lote
        current_o_list.clear()
        current_d_set.clear()
        current_d_rows.clear()

    # Construcción de lotes
        # ===============================
    # Construcción de lotes (cap D)
    # Troceamos destinos por origen
    # ===============================
    for (oi, oname, olat, olon), block in by_o:
        origin_loc = {"location": {"latLng": {"latitude": float(olat), "longitude": float(olon)}}}

        # Siempre trabajaremos con UN solo origen por lote para poder partir D
        if current_o_list:
            flush_batch()
        current_o_list.append(origin_loc)

        d_this = block[["d_lat", "d_lon", "dest_name", "cat"]].drop_duplicates()
        # capacidad de destinos para este lote con O=1
        maxD = MAX_ELEMENTS_PER_REQUEST_GOOGLE // max(1, len(current_o_list))  # 625

        # función para reiniciar lote con el mismo origen
        def _restart_batch_same_origin():
            current_o_list.clear()
            current_d_set.clear()
            current_d_rows.clear()
            current_o_list.append(origin_loc)

        for row in d_this.itertuples(index=False):
            key = (float(row.d_lat), float(row.d_lon))

            # si al añadir uno más se excede la capacidad de D, enviamos lote y reiniciamos con el mismo origen
            if len(current_d_set) >= maxD:
                flush_batch()
                _restart_batch_same_origin()
                maxD = MAX_ELEMENTS_PER_REQUEST_GOOGLE // max(1, len(current_o_list))  # vuelve a ser 625

            if key not in current_d_set:
                current_d_set.add(key)
                current_d_rows.append([float(row.d_lat), float(row.d_lon), row.dest_name, row.cat])

        # Enviamos el último chunk de este origen antes de pasar al siguiente
        flush_batch()

    # Cargamos lo acumulado en CSV y lo regresamos
    return pd.read_csv(results_file)

def run_osrm_table(pairs_df):
    if not OSRM_URL:
        raise SystemExit("❌ Falta OSRM_URL para ENGINE=osrm")
    out_file = OUTDIR/"matrix_results.csv"
    total_rows=0

    # agrupamos por origen para lanzar many-to-many por origen en chunks de destinos
    by_o = pairs_df.groupby(["origin_index","origin_name","o_lat","o_lon"], sort=False)
    for (oi,oname,olat,olon), block in by_o:
        dests = block[["d_lat","d_lon","dest_name","cat"]].drop_duplicates().reset_index(drop=True)
        # OSRM: armamos lotes de coordenadas para no sobrepasar límite
        for start in range(0, len(dests), OSRM_MAX_COORDS-1):
            sub = dests.iloc[start:start+OSRM_MAX_COORDS-1]
            coords = [(olon,olat)] + list(zip(sub["d_lon"], sub["d_lat"]))  # lon,lat
            coord_str = ";".join([f"{x:.6f},{y:.6f}" for x,y in coords])
            sources = "0"
            destinations = ";".join(str(i) for i in range(1, len(coords)))
            url = f"{OSRM_URL.rstrip('/')}/table/v1/driving/{coord_str}?sources={sources}&destinations={destinations}"
            for attempt in range(6):
                try:
                    r=requests.get(url, timeout=120)
                    if r.status_code in (429,500,502,503,504): raise requests.HTTPError(r.text)
                    r.raise_for_status()
                    js=r.json()
                    break
                except Exception as e:
                    if attempt==5: raise
                    backoff_sleep(attempt)
            durs = js.get("durations") or [[]]
            dists = js.get("distances") or [[]]
            out=[]
            for idx,(lat,lon,name,cat) in enumerate(zip(sub["d_lat"], sub["d_lon"], sub["dest_name"], sub["cat"]), start=1):
                dur_s = durs[0][idx-1] if durs and durs[0] and durs[0][idx-1] is not None else None
                dist_m = dists[0][idx-1] if dists and dists[0] and dists[0][idx-1] is not None else None
                out.append({
                    "engine":"osrm",
                    "o_lat":float(olat),"o_lon":float(olon),
                    "d_lat":float(lat),"d_lon":float(lon),
                    "dest_name":name,"cat":cat,
                    "distance_m": dist_m,
                    "duration_s": dur_s,
                    "static_duration_s": dur_s,  # OSRM no separa tráfico
                    "status_code": 0,
                    "condition": "ROUTE_EXISTS" if dur_s is not None else "ROUTE_NOT_FOUND",
                    "travelAdvisory_json": None,
                    "localizedValues_json": None,
                    "fallbackInfo_json": None,
                })
            df=pd.DataFrame(out)
            append_rows_csv(df, out_file)
            total_rows += len(df)
            write_checkpoint({"engine":"osrm","rows_cum":total_rows})
    return pd.read_csv(out_file)

def run_valhalla_sources_to_targets(pairs_df):
    if not VALHALLA_URL:
        raise SystemExit("❌ Falta VALHALLA_URL para ENGINE=valhalla")
    out_file = OUTDIR/"matrix_results.csv"
    total_rows=0

    by_o = pairs_df.groupby(["origin_index","origin_name","o_lat","o_lon"], sort=False)
    for (oi,oname,olat,olon), block in by_o:
        dests = block[["d_lat","d_lon","dest_name","cat"]].drop_duplicates().reset_index(drop=True)
        # chunking por limites aproximados
        for start in range(0, len(dests), VALH_MAX_TARGETS):
            sub = dests.iloc[start:start+VALH_MAX_TARGETS]
            body = {
                "sources": [{"lat":float(olat),"lon":float(olon)}],
                "targets": [{"lat":float(r.d_lat),"lon":float(r.d_lon)} for r in sub.itertuples(index=False)],
                "costing": "auto"
            }
            url = f"{VALHALLA_URL.rstrip('/')}/sources_to_targets"
            for attempt in range(6):
                try:
                    r=requests.post(url, json=body, timeout=120)
                    if r.status_code in (429,500,502,503,504): raise requests.HTTPError(r.text)
                    r.raise_for_status()
                    js=r.json()
                    break
                except Exception as e:
                    if attempt==5: raise
                    backoff_sleep(attempt)
            # Valhalla: matrix [sources][targets]
            matrix = (js.get("sources_to_targets") or [[]])[0]
            out=[]
            for row, meta in zip(matrix, sub.itertuples(index=False)):
                dur_s = row.get("time") if row else None
                dist_m = row.get("distance")*1000 if row and row.get("distance") is not None else None
                out.append({
                    "engine":"valhalla",
                    "o_lat":float(olat),"o_lon":float(olon),
                    "d_lat":float(meta.d_lat),"d_lon":float(meta.d_lon),
                    "dest_name":meta.dest_name,"cat":meta.cat,
                    "distance_m": dist_m,
                    "duration_s": dur_s,
                    "static_duration_s": dur_s,
                    "status_code": 0,
                    "condition": "ROUTE_EXISTS" if dur_s is not None else "ROUTE_NOT_FOUND",
                    "travelAdvisory_json": None,
                    "localizedValues_json": None,
                    "fallbackInfo_json": None,
                })
            df=pd.DataFrame(out)
            append_rows_csv(df, out_file)
            total_rows += len(df)
            write_checkpoint({"engine":"valhalla","rows_cum":total_rows})
    return pd.read_csv(out_file)

# ============== Export final (merge + opcional Excel/Parquet) ==============
def finalize_exports(pairs_df, results_df):
    # enrich con metadatos de destino (brand/operator/etc.)
    meta = pairs_df[["d_lat","d_lon","dest_name","cat","brand","operator","opening_hours",
                     "addr_street","addr_housenumber","addr_neighbourhood","addr_city","addr_postcode",
                     "phone","website","osm_type","osm_id"]].drop_duplicates()
    df = results_df.merge(meta, on=["d_lat","d_lon","dest_name","cat"], how="left")
    # derivados
    df["duration_min"] = df["duration_s"].apply(lambda x: x/60 if pd.notna(x) else None)
    df["static_duration_min"] = df["static_duration_s"].apply(lambda x: x/60 if pd.notna(x) else None)
    # CSV/Parquet
    df.to_csv(OUTDIR/"matrix_results.csv", index=False, encoding="utf-8")
    pairs_df.to_csv(OUTDIR/"pairs_selected.csv", index=False, encoding="utf-8")
    if HAS_PARQUET:
        df.to_parquet(OUTDIR/"matrix_results.parquet", index=False)
        pairs_df.to_parquet(OUTDIR/"pairs_selected.parquet", index=False)
    # Excel si hay motor
    excel_path = OUTDIR/"route_matrix_relevant.xlsx"
    def _write_xlsx(engine):
        with pd.ExcelWriter(excel_path, engine=engine) as xl:
            pairs_df.to_excel(xl, sheet_name="pairs_selected", index=False)
            df.to_excel(xl, sheet_name="matrix_results", index=False)
            agg = df.groupby(["engine","cat"], dropna=False)["duration_min"].mean().reset_index()
            agg.pivot(index="engine", columns="cat", values="duration_min").to_excel(xl, sheet_name="avg_min_by_engine_cat")
    try:
        _write_xlsx("xlsxwriter")
    except ModuleNotFoundError:
        try:
            _write_xlsx("openpyxl")
            print("ℹ️ Excel generado con openpyxl.")
        except ModuleNotFoundError:
            print("⚠️ Sin motor Excel; se omitió .xlsx.")

    return df

# ============== MAIN ==============
def main():
    print(f"ENGINE: {ENGINE} | SELECTION_MODE: {SELECTION_MODE} | OSM_TILES: {OSM_TILES} | MAX_KM: {MAX_KM}")
    if ENGINE=="google":
        print(f"Presupuesto: {BUDGET_VALUE} {BUDGET_CURRENCY} (Essentials ~{PRICE_PER_1000_ELEMENTS_USD} USD/1000)")
    # 1) OSM masivo
    poi = overpass_query_tiled(BBOX, CATS, tiles=OSM_TILES, cache_path=CACHE_DIR/"poi_all.json")
    print("POI únicos:", len(poi))
    if poi.empty: raise SystemExit("❌ Overpass devolvió 0 POI.")

    # 2) Selección
    pairs, E_budget, k = select_pairs(poi, ORIGINS, CATS, mode=SELECTION_MODE)
    print(f"Pairs seleccionados: {len(pairs)} | k={k} por categoría/origen | presupuesto≈{E_budget if E_budget else 'n/a'}")
    if pairs.empty: raise SystemExit("❌ Selección vacía (ajusta MAX_KM/k/tiles).")

    # 3) Ruteo por motor con escritura incremental
    # Limpia resultados acumulados previos
    res_file = OUTDIR/"matrix_results.csv"
    if res_file.exists(): res_file.unlink()
    if CHECKPOINT.exists(): CHECKPOINT.unlink()

    if ENGINE=="google":
        df_res = run_google_matrix(pairs)
    elif ENGINE=="osrm":
        df_res = run_osrm_table(pairs)
    elif ENGINE=="valhalla":
        df_res = run_valhalla_sources_to_targets(pairs)
    else:
        raise SystemExit("ENGINE no soportado. Usa google|osrm|valhalla.")

    n = len(df_res)
    print(f"Elementos obtenidos: {n}")
    if ENGINE=="google":
        usd,mxn = summarize_cost(n, PRICE_PER_1000_ELEMENTS_USD)
        print(f"Costo estimado: ~{usd:.2f} USD (~{mxn:.2f} MXN)")

    # 4) Export final y enriquecido
    df_final = finalize_exports(pairs, df_res)
    print("\n✓ Archivos en", OUTDIR)
    print(" - pairs_selected.csv / .parquet (si disponible)")
    print(" - matrix_results.csv / .parquet (si disponible)")
    print(" - route_matrix_relevant.xlsx (si hay motor)")
    print(" - poi_all.csv / .parquet (si disponible)")
    print(" - manifest.jsonl (checkpoints por lote)")

if __name__=="__main__":
    main()
