# route_matrix_stream_ok.py
# Requisitos: pip install requests pandas

import os, re, json
from pathlib import Path
import requests
import pandas as pd

URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
FIELD_MASK = "originIndex,destinationIndex,status,condition,distanceMeters,duration,staticDuration"

API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY") or "TU_API_KEY_AQUI"
if (not API_KEY) or API_KEY == "TU_API_KEY_AQUI" or re.search(r"[^A-Za-z0-9\-\_]", API_KEY):
    raise SystemExit("❌ API key faltante o con caracteres inválidos (nada de ñ/acentos).")

ORIGINS = [
    {"location": {"latLng": {"latitude": 19.287, "longitude": -99.655}}},
    {"location": {"latLng": {"latitude": 19.292, "longitude": -99.661}}},
    {"location": {"latLng": {"latitude": 19.280, "longitude": -99.645}}},
]
DESTINATIONS = [
    {"location": {"latLng": {"latitude": 19.300, "longitude": -99.642}}},
    {"location": {"latLng": {"latitude": 19.275, "longitude": -99.660}}},
    {"location": {"latLng": {"latitude": 19.289, "longitude": -99.671}}},
    {"location": {"latLng": {"latitude": 19.295, "longitude": -99.650}}},
    {"location": {"latLng": {"latitude": 19.282, "longitude": -99.636}}},
]

def build_body():
    return {
        "origins":      [{"waypoint": o} for o in ORIGINS],
        "destinations": [{"waypoint": d} for d in DESTINATIONS],
        "travelMode": "DRIVE",
        "routingPreference": "ROUTING_PREFERENCE_UNSPECIFIED",  # o "TRAFFIC_AWARE"
        "regionCode": "MX",
        "units": "METRIC",
    }

def parse_stream_any(lines, save_path: Path, preview_count=10):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    raw_lines, preview = [], []

    with open(save_path, "w", encoding="utf-8", newline="") as f:
        for raw in lines:
            if raw is None:
                continue
            s = raw if isinstance(raw, str) else raw.decode("utf-8", errors="ignore")
            if not s:
                continue
            f.write(s + "\n")
            raw_lines.append(s)
            if len(preview) < preview_count:
                preview.append(s)

    whole = "".join(raw_lines).strip()
    objs = []
    if whole:
        if whole.startswith(")]}'"):
            whole = whole[4:].lstrip()
        if whole.startswith("["):
            try:
                parsed = json.loads(whole)
                if isinstance(parsed, list):
                    return [o for o in parsed if isinstance(o, dict)], preview
            except json.JSONDecodeError:
                pass

    buf = []
    depth = 0
    in_str = False
    esc = False
    stream_text = "\n".join(raw_lines)

    for ch in stream_text:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            buf.append(ch)
            continue

        if ch == "{":
            depth += 1
            buf.append(ch)
            continue

        if ch == "}":
            depth -= 1
            buf.append(ch)
            if depth == 0 and buf:
                candidate = "".join(buf).strip()
                try:
                    if candidate.startswith(")]}'"):
                        candidate = candidate[4:].lstrip()
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        objs.append(obj)
                except json.JSONDecodeError:
                    pass
                buf = []
            continue

        if depth > 0:
            buf.append(ch)

    return objs, preview

def iso_to_s(x): 
    return float(x[:-1]) if isinstance(x, str) and x.endswith("s") else None

def get_lat(o): return o["location"]["latLng"]["latitude"]
def get_lng(o): return o["location"]["latLng"]["longitude"]

def main():
    outdir = Path("salidas_matrix"); outdir.mkdir(parents=True, exist_ok=True)

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": FIELD_MASK,
    }
    body = build_body()

    print("→ POST", URL)
    print("FIELD_MASK:", FIELD_MASK)
    print("Body:", json.dumps(body))

    # Streaming activado (pero parser robusto)
    with requests.post(URL, headers=headers, json=body, stream=True, timeout=60) as r:
        status = r.status_code
        print("HTTP status:", status)
        if status != 200:
            print("ERROR BODY:\n", r.text)
            r.raise_for_status()

        rows, preview = parse_stream_any(
            r.iter_lines(decode_unicode=True),
            save_path=outdir/"raw_response.ndjson",
            preview_count=10
        )

    print("\n--- Vista previa de líneas crudas (máx 10) ---")
    if preview:
        for i, ln in enumerate(preview, 1):
            print(f"{i:02d}: {ln}")
    else:
        print("(sin líneas)")

    if not rows:
        raise SystemExit("❌ No se parseó ningún objeto. Revisa salidas_matrix/raw_response.ndjson.")

    norm = []
    for rj in rows:
        oi, di = rj.get("originIndex"), rj.get("destinationIndex")
        norm.append({
            "origin_index": oi,
            "destination_index": di,
            "status_code": (rj.get("status") or {}).get("code", 0),
            "condition": rj.get("condition"),
            "distance_m": rj.get("distanceMeters"),
            "duration_s": iso_to_s(rj.get("duration")),
            "static_duration_s": iso_to_s(rj.get("staticDuration")),
            "o_lat": get_lat(ORIGINS[oi]),
            "o_lng": get_lng(ORIGINS[oi]),
            "d_lat": get_lat(DESTINATIONS[di]),
            "d_lng": get_lng(DESTINATIONS[di]),
        })

    df = pd.DataFrame(norm)
    df["duration_min"] = df["duration_s"].apply(lambda x: x/60 if x is not None else None)
    df["static_duration_min"] = df["static_duration_s"].apply(lambda x: x/60 if x is not None else None)

    mat_time = df.pivot(index="origin_index", columns="destination_index", values="duration_min")
    mat_dist = df.pivot(index="origin_index", columns="destination_index", values="distance_m")

    df.to_csv(outdir/"matrix_long.csv", index=False)
    mat_time.to_csv(outdir/"matrix_wide_time_min.csv", float_format="%.2f")
    mat_dist.to_csv(outdir/"matrix_wide_dist_m.csv", float_format="%.0f")

    print("\n✓ Archivos guardados en", outdir)
    print(" - raw_response.ndjson")
    print(" - matrix_long.csv")
    print(" - matrix_wide_time_min.csv")
    print(" - matrix_wide_dist_m.csv")
    print("\nMatriz (min):")
    print(mat_time.round(2))
    print("\nMatriz (m):")
    print(mat_dist.round(0))

if __name__ == "__main__":
    main()
