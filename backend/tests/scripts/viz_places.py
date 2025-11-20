# viz_places.py
# Uso:
#   python viz_places.py
#   # o bien: python viz_places.py --csv "ruta/a/places_sample.csv"

import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ✅ Ruta correcta al CSV por defecto (desde tests/scripts/ → proyecto/backend)
CSV_PATH = Path(__file__).resolve().parents[2] / "app" / "salidas" / "places_sample.csv"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(CSV_PATH),
                        help="Ruta al CSV de lugares (por defecto backend/app/db/salidas/places_sample.csv)")
    args = parser.parse_args()

    path = Path(args.csv).resolve()
    if not path.exists():
        sys.exit(f"❌ No encontré el archivo: {path}. Pasa la ruta correcta con --csv")

    print("Usando CSV en:", path)

    df = pd.read_csv(path)

    # Normaliza columnas esperadas
    expected = ["place_id","name","address","lat","lng","types","rating","user_ratings_total","phone","website"]
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    # Limpieza básica
    df_geo = df.dropna(subset=["lat","lng"]).copy()

    # Guarda salidas junto al CSV
    outdir = path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Métricas básicas
    summary = {
        "total_registros": int(len(df)),
        "con_coordenadas": int(len(df_geo)),
        "sin_coordenadas": int(len(df) - len(df_geo)),
        "rating_promedio": float(df["rating"].dropna().mean()) if "rating" in df.columns else np.nan,
        "opiniones_mediana": float(df["user_ratings_total"].dropna().median()) if "user_ratings_total" in df.columns else np.nan,
    }
    summary_df = pd.DataFrame([summary])
    summary_path = outdir / "places_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ---- Top 50 por calidad (rating y volumen de opiniones)
    top50 = (
        df.sort_values(["rating","user_ratings_total"], ascending=[False, False])
          .dropna(subset=["rating"])
          .head(50)
          .copy()
    )
    top50_path = outdir / "places_top50.csv"
    top50.to_csv(top50_path, index=False)

    # ---- Top 10 tipos
    types_series = (
        df["types"].dropna().astype(str)
          .apply(lambda s: [t.strip() for t in s.split(",") if t.strip()])
    )
    exploded = pd.Series([t for sub in types_series for t in sub], name="type")
    type_counts = exploded.value_counts().head(10)

    # ---- Plot 1: Top 10 tipos
    plt.figure()
    type_counts.sort_values(ascending=True).plot(kind="barh")
    plt.title("Top 10 tipos de lugar (conteo)")
    plt.xlabel("Conteo")
    plt.ylabel("Tipo")
    plt.tight_layout()
    plot1_path = outdir / "plot_top_tipos.png"
    plt.savefig(plot1_path, dpi=150)
    plt.close()

    # ---- Plot 2: Dispersión geográfica
    plt.figure()
    plt.scatter(df_geo["lng"], df_geo["lat"], s=16, alpha=0.8)
    plt.title("Distribución geográfica de puntos (Lng vs Lat)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plot2_path = outdir / "plot_dispersion.png"
    plt.savefig(plot2_path, dpi=150)
    plt.close()

    # ---- Vista rápida en consola
    print("✅ Visualización completada")
    print(f"- Archivo base: {path}")
    print(f"- Resumen:      {summary_path}")
    print(f"- Top 50:       {top50_path}")
    print(f"- Plot tipos:   {plot1_path}")
    print(f"- Plot mapa:    {plot2_path}")
    print("\nPrimeras filas:")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
