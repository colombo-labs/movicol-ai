"""Ingest TransMilenio GeoJSON data into PostGIS tables."""

import json
import os

import psycopg2

DB_URL = os.environ.get("DATABASE_URL", "postgresql://movicol:movicol@movicol-db:5432/movicol")


def get_conn():
    return psycopg2.connect(DB_URL)


def ingest_troncales(path="models/tm_troncales.geojson"):
    with open(path) as f:
        data = json.load(f)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("TRUNCATE tm_troncales RESTART IDENTITY;")
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        geom = json.dumps(feat.get("geometry"))
        troncal = props.get("troncal", "")
        nombre = props.get("nombre_trazado_troncal", "")
        tipo = props.get("tipo_trazado", "")
        cur.execute(
            """INSERT INTO tm_troncales (troncal, nombre_trazado, tipo_trazado, geom, propiedades)
               VALUES (%s, %s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), %s)""",
            (troncal, nombre, tipo, geom, json.dumps(props)),
        )
    conn.commit()
    print(f"Ingested {len(data.get('features', []))} troncales")
    cur.close()
    conn.close()


def ingest_estaciones(path="models/tm_estaciones.geojson"):
    with open(path) as f:
        data = json.load(f)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("TRUNCATE tm_estaciones RESTART IDENTITY;")
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        geom = json.dumps(feat.get("geometry"))
        nombre = props.get("transmisig2.tecnica.estacion_troncal.nom_est", "")
        troncal = props.get("transmisig2.tecnica.estacion_troncal.nom_troncal", "")
        cur.execute(
            """INSERT INTO tm_estaciones (nombre, troncal, geom, propiedades)
               VALUES (%s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), %s)""",
            (nombre, troncal, geom, json.dumps(props)),
        )
    conn.commit()
    print(f"Ingested {len(data.get('features', []))} estaciones")
    cur.close()
    conn.close()


if __name__ == "__main__":
    ingest_troncales()
    ingest_estaciones()
    print("Done!")
