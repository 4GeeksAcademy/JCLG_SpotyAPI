import os
from typing import List, Dict, Any

import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import sqlite3


BASE_URL = "https://ws.audioscrobbler.com/2.0/"


def get_api_key() -> str:
    """Lee la API key desde .env y valida que exista."""
    load_dotenv()
    api_key = os.getenv("LASTFM_API_KEY")
    if not api_key:
        raise ValueError("No se encontró LASTFM_API_KEY en el archivo .env")
    return api_key


def lastfm_get(params: Dict[str, Any]) -> Dict[str, Any]:
    """Helper para hacer GET a Last.fm y devolver JSON como dict."""
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_top_artists(api_key: str, limit: int = 50) -> Dict[str, Any]:
    """Obtiene el ranking global de artistas."""
    params = {
        "method": "chart.gettopartists",
        "api_key": api_key,
        "format": "json",
        "limit": limit,
    }
    return lastfm_get(params)


def top_artists_to_df(data: Dict[str, Any]) -> pd.DataFrame:
    """Convierte el JSON de top artists a un DataFrame limpio."""
    if "artists" not in data or "artist" not in data["artists"]:
        raise ValueError(f"Respuesta inesperada. Claves raíz: {list(data.keys())}")

    artists = data["artists"]["artist"]  # lista de dicts
    df = pd.DataFrame(artists)

    # Columnas relevantes
    df = df[["name", "listeners", "playcount", "url"]].copy()

    # Convertir strings -> números (clave para análisis/ordenación)
    df["listeners"] = pd.to_numeric(df["listeners"], errors="coerce")
    df["playcount"] = pd.to_numeric(df["playcount"], errors="coerce")

    # Métrica de negocio: engagement = reproducciones por oyente
    df["engagement"] = df["playcount"] / df["listeners"]

    return df


def get_artist_top_tags(artist_name: str, api_key: str, limit: int = 5) -> List[str]:
    """Obtiene los top tags (estilo/etiquetas) de un artista."""
    params = {
        "method": "artist.gettoptags",
        "artist": artist_name,
        "api_key": api_key,
        "format": "json",
    }
    data = lastfm_get(params)
    tags = data.get("toptags", {}).get("tag", [])
    return [t.get("name") for t in tags[:limit] if t.get("name")]


def enrich_with_tags(df: pd.DataFrame, api_key: str, limit_per_artist: int = 5) -> pd.DataFrame:
    """Añade columna top_tags (lista) al DataFrame."""
    df = df.copy()
    df["top_tags"] = df["name"].apply(lambda a: get_artist_top_tags(a, api_key, limit=limit_per_artist))
    return df


def save_to_sqlite(df: pd.DataFrame, db_path: str = "lastfm_top_artists.db", table: str = "top_artists") -> None:
    """Guarda el DataFrame en una base SQLite."""
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table, con=conn, if_exists="replace", index=False)
    finally:
        conn.close()


def plot_scatter_listeners_vs_playcount(df: pd.DataFrame) -> float:
    """Scatter plot y correlación listeners vs playcount."""
    corr = df["listeners"].corr(df["playcount"])

    plt.figure()
    plt.scatter(df["listeners"], df["playcount"])
    plt.xlabel("Listeners (oyentes)")
    plt.ylabel("Playcount (reproducciones)")
    plt.title(f"Listeners vs Playcount (corr={corr:.3f})")
    plt.show()

    return corr


def plot_top_tags_bar(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Cuenta tags más comunes y los grafica."""
    tags_series = df["top_tags"].explode().dropna()
    counts = tags_series.value_counts().head(top_n)

    plt.figure()
    plt.barh(counts.index[::-1], counts.values[::-1])
    plt.xlabel("Número de artistas")
    plt.title(f"Top {top_n} tags más comunes (Last.fm Top Artists)")
    plt.show()

    return counts.reset_index().rename(columns={"index": "tag", "count": "artists_count"})


def main():
    api_key = get_api_key()

    # 1) Extraer datos
    data = fetch_top_artists(api_key, limit=50)

    # 2) Transformar a DataFrame
    df = top_artists_to_df(data)
    print("Número de artistas:", len(df))

    # 3) Enriquecer con tags
    df = enrich_with_tags(df, api_key, limit_per_artist=5)

    # 4) Ordenación ejemplo: menor a mayor playcount
    df_sorted = df.sort_values("playcount", ascending=True).reset_index(drop=True)
    print("\nTop 3 (menos playcount dentro del top 50):")
    print(df_sorted[["name", "listeners", "playcount", "engagement"]].head(3))

    # 5) Análisis y gráficos (opcional en ejecución; aquí se muestran)
    corr = plot_scatter_listeners_vs_playcount(df)
    print(f"\nCorrelación Pearson listeners vs playcount: {corr:.3f}")

    _ = plot_top_tags_bar(df, top_n=10)

    # 6) Guardar en SQLite
    save_to_sqlite(df, db_path="lastfm_top_artists.db", table="top_artists")
    print("\nGuardado en SQLite: lastfm_top_artists.db (tabla: top_artists)")


if __name__ == "__main__":
    main()

