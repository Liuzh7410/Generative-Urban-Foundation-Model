"""
Build Tokyo UrbanKG in Neo4j.

This script follows the notebook-style flow used in ref/GraphRAG_project_related:
config, data loading, normalization, entity construction, relationship construction.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from shapely import wkt
from shapely.geometry.base import BaseGeometry
from shapely.wkt import dumps
from tqdm import tqdm


# =============================================================================
# 1. Config
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "urban_data"

WARD_PATH = DATA_ROOT / "polygon" / "Tokyo23.shp"
AREA_PATH = DATA_ROOT / "polygon" / "Tokyo23_towns.shp"
ROAD_PATH = DATA_ROOT / "road" / "DRM_Tokyo.shp"
BASE_POI_PATH = DATA_ROOT / "POI" / "Tokyo_POI.csv"
SCHOOL_POI_PATH = DATA_ROOT / "POI" / "Tokyo_school.shp"
STATION_POI_PATH = DATA_ROOT / "POI" / "Tokyo_station.shp"
BLOCK_CENTER_ROOT = DATA_ROOT / "block_center"
OD_PATH = DATA_ROOT / "other" / "Tokyo_Census_OD.csv"

DEFAULT_CRS = "EPSG:4326"

BATCH_SIZE_ENTITY = 10_000
BATCH_SIZE_POI = 10_000
BATCH_SIZE_ROAD = 10_000
BATCH_SIZE_BLOCK = 5_000
BATCH_SIZE_RELATION = 5_000
BATCH_SIZE_OD = 1_000
NEARBY_ROAD_MAX_DISTANCE_M = 20


# =============================================================================
# 2. Neo4j connection
# =============================================================================


class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str, database: str):
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self.driver.close()

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            cypher,
            parameters_=params or {},
            database_=self.database,
        )
        return [record.data() for record in records]


def get_kg() -> Neo4jClient:
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    missing = [
        key
        for key, value in {
            "NEO4J_URI": uri,
            "NEO4J_USERNAME": username,
            "NEO4J_PASSWORD": password,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing Neo4j config in .env: {', '.join(missing)}")

    return Neo4jClient(uri=uri, username=username, password=password, database=database)


# =============================================================================
# 3. Generic helpers
# =============================================================================


def chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    size = max(1, size)
    return [items[i : i + size] for i in range(0, len(items), size)]


def stable_hash_id(prefix: str, *parts: Any, length: int = 16) -> str:
    text = "|".join("" if pd.isna(part) else str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}:{digest}"


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def ensure_gdf_crs(gdf: gpd.GeoDataFrame, default_crs: str = DEFAULT_CRS) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(default_crs)
    return gdf


def geometry_to_wkt(geometry: BaseGeometry | str | None) -> str | None:
    if geometry is None or (isinstance(geometry, float) and pd.isna(geometry)):
        return None
    if isinstance(geometry, str):
        return dumps(wkt.loads(geometry))
    return dumps(geometry)


def read_csv_with_fallback(path: Path, encodings: list[str] | None = None, **kwargs: Any) -> pd.DataFrame:
    encodings = encodings or ["utf-8", "utf-8-sig", "cp932", "shift-jis"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error if last_error else RuntimeError(f"Failed to read CSV: {path}")


# =============================================================================
# 4. Address normalization helpers
# =============================================================================


def zen_to_han(text: Any) -> Any:
    if pd.isna(text) or not text:
        return text

    zen_digits = "０１２３４５６７８９"
    han_digits = "0123456789"
    text = str(text)

    for zen, han in zip(zen_digits, han_digits):
        text = text.replace(zen, han)
    for zen in "−－ー‐":
        text = text.replace(zen, "-")

    return text


def kanji_num_to_arabic(kanji_num: str) -> str:
    kanji_to_num = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if kanji_num == "十":
        return "10"
    if "十" not in kanji_num:
        return str(kanji_to_num.get(kanji_num, kanji_num))
    if kanji_num.startswith("十"):
        return str(10 + kanji_to_num.get(kanji_num[1:], 0))
    left, _, right = kanji_num.partition("十")
    return str(kanji_to_num.get(left, 0) * 10 + kanji_to_num.get(right, 0))


def normalize_address_block(pref: Any, city: Any, choume: Any, block_num: Any) -> str:
    match = re.search(r"(.+?)([一二三四五六七八九十]+)丁目", str(choume))
    if match:
        standardized_choume = f"{match.group(1)} {kanji_num_to_arabic(match.group(2))}丁目"
    else:
        standardized_choume = str(choume)

    block = zen_to_han(str(block_num))
    if "-" in block:
        block = block.split("-")[0]

    return f"{pref} {city} {standardized_choume} {block}".strip()


def normalize_address_poi(address_str: Any) -> str:
    if pd.isna(address_str):
        return ""

    address = str(address_str).replace("　", " ")
    address = zen_to_han(address)
    address = re.sub(r"\s+", " ", address)
    return address.strip()


def parse_address(address: Any) -> dict[str, str | None] | None:
    if pd.isna(address) or not address:
        return None

    parts = str(address).strip().split()
    if len(parts) < 3:
        return None

    result: dict[str, str | None] = {
        "pref": parts[0],
        "city": parts[1],
        "town": None,
        "chome": None,
        "block": None,
        "banchi": None,
    }

    remaining = " ".join(parts[2:])
    match = re.search(r"(.+?)\s+(\d+)丁目\s+([\d\-]+)", remaining)
    if match:
        result["town"] = match.group(1)
        result["chome"] = f"{match.group(2)}丁目"
        block_part = match.group(3)
    else:
        match = re.search(r"(.+?)\s+([\d\-]+)", remaining)
        if not match:
            return result
        result["town"] = match.group(1)
        block_part = match.group(2)

    if "-" in block_part:
        block_banchi = block_part.split("-")
        result["block"] = block_banchi[0]
        result["banchi"] = "-".join(block_banchi[1:])
    else:
        result["block"] = block_part

    return result


def standardize_area_town_name(town_name: Any) -> Any:
    if pd.isna(town_name) or not town_name:
        return town_name

    match = re.search(r"(.+?)([一二三四五六七八九十]+)丁目", str(town_name))
    if not match:
        return str(town_name)
    return f"{match.group(1)} {kanji_num_to_arabic(match.group(2))}丁目"


# =============================================================================
# 5. Data loading and POI normalization
# =============================================================================


def load_ward_area_road() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    ward = ensure_gdf_crs(gpd.read_file(WARD_PATH)).to_crs(DEFAULT_CRS)
    area = ensure_gdf_crs(gpd.read_file(AREA_PATH)).to_crs(DEFAULT_CRS)
    road = ensure_gdf_crs(gpd.read_file(ROAD_PATH)).to_crs(DEFAULT_CRS)
    return ward, area, road


def load_base_poi() -> gpd.GeoDataFrame:
    poi = pd.read_csv(BASE_POI_PATH, low_memory=False)
    geometries = poi["geometry"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(poi, geometry=geometries, crs=DEFAULT_CRS).to_crs(DEFAULT_CRS)

    records = []
    for _, row in gdf.iterrows():
        address = normalize_address_poi(row.get("ADD"))
        lon = float(row.geometry.x)
        lat = float(row.geometry.y)
        poi_id = stable_hash_id("poi", "zenrin", row.get("DN"), address, round(lon, 7), round(lat, 7))
        records.append(
            {
                "id": poi_id,
                "poi_id": poi_id,
                "legacy_poi_id": clean_value(row.get("poi_id")),
                "name": clean_value(row.get("DN")),
                "address": address,
                "geometry": row.geometry,
                "ptypes": [str(row.get("PTYPE"))] if not pd.isna(row.get("PTYPE")) else [],
                "busc": clean_value(row.get("BUSC")),
                "source": "zenrin",
                "fixed_cates": [],
                "lon": lon,
                "lat": lat,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=DEFAULT_CRS)


def load_school_poi() -> gpd.GeoDataFrame:
    school = ensure_gdf_crs(gpd.read_file(SCHOOL_POI_PATH)).to_crs(DEFAULT_CRS)
    records = []
    for _, row in school.iterrows():
        address = normalize_address_poi(row.get("P29_005"))
        lon = float(row.geometry.x)
        lat = float(row.geometry.y)
        poi_id = stable_hash_id("poi", "school", row.get("P29_004"), address, round(lon, 7), round(lat, 7))
        records.append(
            {
                "id": poi_id,
                "poi_id": poi_id,
                "legacy_poi_id": clean_value(row.get("P29_002")),
                "name": clean_value(row.get("P29_004")),
                "address": address,
                "geometry": row.geometry,
                "ptypes": ["37"],
                "busc": None,
                "source": "school",
                "fixed_cates": ["School"],
                "lon": lon,
                "lat": lat,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=DEFAULT_CRS)


def load_station_poi() -> gpd.GeoDataFrame:
    station = ensure_gdf_crs(gpd.read_file(STATION_POI_PATH)).to_crs(DEFAULT_CRS)
    records = []
    for _, row in station.iterrows():
        address = normalize_address_poi(row.get("address"))
        lon = float(row.geometry.x)
        lat = float(row.geometry.y)
        poi_id = stable_hash_id("poi", "station", row.get("name"), address, round(lon, 7), round(lat, 7))
        records.append(
            {
                "id": poi_id,
                "poi_id": poi_id,
                "legacy_poi_id": None,
                "name": clean_value(row.get("name")),
                "address": address,
                "geometry": row.geometry,
                "ptypes": ["22"],
                "busc": None,
                "source": "station",
                "fixed_cates": ["Transportation"],
                "lon": lon,
                "lat": lat,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=DEFAULT_CRS)


def load_all_poi() -> gpd.GeoDataFrame:
    poi = pd.concat(
        [load_base_poi(), load_school_poi(), load_station_poi()],
        ignore_index=True,
    )
    poi = gpd.GeoDataFrame(poi, geometry="geometry", crs=DEFAULT_CRS)
    duplicated = poi["id"].duplicated().sum()
    if duplicated:
        print(f"Warning: duplicate POI ids found and will MERGE in Neo4j: {duplicated}")
    return poi


def load_od() -> pd.DataFrame:
    return pd.read_csv(OD_PATH)


# =============================================================================
# 6. Entity construction
# =============================================================================


def create_constraints(kg: Neo4jClient) -> None:
    constraints = [
        "CREATE CONSTRAINT ward_id_unique IF NOT EXISTS FOR (w:Ward) REQUIRE w.id IS UNIQUE",
        "CREATE CONSTRAINT area_id_unique IF NOT EXISTS FOR (a:Area) REQUIRE a.id IS UNIQUE",
        "CREATE CONSTRAINT poi_id_unique IF NOT EXISTS FOR (p:POI) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT road_id_unique IF NOT EXISTS FOR (r:Road) REQUIRE r.id IS UNIQUE",
        "CREATE CONSTRAINT block_id_unique IF NOT EXISTS FOR (b:Block) REQUIRE b.id IS UNIQUE",
        "CREATE CONSTRAINT cate_id_unique IF NOT EXISTS FOR (c:Cate) REQUIRE c.id IS UNIQUE",
    ]
    for query in constraints:
        kg.query(query)
    print("Constraints ensured.")


def create_wards_batch(kg: Neo4jClient, ward: gpd.GeoDataFrame) -> None:
    rows = []
    for _, row in ward.iterrows():
        ward_code = str(row["N03_007"])
        rows.append(
            {
                "id": f"ward:{ward_code}",
                "ward_code": ward_code,
                "name": clean_value(row["N03_004"]),
                "prefecture": clean_value(row["N03_001"]),
                "geometry": geometry_to_wkt(row.geometry),
            }
        )

    query = """
    UNWIND $batch AS row
    MERGE (w:Ward {id: row.id})
    SET w.ward_code = row.ward_code,
        w.name = row.name,
        w.prefecture = row.prefecture,
        w.geometry = row.geometry
    """
    for batch in chunks(rows, BATCH_SIZE_ENTITY):
        kg.query(query, {"batch": batch})
    print(f"Ward nodes upserted: {len(rows)}")


def create_areas_batch(kg: Neo4jClient, area: gpd.GeoDataFrame) -> None:
    rows = []
    for _, row in area.iterrows():
        town_code = str(row["KEY_CODE"])
        rows.append(
            {
                "id": f"area:{town_code}",
                "town_code": town_code,
                "name": standardize_area_town_name(row["S_NAME"]),
                "raw_name": clean_value(row["S_NAME"]),
                "ward_name": clean_value(row["CITY_NAME"]),
                "prefecture": clean_value(row["PREF_NAME"]),
                "geometry": geometry_to_wkt(row.geometry),
            }
        )

    query = """
    UNWIND $batch AS row
    MERGE (a:Area {id: row.id})
    SET a.town_code = row.town_code,
        a.name = row.name,
        a.raw_name = row.raw_name,
        a.ward_name = row.ward_name,
        a.prefecture = row.prefecture,
        a.geometry = row.geometry
    """
    for batch in chunks(rows, BATCH_SIZE_ENTITY):
        kg.query(query, {"batch": batch})
    print(f"Area nodes upserted: {len(rows)}")


def create_pois_batch(kg: Neo4jClient, poi: gpd.GeoDataFrame) -> None:
    rows = []
    for _, row in poi.iterrows():
        rows.append(
            {
                "id": row["id"],
                "poi_id": row["poi_id"],
                "legacy_poi_id": clean_value(row.get("legacy_poi_id")),
                "name": clean_value(row.get("name")),
                "address": clean_value(row.get("address")),
                "geometry": geometry_to_wkt(row.geometry),
                "ptypes": row.get("ptypes") or [],
                "busc": clean_value(row.get("busc")),
                "source": clean_value(row.get("source")),
                "lon": clean_value(row.get("lon")),
                "lat": clean_value(row.get("lat")),
            }
        )

    query = """
    UNWIND $batch AS row
    MERGE (p:POI {id: row.id})
    SET p.poi_id = row.poi_id,
        p.legacy_poi_id = row.legacy_poi_id,
        p.name = row.name,
        p.address = row.address,
        p.geometry = row.geometry,
        p.ptypes = row.ptypes,
        p.busc = row.busc,
        p.source = row.source,
        p.lon = row.lon,
        p.lat = row.lat
    """
    for i, batch in enumerate(chunks(rows, BATCH_SIZE_POI), start=1):
        kg.query(query, {"batch": batch})
        print(f"  POI batch {i}: {len(batch)}")
    print(f"POI nodes upserted: {len(rows)}")


def create_roads_batch(kg: Neo4jClient, road: gpd.GeoDataFrame) -> None:
    rows = []
    for _, row in road.iterrows():
        road_id = f"road:{row['gid']}"
        rows.append(
            {
                "id": road_id,
                "road_id": str(row["gid"]),
                "rtype": clean_value(row.get("rdclasscd")),
                "length": clean_value(row.get("length")),
                "geometry": geometry_to_wkt(row.geometry),
            }
        )

    query = """
    UNWIND $batch AS row
    MERGE (r:Road {id: row.id})
    SET r.road_id = row.road_id,
        r.rtype = row.rtype,
        r.length = row.length,
        r.geometry = row.geometry
    """
    for i, batch in enumerate(chunks(rows, BATCH_SIZE_ROAD), start=1):
        kg.query(query, {"batch": batch})
        print(f"  Road batch {i}: {len(batch)}")
    print(f"Road nodes upserted: {len(rows)}")


def create_blocks_from_poi(kg: Neo4jClient, poi: gpd.GeoDataFrame) -> None:
    block_counts: dict[str, int] = {}

    for _, row in tqdm(poi.iterrows(), total=len(poi), desc="collecting blocks from POI"):
        parsed = parse_address(row.get("address"))
        if not parsed or not parsed.get("block"):
            continue

        block_parts = [
            parsed["pref"],
            parsed["city"],
            parsed["town"],
            parsed["chome"] or "",
            parsed["block"],
        ]
        block_address = " ".join([part for part in block_parts if part])
        block_counts[block_address] = block_counts.get(block_address, 0) + 1

    rows = [
        {
            "id": stable_hash_id("block", address),
            "address": address,
            "poi_count": count,
        }
        for address, count in block_counts.items()
    ]

    query = """
    UNWIND $batch AS row
    MERGE (b:Block {id: row.id})
    SET b.address = row.address,
        b.poi_count = row.poi_count
    """
    for batch in chunks(rows, BATCH_SIZE_BLOCK):
        kg.query(query, {"batch": batch})
    print(f"Block nodes upserted: {len(rows)}")


def create_blocks_from_center_points(kg: Neo4jClient) -> None:
    rows_by_id: dict[str, dict[str, Any]] = {}
    csv_paths = sorted(BLOCK_CENTER_ROOT.glob("*/*.csv"))

    for path in tqdm(csv_paths, desc="reading block center CSVs"):
        block_df = read_csv_with_fallback(path, encodings=["cp932", "shift-jis", "utf-8", "utf-8-sig"])
        for _, row in block_df.iterrows():
            address = normalize_address_block(
                row["都道府県名"],
                row["市区町村名"],
                row["大字_丁目名"],
                row["街区符号_地番"],
            )
            block_id = stable_hash_id("block", address)
            rows_by_id[block_id] = {
                "id": block_id,
                "address": address,
                "lat": clean_value(row.get("緯度")),
                "lon": clean_value(row.get("経度")),
                "geometry": f"POINT ({row.get('経度')} {row.get('緯度')})",
            }

    rows = list(rows_by_id.values())
    query = """
    UNWIND $batch AS row
    MERGE (b:Block {id: row.id})
    SET b.address = row.address,
        b.lat = row.lat,
        b.lon = row.lon,
        b.geometry = row.geometry
    """
    for batch in chunks(rows, BATCH_SIZE_BLOCK):
        kg.query(query, {"batch": batch})
    print(f"Block center data upserted: {len(rows)}")


# =============================================================================
# 7. Relationship construction
# =============================================================================


def create_relationships_poi_area(kg: Neo4jClient, poi: gpd.GeoDataFrame, area: gpd.GeoDataFrame) -> None:
    poi_gdf = ensure_gdf_crs(poi).to_crs(DEFAULT_CRS)
    area_gdf = ensure_gdf_crs(area).to_crs(DEFAULT_CRS)

    join_df = gpd.sjoin(
        poi_gdf[["id", "geometry"]],
        area_gdf[["KEY_CODE", "geometry"]],
        how="inner",
        predicate="within",
    )[["id", "KEY_CODE"]].drop_duplicates()

    rows = [
        {"poi_id": str(row.id), "area_id": f"area:{row.KEY_CODE}"}
        for row in join_df.itertuples(index=False)
    ]
    query = """
    UNWIND $batch AS row
    MATCH (p:POI {id: row.poi_id})
    MATCH (a:Area {id: row.area_id})
    MERGE (p)-[:locatesAt]->(a)
    """
    for batch in chunks(rows, BATCH_SIZE_RELATION):
        kg.query(query, {"batch": batch})
    print(f"POI-[:locatesAt]->Area relationships created: {len(rows)}")


def create_relationships_area_ward(kg: Neo4jClient, area: gpd.GeoDataFrame, ward: gpd.GeoDataFrame) -> None:
    area_gdf = ensure_gdf_crs(area).to_crs(DEFAULT_CRS)
    ward_gdf = ensure_gdf_crs(ward).to_crs(DEFAULT_CRS)

    join_df = gpd.sjoin(
        area_gdf[["KEY_CODE", "geometry"]],
        ward_gdf[["N03_007", "geometry"]],
        how="inner",
        predicate="within",
    )[["KEY_CODE", "N03_007"]].drop_duplicates()

    rows = [
        {"area_id": f"area:{row.KEY_CODE}", "ward_id": f"ward:{row.N03_007}"}
        for row in join_df.itertuples(index=False)
    ]
    query = """
    UNWIND $batch AS row
    MATCH (a:Area {id: row.area_id})
    MATCH (w:Ward {id: row.ward_id})
    MERGE (a)-[:belongsTo]->(w)
    """
    for batch in chunks(rows, BATCH_SIZE_RELATION):
        kg.query(query, {"batch": batch})
    print(f"Area-[:belongsTo]->Ward relationships created: {len(rows)}")


def create_relationships_road_area(kg: Neo4jClient, road: gpd.GeoDataFrame, area: gpd.GeoDataFrame) -> None:
    road_gdf = ensure_gdf_crs(road).to_crs(DEFAULT_CRS)
    area_gdf = ensure_gdf_crs(area).to_crs(DEFAULT_CRS)
    query = """
    UNWIND $batch AS row
    MATCH (r:Road {id: row.road_id})
    MATCH (a:Area {id: row.area_id})
    MERGE (r)-[:in]->(a)
    """

    total_pairs = 0
    chunk_size = 25_000
    total_chunks = (len(road_gdf) + chunk_size - 1) // chunk_size

    for chunk_index, start in enumerate(range(0, len(road_gdf), chunk_size), start=1):
        road_chunk = road_gdf.iloc[start : start + chunk_size][["gid", "geometry"]]
        pairs = gpd.sjoin(
            road_chunk,
            area_gdf[["KEY_CODE", "geometry"]],
            how="inner",
            predicate="intersects",
        )[["gid", "KEY_CODE"]].drop_duplicates()

        rows = [
            {"road_id": f"road:{row.gid}", "area_id": f"area:{row.KEY_CODE}"}
            for row in pairs.itertuples(index=False)
        ]
        total_pairs += len(rows)
        print(
            f"  Road-Area spatial chunk {chunk_index}/{total_chunks}: "
            f"roads={len(road_chunk)} pairs={len(rows)}"
        )

        for batch_index, batch in enumerate(chunks(rows, BATCH_SIZE_RELATION), start=1):
            kg.query(query, {"batch": batch})
            print(
                f"    Road-Area write batch {chunk_index}.{batch_index}: "
                f"{len(batch)}"
            )

    print(f"Road-[:in]->Area relationships created: {total_pairs}")


def create_relationships_area_border(kg: Neo4jClient, area: gpd.GeoDataFrame) -> None:
    area_gdf = ensure_gdf_crs(area).to_crs(DEFAULT_CRS).reset_index(drop=True)
    spatial_index = area_gdf.sindex

    rels = []
    for i, row in tqdm(area_gdf.iterrows(), total=len(area_gdf), desc="finding area borders"):
        candidate_indices = list(spatial_index.query(row.geometry, predicate="touches"))
        c1 = str(row["KEY_CODE"])
        for j in candidate_indices:
            if i == j:
                continue
            c2 = str(area_gdf.iloc[j]["KEY_CODE"])
            if c1 < c2:
                rels.append({"a1": f"area:{c1}", "a2": f"area:{c2}"})

    query = """
    UNWIND $batch AS row
    MATCH (a1:Area {id: row.a1})
    MATCH (a2:Area {id: row.a2})
    MERGE (a1)-[:borderBy]->(a2)
    MERGE (a2)-[:borderBy]->(a1)
    """
    for batch in chunks(rels, BATCH_SIZE_RELATION):
        kg.query(query, {"batch": batch})
    print(f"Area-[:borderBy]->Area border pairs created: {len(rels)}")


def create_relationships_poi_block(kg: Neo4jClient, poi: gpd.GeoDataFrame) -> None:
    skipped = 0
    query = """
    UNWIND $batch AS row
    MATCH (p:POI {id: row.poi_id})
    MATCH (b:Block {id: row.block_id})
    MERGE (p)-[:locatesAt]->(b)
    """

    total_rows = 0
    chunk_size = 25_000
    total_chunks = (len(poi) + chunk_size - 1) // chunk_size

    for chunk_index, start in enumerate(range(0, len(poi), chunk_size), start=1):
        poi_chunk = poi.iloc[start : start + chunk_size]
        rows = []

        for _, row in tqdm(
            poi_chunk.iterrows(),
            total=len(poi_chunk),
            desc=f"matching POI to Block by address ({chunk_index}/{total_chunks})",
        ):
            parsed = parse_address(row.get("address"))
            if not parsed or not parsed.get("block"):
                skipped += 1
                continue

            block_parts = [
                parsed["pref"],
                parsed["city"],
                parsed["town"],
                parsed["chome"] or "",
                parsed["block"],
            ]
            block_address = " ".join([part for part in block_parts if part])
            rows.append({"poi_id": row["id"], "block_id": stable_hash_id("block", block_address)})

        total_rows += len(rows)
        print(
            f"  POI-Block match chunk {chunk_index}/{total_chunks}: "
            f"pois={len(poi_chunk)} matched={len(rows)} skipped_so_far={skipped}"
        )

        for batch_index, batch in enumerate(chunks(rows, BATCH_SIZE_RELATION), start=1):
            kg.query(query, {"batch": batch})
            print(
                f"    POI-Block write batch {chunk_index}.{batch_index}: "
                f"{len(batch)}"
            )

    print(f"POI-[:locatesAt]->Block relationships created: {total_rows}; skipped: {skipped}")


def create_relationships_block_area(kg: Neo4jClient, area: gpd.GeoDataFrame) -> None:
    area_mapping = {}
    for _, row in area.iterrows():
        town_std = standardize_area_town_name(str(row["S_NAME"]))
        area_key = f"{row['PREF_NAME']} {row['CITY_NAME']} {town_std}".strip()
        area_mapping[area_key] = f"area:{row['KEY_CODE']}"

    blocks = kg.query("MATCH (b:Block) RETURN b.id AS id, b.address AS address")
    rels = []
    skipped = 0

    for block in tqdm(blocks, total=len(blocks), desc="matching Block to Area by address"):
        parsed = parse_address(block["address"])
        if not parsed or not parsed.get("town"):
            skipped += 1
            continue

        town_std = standardize_area_town_name(parsed["town"])
        if parsed.get("chome"):
            area_key = f"{parsed['pref']} {parsed['city']} {town_std} {parsed['chome']}"
        else:
            area_key = f"{parsed['pref']} {parsed['city']} {town_std}"

        area_id = area_mapping.get(area_key)
        if area_id:
            rels.append({"block_id": block["id"], "area_id": area_id})
        else:
            skipped += 1

    query = """
    UNWIND $batch AS row
    MATCH (b:Block {id: row.block_id})
    MATCH (a:Area {id: row.area_id})
    MERGE (b)-[:belongsTo]->(a)
    """
    for batch in chunks(rels, BATCH_SIZE_RELATION):
        kg.query(query, {"batch": batch})
    print(f"Block-[:belongsTo]->Area relationships created: {len(rels)}; skipped: {skipped}")


def create_relationships_poi_nearby_road(
    kg: Neo4jClient,
    poi: gpd.GeoDataFrame,
    road: gpd.GeoDataFrame,
    max_distance_m: int = NEARBY_ROAD_MAX_DISTANCE_M,
) -> None:
    poi_gdf = ensure_gdf_crs(poi).to_crs(DEFAULT_CRS)
    road_gdf = ensure_gdf_crs(road).to_crs(DEFAULT_CRS)
    proj_crs = poi_gdf.estimate_utm_crs()
    poi_proj = poi_gdf.to_crs(proj_crs)
    road_proj = road_gdf.to_crs(proj_crs)
    query = """
    UNWIND $batch AS row
    MATCH (p:POI {id: row.poi_id})
    MATCH (r:Road {id: row.road_id})
    MERGE (p)-[rel:nearby]->(r)
    SET rel.distance_m = row.distance_m
    """

    total_rows = 0
    chunk_size = 25_000
    total_chunks = (len(poi_proj) + chunk_size - 1) // chunk_size

    for chunk_index, start in enumerate(range(0, len(poi_proj), chunk_size), start=1):
        poi_chunk = poi_proj.iloc[start : start + chunk_size][["id", "geometry"]]

        near_df = gpd.sjoin_nearest(
            poi_chunk,
            road_proj[["gid", "geometry"]],
            how="left",
            max_distance=max_distance_m,
            distance_col="dist_m",
        )[["id", "gid", "dist_m"]].dropna().drop_duplicates()

        rows = [
            {"poi_id": row.id, "road_id": f"road:{row.gid}", "distance_m": float(row.dist_m)}
            for row in near_df.itertuples(index=False)
        ]
        total_rows += len(rows)
        print(
            f"  POI-Road spatial chunk {chunk_index}/{total_chunks}: "
            f"pois={len(poi_chunk)} pairs={len(rows)}"
        )

        for batch_index, batch in enumerate(chunks(rows, BATCH_SIZE_RELATION), start=1):
            kg.query(query, {"batch": batch})
            print(
                f"    POI-Road write batch {chunk_index}.{batch_index}: "
                f"{len(batch)}"
            )

    print(f"POI-[:nearby]->Road relationships created: {total_rows}")


categories_ranges = {
    "Factory": [(100000, 1900000)],
    "Business": [(1900000, 2200000), (2500000, 2700000), (2800000, 2900000)],
    "Transportation": [(2200000, 2200400)],
    "Amusement": [(2900000, 2919000), (3100000, 3122008), (3178000, 3212000)],
    "Meal": [(3123000, 3178000)],
    "Accommodation": [(3213000, 3226000)],
    "Medical_service": [(3300000, 3340000)],
    "School": [(3340000, 3340000), (3700000, 3800000)],
    "Welfare": [(3341000, 3400000)],
    "Life_related": [(3000000, 3008000), (3400000, 3600000), (3602000, 3615000)],
    "Car_service": [(3600000, 3602000), (3615000, 3615000)],
    "Government": [(3800000, 3900000)],
    "Shopping": [(3400000, 3540000), (3540012, 3561003)],
    "others": [],
}


def extract_busc_codes(busc_value: Any) -> list[int]:
    if busc_value is None:
        return []
    if isinstance(busc_value, str):
        raw_values = busc_value.split(",")
    elif isinstance(busc_value, (list, tuple, set, np.ndarray, pd.Series)):
        raw_values = list(busc_value)
    else:
        raw_values = [busc_value]

    codes = []
    for value in raw_values:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        match = re.search(r"\d+", str(value).strip())
        if match:
            codes.append(int(match.group()))
    return codes


def map_busc_to_cates(busc_codes: list[int]) -> list[str]:
    matched = set()
    for code in busc_codes:
        for cate, ranges in categories_ranges.items():
            if cate == "others":
                continue
            for low, high in ranges:
                if low <= code <= high:
                    matched.add(cate)
                    break
    if not matched:
        matched.add("others")
    return sorted(matched)


def create_relationships_poi_cate(kg: Neo4jClient, poi: gpd.GeoDataFrame) -> None:
    rows = []
    for _, row in tqdm(poi.iterrows(), total=len(poi), desc="mapping POI to Cate"):
        fixed_cates = row.get("fixed_cates") or []
        cates = list(fixed_cates) if fixed_cates else map_busc_to_cates(extract_busc_codes(row.get("busc")))
        for cate in cates:
            rows.append(
                {
                    "poi_id": row["id"],
                    "cate_id": stable_hash_id("cate", cate),
                    "cate_name": cate,
                }
            )

    query = """
    UNWIND $batch AS row
    MATCH (p:POI {id: row.poi_id})
    MERGE (c:Cate {id: row.cate_id})
    SET c.name = row.cate_name
    MERGE (p)-[:hasType]->(c)
    """
    for batch in chunks(rows, BATCH_SIZE_RELATION):
        kg.query(query, {"batch": batch})
    print(f"POI-[:hasType]->Cate relationships created: {len(rows)}")


OD_ALIASES = {
    "S05b_010": "rail_total",
    "S05b_016": "bus_total",
    "S05b_022": "car_total",
    "S05b_028": "motorcycle_total",
    "S05b_034": "walk_total",
    "S05b_035": "total_trips",
}


def create_relationships_ward_od_flow(kg: Neo4jClient, od: pd.DataFrame, ward: gpd.GeoDataFrame) -> None:
    ward_name_to_id = {
        str(row["N03_004"]): f"ward:{row['N03_007']}"
        for _, row in ward.iterrows()
    }
    trip_cols = [col for col in od.columns if col.startswith("S05b_")]
    rows = []
    unmatched = []

    for _, row in od.iterrows():
        origin_id = ward_name_to_id.get(str(row["orig_ward"]))
        dest_id = ward_name_to_id.get(str(row["dest_ward"]))
        if not origin_id or not dest_id:
            unmatched.append((row["orig_ward"], row["dest_ward"]))
            continue

        props = {col: int(row[col]) for col in trip_cols}
        for raw_col, alias in OD_ALIASES.items():
            if raw_col in props:
                props[alias] = props[raw_col]

        rows.append(
            {
                "origin_id": origin_id,
                "dest_id": dest_id,
                "orig_ward": row["orig_ward"],
                "dest_ward": row["dest_ward"],
                "props": props,
            }
        )

    query = """
    UNWIND $batch AS row
    MATCH (o:Ward {id: row.origin_id})
    MATCH (d:Ward {id: row.dest_id})
    MERGE (o)-[r:hasODFlow]->(d)
    SET r += row.props,
        r.orig_ward = row.orig_ward,
        r.dest_ward = row.dest_ward
    """
    for batch in chunks(rows, BATCH_SIZE_OD):
        kg.query(query, {"batch": batch})

    print(f"Ward-[:hasODFlow]->Ward relationships created: {len(rows)}")
    if unmatched:
        print(f"Unmatched OD rows: {len(unmatched)}")
        print(f"  Sample unmatched OD rows: {unmatched[:10]}")


# =============================================================================
# 8. Main pipeline
# =============================================================================


START_STAGES = (
    "all",
    "blocks",
    "relationships",
    "road-area",
    "area-border",
    "poi-block",
    "block-area",
    "poi-road",
    "poi-cate",
    "od-flow",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Tokyo UrbanKG in Neo4j.")
    parser.add_argument(
        "--start-from",
        choices=START_STAGES,
        default="all",
        help=(
            "Pipeline entry point: "
            "'all' runs the full pipeline, "
            "'blocks' skips Ward/Area/POI/Road creation and starts from Block creation, "
            "'relationships' skips all entity creation and starts from relationship creation, "
            "and later stages start from the named relationship step."
        ),
    )
    return parser.parse_args()


def print_counts(kg: Neo4jClient) -> None:
    result = kg.query(
        """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY label
        """
    )
    print("\nNode counts:")
    for row in result:
        print(f"  {row['label']}: {row['count']}")

    result = kg.query(
        """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(r) AS count
        ORDER BY type
        """
    )
    print("\nRelationship counts:")
    for row in result:
        print(f"  {row['type']}: {row['count']}")


def run_relationship_pipeline(
    kg: Neo4jClient,
    poi: gpd.GeoDataFrame,
    area: gpd.GeoDataFrame,
    ward: gpd.GeoDataFrame,
    road: gpd.GeoDataFrame,
    od: pd.DataFrame,
    start_from: str,
) -> None:
    steps = [
        ("relationships", None),
        ("poi-area", lambda: create_relationships_poi_area(kg, poi, area)),
        ("area-ward", lambda: create_relationships_area_ward(kg, area, ward)),
        ("road-area", lambda: create_relationships_road_area(kg, road, area)),
        ("area-border", lambda: create_relationships_area_border(kg, area)),
        ("poi-block", lambda: create_relationships_poi_block(kg, poi)),
        ("block-area", lambda: create_relationships_block_area(kg, area)),
        ("poi-road", lambda: create_relationships_poi_nearby_road(kg, poi, road)),
        ("poi-cate", lambda: create_relationships_poi_cate(kg, poi)),
        ("od-flow", lambda: create_relationships_ward_od_flow(kg, od, ward)),
    ]
    step_order = [name for name, _ in steps]

    if start_from not in step_order:
        raise ValueError(f"Unsupported relationship start stage: {start_from}")

    start_index = step_order.index(start_from)
    if start_from == "relationships":
        start_index = 1

    for name, fn in steps[start_index:]:
        if fn is None:
            continue
        print(f"  -> {name}")
        fn()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("Tokyo UrbanKG construction")
    print("=" * 80)
    print(f"Start stage: {args.start_from}")

    print("\n[1/6] Loading data")
    ward, area, road = load_ward_area_road()
    poi = load_all_poi()
    od = load_od()
    print(f"  Ward rows: {len(ward)}")
    print(f"  Area rows: {len(area)}")
    print(f"  Road rows: {len(road)}")
    print(f"  POI rows: {len(poi)}")
    print(f"  OD rows: {len(od)}")
    print(f"  POI sources: {poi['source'].value_counts().to_dict()}")

    print("\n[2/6] Connecting Neo4j")
    kg = get_kg()

    try:
        print("\n[3/6] Creating constraints")
        create_constraints(kg)

        print("\n[4/6] Creating entities")
        if args.start_from == "all":
            create_wards_batch(kg, ward)
            create_areas_batch(kg, area)
            create_pois_batch(kg, poi)
            create_roads_batch(kg, road)
            create_blocks_from_poi(kg, poi)
            create_blocks_from_center_points(kg)
        elif args.start_from == "blocks":
            print("  Skipping Ward/Area/POI/Road creation.")
            create_blocks_from_poi(kg, poi)
            create_blocks_from_center_points(kg)
        else:
            print("  Skipping all entity creation.")

        print("\n[5/6] Creating relationships")
        if args.start_from in ("all", "blocks"):
            relationship_start = "relationships"
        elif args.start_from == "relationships":
            relationship_start = "relationships"
        else:
            relationship_start = args.start_from

        run_relationship_pipeline(
            kg=kg,
            poi=poi,
            area=area,
            ward=ward,
            road=road,
            od=od,
            start_from=relationship_start,
        )

        print("\n[6/6] Summary")
        print_counts(kg)
    finally:
        kg.close()


if __name__ == "__main__":
    main()
