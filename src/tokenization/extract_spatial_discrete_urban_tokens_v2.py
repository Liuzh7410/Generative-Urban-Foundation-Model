"""
Extract spatially enriched discrete UrbanToken V2 sequences from Neo4j UrbanKG.

V2 keeps the Area-centered idea from V1, but adds two changes:

1. Spatial tokens computed from WKT geometries stored in Neo4j.
   Neo4j geometries are assumed to be EPSG:4326 lon/lat WKT, matching
   scripts/build_tokyo_ukg.py. Metric features are computed after projecting
   to EPSG:32654, a UTM zone suitable for Tokyo.
2. Structured output sections plus a linearized token sequence.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pyproj import Transformer
from shapely import wkt
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from extract_discrete_urban_tokens import (
    Neo4jClient,
    QuantileBinner,
    attr_token,
    get_kg,
    id_token,
    token_fragment,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "token/urban_tokens_area_centered_v2.jsonl"
SOURCE_CRS = "EPSG:4326"
METRIC_CRS = "EPSG:32654"


@dataclass(frozen=True)
class AreaRecord:
    area_id: str
    area_name: str | None
    town_code: str | None
    ward_id: str | None
    ward_name: str | None
    ward_code: str | None
    geometry_wkt: str | None
    ward_geometry_wkt: str | None


@dataclass
class AreaSpatial:
    geom: BaseGeometry
    geom_m: BaseGeometry
    centroid: Point
    centroid_m: Point
    area_m2: float
    area_km2: float
    perimeter_m: float
    compactness: float
    grid_token: str
    ward_center_dist_m: float | None
    ward_relative_dir: str | None


@dataclass(frozen=True)
class Anchor:
    poi_id: str
    name: str | None
    source: str
    x: float
    y: float


class GeometryProjector:
    def __init__(self, source_crs: str = SOURCE_CRS, metric_crs: str = METRIC_CRS):
        self.transformer = Transformer.from_crs(source_crs, metric_crs, always_xy=True)

    def project_geom(self, geom: BaseGeometry) -> BaseGeometry:
        return shapely_transform(self.transformer.transform, geom)

    def project_point(self, lon: float, lat: float) -> tuple[float, float]:
        x, y = self.transformer.transform(lon, lat)
        return float(x), float(y)

    def project_points(self, lons: list[float], lats: list[float]) -> tuple[list[float], list[float]]:
        xs, ys = self.transformer.transform(lons, lats)
        return list(xs), list(ys)


def parse_wkt(value: str | None) -> BaseGeometry | None:
    if not value:
        return None
    try:
        return wkt.loads(value)
    except Exception:
        return None


def direction_token(dx: float, dy: float) -> str:
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return "DIR_CENTER"

    angle = math.degrees(math.atan2(dy, dx))
    directions = [
        ("E", -22.5, 22.5),
        ("NE", 22.5, 67.5),
        ("N", 67.5, 112.5),
        ("NW", 112.5, 157.5),
        ("W", 157.5, 180.0),
        ("W", -180.0, -157.5),
        ("SW", -157.5, -112.5),
        ("S", -112.5, -67.5),
        ("SE", -67.5, -22.5),
    ]
    for label, low, high in directions:
        if low <= angle < high:
            return f"DIR_{label}"
    return "DIR_UNKNOWN"


def distance_bucket(distance_m: float | int | None) -> str:
    if distance_m is None:
        return "DIST_UNKNOWN"
    value = float(distance_m)
    if value <= 100:
        return "DIST_0_100M"
    if value <= 250:
        return "DIST_100_250M"
    if value <= 500:
        return "DIST_250_500M"
    if value <= 1000:
        return "DIST_500_1000M"
    if value <= 2000:
        return "DIST_1_2KM"
    if value <= 5000:
        return "DIST_2_5KM"
    return "DIST_5KM_PLUS"


def grid_token(lon: float, lat: float, precision: int = 2) -> str:
    lon_i = int(round(lon * (10**precision)))
    lat_i = int(round(lat * (10**precision)))
    return f"GRID_{lat_i}_{lon_i}"


def density(value: float | int | None, area_km2: float) -> float:
    if not area_km2:
        return 0.0
    return (0.0 if value is None else float(value)) / area_km2


def fetch_areas(kg: Neo4jClient, limit: int | None) -> list[AreaRecord]:
    cypher = """
    MATCH (a:Area)
    OPTIONAL MATCH (a)-[:belongsTo]->(w:Ward)
    OPTIONAL MATCH (name_w:Ward {name: a.ward_name})
    WITH a, coalesce(w, name_w) AS ward
    RETURN a.id AS area_id,
           a.name AS area_name,
           a.town_code AS town_code,
           ward.id AS ward_id,
           coalesce(ward.name, a.ward_name) AS ward_name,
           ward.ward_code AS ward_code,
           a.geometry AS geometry_wkt,
           ward.geometry AS ward_geometry_wkt
    ORDER BY a.id
    """
    if limit is not None:
        cypher += "\nLIMIT $limit"

    rows = kg.query(cypher, {"limit": limit} if limit is not None else {})
    return [
        AreaRecord(
            area_id=row["area_id"],
            area_name=row.get("area_name"),
            town_code=row.get("town_code"),
            ward_id=row.get("ward_id"),
            ward_name=row.get("ward_name"),
            ward_code=row.get("ward_code"),
            geometry_wkt=row.get("geometry_wkt"),
            ward_geometry_wkt=row.get("ward_geometry_wkt"),
        )
        for row in rows
    ]


def fetch_poi_points(kg: Neo4jClient, area_ids: list[str]) -> list[dict[str, Any]]:
    return kg.query(
        """
        MATCH (p:POI)-[:locatesAt]->(a:Area)
        WHERE a.id IN $area_ids AND p.lon IS NOT NULL AND p.lat IS NOT NULL
        MATCH (p)-[:hasType]->(c:Cate)
        RETURN a.id AS area_id,
               p.id AS poi_id,
               p.lon AS lon,
               p.lat AS lat,
               c.id AS cate_id,
               c.name AS cate_name
        ORDER BY area_id, cate_name
        """,
        {"area_ids": area_ids},
    )


def fetch_road_types(kg: Neo4jClient, area_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    rows = kg.query(
        """
        MATCH (r:Road)-[:in]->(a:Area)
        WHERE a.id IN $area_ids
        RETURN a.id AS area_id,
               coalesce(toString(r.rtype), "UNKNOWN") AS road_type,
               count(DISTINCT r) AS road_count,
               sum(coalesce(r.length, 0.0)) AS road_length
        ORDER BY area_id, road_count DESC, road_type
        """,
        {"area_ids": area_ids},
    )
    return group_by_area(rows)


def fetch_neighbors(kg: Neo4jClient, area_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    rows = kg.query(
        """
        MATCH (a:Area)-[:borderBy]->(n:Area)
        WHERE a.id IN $area_ids
        RETURN a.id AS area_id,
               n.id AS neighbor_id,
               n.name AS neighbor_name,
               n.geometry AS neighbor_geometry_wkt
        ORDER BY area_id, neighbor_id
        """,
        {"area_ids": area_ids},
    )
    return group_by_area(rows)


def fetch_blocks(kg: Neo4jClient, area_ids: list[str]) -> dict[str, dict[str, Any]]:
    rows = kg.query(
        """
        MATCH (b:Block)-[:belongsTo]->(a:Area)
        WHERE a.id IN $area_ids
        RETURN a.id AS area_id,
               count(DISTINCT b) AS block_count,
               sum(coalesce(b.poi_count, 0)) AS block_poi_count
        ORDER BY area_id
        """,
        {"area_ids": area_ids},
    )
    return {row["area_id"]: row for row in rows}


def fetch_od_flows(kg: Neo4jClient, ward_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    rows = kg.query(
        """
        MATCH (origin:Ward)-[flow:hasODFlow]->(dest:Ward)
        WHERE origin.id IN $ward_ids
        RETURN origin.id AS origin_ward_id,
               origin.name AS origin_ward_name,
               dest.id AS dest_ward_id,
               dest.name AS dest_ward_name,
               coalesce(flow.total_trips, flow.S05b_035, 0) AS total_trips,
               coalesce(flow.rail_total, flow.S05b_010, 0) AS rail_total,
               coalesce(flow.bus_total, flow.S05b_016, 0) AS bus_total,
               coalesce(flow.car_total, flow.S05b_022, 0) AS car_total,
               coalesce(flow.walk_total, flow.S05b_034, 0) AS walk_total
        ORDER BY origin_ward_id, total_trips DESC, dest_ward_id
        """,
        {"ward_ids": ward_ids},
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["origin_ward_id"]].append(row)
    return dict(grouped)


def fetch_anchors(kg: Neo4jClient) -> list[dict[str, Any]]:
    return kg.query(
        """
        MATCH (p:POI)
        WHERE p.source IN ["station", "school"]
          AND p.lon IS NOT NULL
          AND p.lat IS NOT NULL
        RETURN p.id AS poi_id,
               p.name AS name,
               p.source AS source,
               p.lon AS lon,
               p.lat AS lat
        ORDER BY p.source, p.id
        """
    )


def group_by_area(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["area_id"]].append(row)
    return dict(grouped)


def compute_area_spatial(
    areas: list[AreaRecord],
    projector: GeometryProjector,
) -> dict[str, AreaSpatial]:
    spatial: dict[str, AreaSpatial] = {}
    ward_cache: dict[str, BaseGeometry] = {}

    for area in areas:
        geom = parse_wkt(area.geometry_wkt)
        if geom is None or geom.is_empty:
            continue

        geom_m = projector.project_geom(geom)
        centroid = geom.centroid
        centroid_m = geom_m.centroid
        area_m2 = float(geom_m.area)
        perimeter_m = float(geom_m.length)
        compactness = 0.0
        if perimeter_m > 0:
            compactness = float((4 * math.pi * area_m2) / (perimeter_m * perimeter_m))

        ward_dist = None
        ward_dir = None
        if area.ward_geometry_wkt:
            if area.ward_id not in ward_cache:
                ward_geom = parse_wkt(area.ward_geometry_wkt)
                if ward_geom is not None:
                    ward_cache[area.ward_id or area.area_id] = projector.project_geom(ward_geom)
            ward_geom_m = ward_cache.get(area.ward_id or area.area_id)
            if ward_geom_m is not None:
                ward_centroid_m = ward_geom_m.centroid
                dx = float(centroid_m.x - ward_centroid_m.x)
                dy = float(centroid_m.y - ward_centroid_m.y)
                ward_dist = math.hypot(dx, dy)
                ward_dir = direction_token(dx, dy)

        spatial[area.area_id] = AreaSpatial(
            geom=geom,
            geom_m=geom_m,
            centroid=centroid,
            centroid_m=centroid_m,
            area_m2=area_m2,
            area_km2=area_m2 / 1_000_000,
            perimeter_m=perimeter_m,
            compactness=compactness,
            grid_token=grid_token(float(centroid.x), float(centroid.y)),
            ward_center_dist_m=ward_dist,
            ward_relative_dir=ward_dir,
        )

    return spatial


def aggregate_poi_cates(
    poi_rows: list[dict[str, Any]],
    area_spatial: dict[str, AreaSpatial],
    projector: GeometryProjector,
) -> dict[str, list[dict[str, Any]]]:
    valid_rows = [
        row
        for row in poi_rows
        if row.get("area_id") in area_spatial and row.get("lon") is not None and row.get("lat") is not None
    ]
    lons = [float(row["lon"]) for row in valid_rows]
    lats = [float(row["lat"]) for row in valid_rows]
    xs, ys = projector.project_points(lons, lats) if valid_rows else ([], [])

    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    for row, x, y in zip(valid_rows, xs, ys):
        area_id = row["area_id"]
        key = (area_id, str(row.get("cate_name") or row.get("cate_id") or "UNKNOWN"))
        area = area_spatial[area_id]
        distance_m = math.hypot(float(x - area.centroid_m.x), float(y - area.centroid_m.y))

        if key not in aggregates:
            aggregates[key] = {
                "area_id": area_id,
                "cate_id": row.get("cate_id"),
                "cate_name": row.get("cate_name"),
                "poi_ids": set(),
                "distance_sum_m": 0.0,
            }
        aggregates[key]["poi_ids"].add(row.get("poi_id"))
        aggregates[key]["distance_sum_m"] += distance_m

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in aggregates.values():
        poi_count = len(item["poi_ids"])
        avg_dist_m = item["distance_sum_m"] / poi_count if poi_count else None
        area = area_spatial[item["area_id"]]
        grouped[item["area_id"]].append(
            {
                "area_id": item["area_id"],
                "cate_id": item["cate_id"],
                "cate_name": item["cate_name"],
                "poi_count": poi_count,
                "poi_density": density(poi_count, area.area_km2),
                "avg_dist_m": avg_dist_m,
            }
        )

    for rows in grouped.values():
        rows.sort(key=lambda row: (-row["poi_count"], str(row.get("cate_name"))))
    return dict(grouped)


def compute_neighbors(
    raw_neighbors: dict[str, list[dict[str, Any]]],
    area_spatial: dict[str, AreaSpatial],
    projector: GeometryProjector,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    neighbor_geom_cache: dict[str, BaseGeometry] = {}

    for area_id, rows in raw_neighbors.items():
        area = area_spatial.get(area_id)
        if area is None:
            continue

        for row in rows:
            neighbor_id = row.get("neighbor_id")
            if not neighbor_id:
                continue
            if neighbor_id not in neighbor_geom_cache:
                geom = parse_wkt(row.get("neighbor_geometry_wkt"))
                if geom is not None:
                    neighbor_geom_cache[neighbor_id] = projector.project_geom(geom)

            neighbor_geom_m = neighbor_geom_cache.get(neighbor_id)
            if neighbor_geom_m is None:
                continue

            neighbor_centroid = neighbor_geom_m.centroid
            dx = float(neighbor_centroid.x - area.centroid_m.x)
            dy = float(neighbor_centroid.y - area.centroid_m.y)
            centroid_dist_m = math.hypot(dx, dy)
            border_len_m = 0.0
            try:
                border_len_m = float(area.geom_m.boundary.intersection(neighbor_geom_m.boundary).length)
            except Exception:
                border_len_m = 0.0

            grouped[area_id].append(
                {
                    "area_id": area_id,
                    "neighbor_id": neighbor_id,
                    "neighbor_name": row.get("neighbor_name"),
                    "direction": direction_token(dx, dy),
                    "centroid_dist_m": centroid_dist_m,
                    "border_len_m": border_len_m,
                }
            )

    for rows in grouped.values():
        rows.sort(key=lambda row: (-row["border_len_m"], row["centroid_dist_m"], row["neighbor_id"]))
    return dict(grouped)


def prepare_anchors(raw_rows: list[dict[str, Any]], projector: GeometryProjector) -> dict[str, list[Anchor]]:
    rows = [row for row in raw_rows if row.get("lon") is not None and row.get("lat") is not None]
    xs, ys = projector.project_points([float(row["lon"]) for row in rows], [float(row["lat"]) for row in rows])
    grouped: dict[str, list[Anchor]] = defaultdict(list)
    for row, x, y in zip(rows, xs, ys):
        grouped[str(row["source"])].append(
            Anchor(
                poi_id=row["poi_id"],
                name=row.get("name"),
                source=str(row["source"]),
                x=float(x),
                y=float(y),
            )
        )
    return dict(grouped)


def nearest_anchor(area: AreaSpatial, anchors: list[Anchor]) -> dict[str, Any] | None:
    if not anchors:
        return None
    best = None
    best_dist = float("inf")
    for anchor in anchors:
        dx = anchor.x - float(area.centroid_m.x)
        dy = anchor.y - float(area.centroid_m.y)
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best = (anchor, dx, dy)
    if best is None:
        return None
    anchor, dx, dy = best
    return {
        "poi_id": anchor.poi_id,
        "name": anchor.name,
        "distance_m": math.sqrt(best_dist),
        "direction": direction_token(dx, dy),
    }


def make_binners(
    areas: list[AreaRecord],
    area_spatial: dict[str, AreaSpatial],
    poi_cates: dict[str, list[dict[str, Any]]],
    road_types: dict[str, list[dict[str, Any]]],
    neighbors: dict[str, list[dict[str, Any]]],
    blocks: dict[str, dict[str, Any]],
    od_flows: dict[str, list[dict[str, Any]]],
) -> dict[str, QuantileBinner]:
    area_ids = [area.area_id for area in areas]
    block_rows = [blocks.get(area_id, {}) for area_id in area_ids]
    poi_rows = [row for rows in poi_cates.values() for row in rows]
    road_rows = [row for rows in road_types.values() for row in rows]
    neighbor_rows = [row for rows in neighbors.values() for row in rows]
    od_rows = [row for rows in od_flows.values() for row in rows]

    road_density_values = []
    for row in road_rows:
        area = area_spatial.get(row["area_id"])
        road_density_values.append(density(row.get("road_length"), area.area_km2 if area else 0.0))

    return {
        "area_m2": QuantileBinner(area.area_m2 for area in area_spatial.values()),
        "compactness": QuantileBinner(area.compactness for area in area_spatial.values()),
        "block_count": QuantileBinner(row.get("block_count", 0) for row in block_rows),
        "block_poi_count": QuantileBinner(row.get("block_poi_count", 0) for row in block_rows),
        "poi_count": QuantileBinner(row.get("poi_count") for row in poi_rows),
        "poi_density": QuantileBinner(row.get("poi_density") for row in poi_rows),
        "road_count": QuantileBinner(row.get("road_count") for row in road_rows),
        "road_length": QuantileBinner(row.get("road_length") for row in road_rows),
        "road_density": QuantileBinner(road_density_values),
        "border_len": QuantileBinner(row.get("border_len_m") for row in neighbor_rows),
        "total_trips": QuantileBinner(row.get("total_trips") for row in od_rows),
        "rail_total": QuantileBinner(row.get("rail_total") for row in od_rows),
        "bus_total": QuantileBinner(row.get("bus_total") for row in od_rows),
        "car_total": QuantileBinner(row.get("car_total") for row in od_rows),
        "walk_total": QuantileBinner(row.get("walk_total") for row in od_rows),
    }


def build_structure(
    area: AreaRecord,
    area_spatial: dict[str, AreaSpatial],
    poi_cates: dict[str, list[dict[str, Any]]],
    road_types: dict[str, list[dict[str, Any]]],
    neighbors: dict[str, list[dict[str, Any]]],
    blocks: dict[str, dict[str, Any]],
    od_flows: dict[str, list[dict[str, Any]]],
    anchors: dict[str, list[Anchor]],
    binners: dict[str, QuantileBinner],
    top_poi_cates: int,
    top_road_types: int,
    top_neighbors: int,
    top_od_flows: int,
) -> dict[str, list[list[str]]]:
    area_tok = id_token("AREA", area.area_id)
    spatial = area_spatial.get(area.area_id)
    structure: dict[str, list[list[str]]] = {
        "center": [["CENTER", area_tok]],
        "hierarchy": [],
        "local_inventory": [],
        "spatial_neighbors": [],
        "mobility_context": [],
    }

    if spatial is not None:
        structure["center"].append(
            [
                "SPATIAL",
                area_tok,
                attr_token("AREA_SIZE", spatial.area_m2, binners["area_m2"]),
                attr_token("COMPACTNESS", spatial.compactness, binners["compactness"]),
                spatial.grid_token,
            ]
        )
        if spatial.ward_center_dist_m is not None and spatial.ward_relative_dir is not None:
            structure["center"].append(
                [
                    "WARD_REL_POS",
                    area_tok,
                    spatial.ward_relative_dir,
                    distance_bucket(spatial.ward_center_dist_m),
                ]
            )

        for source, token_name in (("station", "NEAREST_STATION"), ("school", "NEAREST_SCHOOL")):
            nearest = nearest_anchor(spatial, anchors.get(source, []))
            if nearest is not None:
                structure["center"].append(
                    [
                        "ANCHOR",
                        area_tok,
                        token_name,
                        id_token("POI", nearest["poi_id"]),
                        distance_bucket(nearest["distance_m"]),
                        nearest["direction"],
                    ]
                )

    if area.ward_id:
        structure["hierarchy"].append(["HIER", area_tok, "REL_BELONGS_TO", id_token("WARD", area.ward_id)])

    block_row = blocks.get(area.area_id, {})
    structure["local_inventory"].append(
        [
            "BLOCK",
            area_tok,
            attr_token("BLOCK_COUNT", block_row.get("block_count", 0), binners["block_count"]),
            attr_token("BLOCK_POI_COUNT", block_row.get("block_poi_count", 0), binners["block_poi_count"]),
        ]
    )

    for row in poi_cates.get(area.area_id, [])[:top_poi_cates]:
        structure["local_inventory"].append(
            [
                "POI_CATE",
                area_tok,
                id_token("CATE", row.get("cate_name"), row.get("cate_id")),
                attr_token("COUNT", row.get("poi_count"), binners["poi_count"]),
                attr_token("DENSITY", row.get("poi_density"), binners["poi_density"]),
                distance_bucket(row.get("avg_dist_m")),
            ]
        )

    for row in road_types.get(area.area_id, [])[:top_road_types]:
        spatial_area = area_spatial.get(area.area_id)
        road_density = density(row.get("road_length"), spatial_area.area_km2 if spatial_area else 0.0)
        structure["local_inventory"].append(
            [
                "ROAD_TYPE",
                area_tok,
                id_token("ROADTYPE", row.get("road_type")),
                attr_token("COUNT", row.get("road_count"), binners["road_count"]),
                attr_token("LENGTH", row.get("road_length"), binners["road_length"]),
                attr_token("DENSITY", road_density, binners["road_density"]),
            ]
        )

    for row in neighbors.get(area.area_id, [])[:top_neighbors]:
        structure["spatial_neighbors"].append(
            [
                "NEIGHBOR_HOP_1",
                area_tok,
                "REL_BORDER_BY",
                id_token("AREA", row.get("neighbor_id")),
                row.get("direction", "DIR_UNKNOWN"),
                distance_bucket(row.get("centroid_dist_m")),
                attr_token("BORDER_LEN", row.get("border_len_m"), binners["border_len"]),
            ]
        )

    if area.ward_id:
        origin_tok = id_token("WARD", area.ward_id)
        for row in od_flows.get(area.ward_id, [])[:top_od_flows]:
            structure["mobility_context"].append(
                [
                    "OD",
                    origin_tok,
                    "REL_HAS_OD_FLOW",
                    id_token("WARD", row.get("dest_ward_id")),
                    attr_token("TOTAL_TRIPS", row.get("total_trips"), binners["total_trips"]),
                    attr_token("RAIL", row.get("rail_total"), binners["rail_total"]),
                    attr_token("BUS", row.get("bus_total"), binners["bus_total"]),
                    attr_token("CAR", row.get("car_total"), binners["car_total"]),
                    attr_token("WALK", row.get("walk_total"), binners["walk_total"]),
                ]
            )

    return structure


def linearize_structure(structure: dict[str, list[list[str]]]) -> list[list[str]]:
    section_order = [
        ("center", "<CENTER>"),
        ("hierarchy", "<HIER>"),
        ("local_inventory", "<LOCAL>"),
        ("spatial_neighbors", "<NEIGHBOR>"),
        ("mobility_context", "<OD>"),
    ]
    tokens: list[list[str]] = [["<URBAN>"]]
    for section, marker in section_order:
        section_tokens = structure.get(section, [])
        if not section_tokens:
            continue
        tokens.append([marker])
        tokens.extend(section_tokens)
    tokens.append(["</URBAN>"])
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Area-centered spatial discrete UrbanToken V2 JSONL.")
    parser.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env", help="Path to .env file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of Area centers.")
    parser.add_argument("--top-poi-cates", type=int, default=5, help="Top POI categories per Area.")
    parser.add_argument("--top-road-types", type=int, default=5, help="Top road types per Area.")
    parser.add_argument("--top-neighbors", type=int, default=5, help="Top neighbor Areas per Area.")
    parser.add_argument("--top-od-flows", type=int, default=5, help="Top OD destination Wards per origin Ward.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kg = get_kg(args.env)
    projector = GeometryProjector()

    try:
        areas = fetch_areas(kg, args.limit)
        area_ids = [area.area_id for area in areas]
        ward_ids = sorted({area.ward_id for area in areas if area.ward_id})

        area_spatial = compute_area_spatial(areas, projector)
        poi_cates = aggregate_poi_cates(fetch_poi_points(kg, area_ids), area_spatial, projector)
        road_types = fetch_road_types(kg, area_ids)
        neighbors = compute_neighbors(fetch_neighbors(kg, area_ids), area_spatial, projector)
        blocks = fetch_blocks(kg, area_ids)
        od_flows = fetch_od_flows(kg, ward_ids)
        anchors = prepare_anchors(fetch_anchors(kg), projector)
        binners = make_binners(areas, area_spatial, poi_cates, road_types, neighbors, blocks, od_flows)

        def records() -> Iterable[dict[str, Any]]:
            for area in areas:
                structure = build_structure(
                    area=area,
                    area_spatial=area_spatial,
                    poi_cates=poi_cates,
                    road_types=road_types,
                    neighbors=neighbors,
                    blocks=blocks,
                    od_flows=od_flows,
                    anchors=anchors,
                    binners=binners,
                    top_poi_cates=args.top_poi_cates,
                    top_road_types=args.top_road_types,
                    top_neighbors=args.top_neighbors,
                    top_od_flows=args.top_od_flows,
                )
                tokens = linearize_structure(structure)
                yield {
                    "tokenizer": "spatial_discrete_v2",
                    "center_type": "Area",
                    "center": {
                        "id": area.area_id,
                        "name": area.area_name,
                        "town_code": area.town_code,
                        "ward_id": area.ward_id,
                        "ward_name": area.ward_name,
                    },
                    "crs": {
                        "source": SOURCE_CRS,
                        "metric": METRIC_CRS,
                    },
                    "structure": structure,
                    "tokens": tokens,
                    "token_count": len(tokens),
                }

        count = write_jsonl(records(), args.output)
        print(f"Wrote {count} Area-centered spatial UrbanToken V2 records to {args.output}")
    finally:
        kg.close()


if __name__ == "__main__":
    main()
