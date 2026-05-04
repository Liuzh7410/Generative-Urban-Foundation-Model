"""
Extract discrete UrbanToken V1 sequences from Neo4j UrbanKG.

The first implementation is Area-centered and converts a local KG view into
interpretable list tokens such as:

    ["HIER", "AREA_131010001", "REL_BELONGS_TO", "WARD_13101"]
    ["POI", "AREA_131010001", "REL_HAS_POI_CATE", "CATE_MEAL", "COUNT_HIGH"]

Output is JSONL so downstream model builders can stream it directly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "urban_tokens_area_centered_v1.jsonl"


@dataclass(frozen=True)
class AreaRecord:
    area_id: str
    area_name: str | None
    town_code: str | None
    ward_id: str | None
    ward_name: str | None
    ward_code: str | None


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


class QuantileBinner:
    def __init__(self, values: Iterable[float | int | None]):
        cleaned = sorted(float(value) for value in values if value is not None)
        self.q33 = self._percentile(cleaned, 1 / 3)
        self.q66 = self._percentile(cleaned, 2 / 3)

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        index = round((len(values) - 1) * q)
        return values[index]

    def label(self, value: float | int | None) -> str:
        numeric = 0.0 if value is None else float(value)
        if numeric <= self.q33:
            return "LOW"
        if numeric <= self.q66:
            return "MED"
        return "HIGH"


def get_kg(env_path: Path) -> Neo4jClient:
    load_dotenv(env_path, override=True)

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
        raise RuntimeError(f"Missing Neo4j config in {env_path}: {', '.join(missing)}")

    return Neo4jClient(uri=uri, username=username, password=password, database=database)


def token_fragment(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().upper()
    text = re.sub(r"[^0-9A-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "UNKNOWN"


def id_token(label: str, raw_id: Any, fallback: Any = None) -> str:
    value = raw_id if raw_id not in (None, "") else fallback
    text = "" if value is None else str(value)
    if ":" in text:
        text = text.split(":", 1)[1]
    return f"{label.upper()}_{token_fragment(text)}"


def attr_token(name: str, value: float | int | None, binner: QuantileBinner) -> str:
    return f"{token_fragment(name)}_{binner.label(value)}"


def fetch_areas(kg: Neo4jClient, limit: int | None) -> list[AreaRecord]:
    cypher = """
    MATCH (a:Area)
    OPTIONAL MATCH (a)-[:belongsTo]->(w:Ward)
    OPTIONAL MATCH (name_w:Ward {name: a.ward_name})
    RETURN a.id AS area_id,
           a.name AS area_name,
           a.town_code AS town_code,
           coalesce(w.id, name_w.id) AS ward_id,
           coalesce(w.name, name_w.name, a.ward_name) AS ward_name,
           coalesce(w.ward_code, name_w.ward_code) AS ward_code
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
        )
        for row in rows
    ]


def fetch_poi_cates(kg: Neo4jClient, area_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    rows = kg.query(
        """
        MATCH (p:POI)-[:locatesAt]->(a:Area)
        WHERE a.id IN $area_ids
        MATCH (p)-[:hasType]->(c:Cate)
        RETURN a.id AS area_id,
               c.id AS cate_id,
               c.name AS cate_name,
               count(DISTINCT p) AS poi_count
        ORDER BY area_id, poi_count DESC, cate_name
        """,
        {"area_ids": area_ids},
    )
    return group_by_area(rows)


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
               n.name AS neighbor_name
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


def group_by_area(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["area_id"]].append(row)
    return dict(grouped)


def make_binners(
    areas: list[AreaRecord],
    poi_cates: dict[str, list[dict[str, Any]]],
    road_types: dict[str, list[dict[str, Any]]],
    blocks: dict[str, dict[str, Any]],
    od_flows: dict[str, list[dict[str, Any]]],
) -> dict[str, QuantileBinner]:
    area_ids = [area.area_id for area in areas]
    block_rows = [blocks.get(area_id, {}) for area_id in area_ids]
    od_rows = [row for rows in od_flows.values() for row in rows]
    road_rows = [row for rows in road_types.values() for row in rows]
    poi_rows = [row for rows in poi_cates.values() for row in rows]

    return {
        "poi_count": QuantileBinner(row.get("poi_count") for row in poi_rows),
        "road_count": QuantileBinner(row.get("road_count") for row in road_rows),
        "road_length": QuantileBinner(row.get("road_length") for row in road_rows),
        "block_count": QuantileBinner(row.get("block_count", 0) for row in block_rows),
        "block_poi_count": QuantileBinner(row.get("block_poi_count", 0) for row in block_rows),
        "total_trips": QuantileBinner(row.get("total_trips") for row in od_rows),
        "rail_total": QuantileBinner(row.get("rail_total") for row in od_rows),
        "bus_total": QuantileBinner(row.get("bus_total") for row in od_rows),
        "car_total": QuantileBinner(row.get("car_total") for row in od_rows),
        "walk_total": QuantileBinner(row.get("walk_total") for row in od_rows),
    }


def build_area_tokens(
    area: AreaRecord,
    poi_cates: dict[str, list[dict[str, Any]]],
    road_types: dict[str, list[dict[str, Any]]],
    neighbors: dict[str, list[dict[str, Any]]],
    blocks: dict[str, dict[str, Any]],
    od_flows: dict[str, list[dict[str, Any]]],
    binners: dict[str, QuantileBinner],
    top_poi_cates: int,
    top_road_types: int,
    top_neighbors: int,
    top_od_flows: int,
) -> list[list[str]]:
    area_tok = id_token("AREA", area.area_id)
    tokens: list[list[str]] = [["CENTER", area_tok]]

    if area.ward_id:
        tokens.append(["HIER", area_tok, "REL_BELONGS_TO", id_token("WARD", area.ward_id)])

    block_row = blocks.get(area.area_id, {})
    tokens.append(
        [
            "ATTR",
            area_tok,
            attr_token("BLOCK_COUNT", block_row.get("block_count", 0), binners["block_count"]),
            attr_token("BLOCK_POI_COUNT", block_row.get("block_poi_count", 0), binners["block_poi_count"]),
        ]
    )

    for row in poi_cates.get(area.area_id, [])[:top_poi_cates]:
        tokens.append(
            [
                "POI",
                area_tok,
                "REL_HAS_POI_CATE",
                id_token("CATE", row.get("cate_name"), row.get("cate_id")),
                attr_token("COUNT", row.get("poi_count"), binners["poi_count"]),
            ]
        )

    for row in road_types.get(area.area_id, [])[:top_road_types]:
        tokens.append(
            [
                "ROAD",
                area_tok,
                "REL_HAS_ROAD_TYPE",
                id_token("ROADTYPE", row.get("road_type")),
                attr_token("COUNT", row.get("road_count"), binners["road_count"]),
                attr_token("LENGTH", row.get("road_length"), binners["road_length"]),
            ]
        )

    for row in neighbors.get(area.area_id, [])[:top_neighbors]:
        tokens.append(
            [
                "NEIGHBOR",
                area_tok,
                "REL_BORDER_BY",
                id_token("AREA", row.get("neighbor_id")),
            ]
        )

    if area.ward_id:
        origin_tok = id_token("WARD", area.ward_id)
        for row in od_flows.get(area.ward_id, [])[:top_od_flows]:
            tokens.append(
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

    return tokens


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Area-centered discrete UrbanToken V1 JSONL.")
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

    try:
        areas = fetch_areas(kg, args.limit)
        area_ids = [area.area_id for area in areas]
        ward_ids = sorted({area.ward_id for area in areas if area.ward_id})

        poi_cates = fetch_poi_cates(kg, area_ids)
        road_types = fetch_road_types(kg, area_ids)
        neighbors = fetch_neighbors(kg, area_ids)
        blocks = fetch_blocks(kg, area_ids)
        od_flows = fetch_od_flows(kg, ward_ids)
        binners = make_binners(areas, poi_cates, road_types, blocks, od_flows)

        def records() -> Iterable[dict[str, Any]]:
            for area in areas:
                tokens = build_area_tokens(
                    area=area,
                    poi_cates=poi_cates,
                    road_types=road_types,
                    neighbors=neighbors,
                    blocks=blocks,
                    od_flows=od_flows,
                    binners=binners,
                    top_poi_cates=args.top_poi_cates,
                    top_road_types=args.top_road_types,
                    top_neighbors=args.top_neighbors,
                    top_od_flows=args.top_od_flows,
                )
                yield {
                    "tokenizer": "discrete_v1",
                    "center_type": "Area",
                    "center": {
                        "id": area.area_id,
                        "name": area.area_name,
                        "town_code": area.town_code,
                        "ward_id": area.ward_id,
                        "ward_name": area.ward_name,
                    },
                    "tokens": tokens,
                    "token_count": len(tokens),
                }

        count = write_jsonl(records(), args.output)
        print(f"Wrote {count} Area-centered UrbanToken records to {args.output}")
    finally:
        kg.close()


if __name__ == "__main__":
    main()
