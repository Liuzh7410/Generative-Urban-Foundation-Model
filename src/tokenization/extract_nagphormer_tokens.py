"""
Extract continuous Hop2Token data for NAGphormer from Neo4j UrbanKG.

This script builds:

    X: Area feature matrix
    A_hat: normalized Area adjacency from borderBy
    hop_tokens: [N, hops + 1, feature_dim], where X_k = A_hat^k X
    labels: dominant POI category per Area

The output is designed for src/trainers/train_nagphormer.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyproj import Transformer
from shapely import wkt
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "token" / "nagphormer"
SOURCE_CRS = "EPSG:4326"
METRIC_CRS = "EPSG:32654"


@dataclass(frozen=True)
class AreaRow:
    id: str
    name: str | None
    town_code: str | None
    ward_id: str | None
    ward_name: str | None
    geometry: str | None


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


def parse_wkt(value: str | None) -> BaseGeometry | None:
    if not value:
        return None
    try:
        return wkt.loads(value)
    except Exception:
        return None


def project_geom(geom: BaseGeometry) -> BaseGeometry:
    transformer = Transformer.from_crs(SOURCE_CRS, METRIC_CRS, always_xy=True)
    return shapely_transform(transformer.transform, geom)


def fetch_areas(kg: Neo4jClient, limit: int | None) -> list[AreaRow]:
    cypher = """
    MATCH (a:Area)
    OPTIONAL MATCH (a)-[:belongsTo]->(w:Ward)
    OPTIONAL MATCH (name_w:Ward {name: a.ward_name})
    WITH a, coalesce(w, name_w) AS ward
    RETURN a.id AS id,
           a.name AS name,
           a.town_code AS town_code,
           ward.id AS ward_id,
           coalesce(ward.name, a.ward_name) AS ward_name,
           a.geometry AS geometry
    ORDER BY a.id
    """
    if limit is not None:
        cypher += "\nLIMIT $limit"
    rows = kg.query(cypher, {"limit": limit} if limit is not None else {})
    return [
        AreaRow(
            id=row["id"],
            name=row.get("name"),
            town_code=row.get("town_code"),
            ward_id=row.get("ward_id"),
            ward_name=row.get("ward_name"),
            geometry=row.get("geometry"),
        )
        for row in rows
    ]


def fetch_border_edges(kg: Neo4jClient, area_ids: list[str]) -> list[tuple[str, str]]:
    rows = kg.query(
        """
        MATCH (a:Area)-[:borderBy]->(b:Area)
        WHERE a.id IN $area_ids AND b.id IN $area_ids
        RETURN a.id AS src, b.id AS dst
        """,
        {"area_ids": area_ids},
    )
    return [(row["src"], row["dst"]) for row in rows]


def fetch_poi_category_counts(kg: Neo4jClient, area_ids: list[str]) -> dict[str, Counter[str]]:
    rows = kg.query(
        """
        MATCH (p:POI)-[:locatesAt]->(a:Area)
        WHERE a.id IN $area_ids
        MATCH (p)-[:hasType]->(c:Cate)
        RETURN a.id AS area_id, c.name AS cate_name, count(DISTINCT p) AS count
        """,
        {"area_ids": area_ids},
    )
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[row["area_id"]][str(row.get("cate_name") or "UNKNOWN")] += int(row["count"])
    return counts


def fetch_block_stats(kg: Neo4jClient, area_ids: list[str]) -> dict[str, dict[str, float]]:
    rows = kg.query(
        """
        MATCH (b:Block)-[:belongsTo]->(a:Area)
        WHERE a.id IN $area_ids
        RETURN a.id AS area_id,
               count(DISTINCT b) AS block_count,
               sum(coalesce(b.poi_count, 0)) AS block_poi_count
        """,
        {"area_ids": area_ids},
    )
    return {
        row["area_id"]: {
            "block_count": float(row.get("block_count") or 0.0),
            "block_poi_count": float(row.get("block_poi_count") or 0.0),
        }
        for row in rows
    }


def fetch_road_stats(kg: Neo4jClient, area_ids: list[str]) -> tuple[dict[str, dict[str, float]], list[str]]:
    rows = kg.query(
        """
        MATCH (r:Road)-[:in]->(a:Area)
        WHERE a.id IN $area_ids
        RETURN a.id AS area_id,
               coalesce(toString(r.rtype), "UNKNOWN") AS road_type,
               count(DISTINCT r) AS road_count,
               sum(coalesce(r.length, 0.0)) AS road_length
        """,
        {"area_ids": area_ids},
    )
    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    road_types = set()
    for row in rows:
        area_id = row["area_id"]
        road_type = str(row.get("road_type") or "UNKNOWN")
        road_types.add(road_type)
        count = float(row.get("road_count") or 0.0)
        length = float(row.get("road_length") or 0.0)
        stats[area_id]["road_count_total"] += count
        stats[area_id]["road_length_total"] += length
        stats[area_id][f"road_count_{road_type}"] += count
        stats[area_id][f"road_length_{road_type}"] += length
    return {key: dict(value) for key, value in stats.items()}, sorted(road_types)


def fetch_ward_od_stats(kg: Neo4jClient, ward_ids: list[str]) -> dict[str, dict[str, float]]:
    rows = kg.query(
        """
        MATCH (w:Ward)-[flow:hasODFlow]->(:Ward)
        WHERE w.id IN $ward_ids
        RETURN w.id AS ward_id,
               sum(coalesce(flow.total_trips, flow.S05b_035, 0)) AS total_trips,
               sum(coalesce(flow.rail_total, flow.S05b_010, 0)) AS rail_total,
               sum(coalesce(flow.bus_total, flow.S05b_016, 0)) AS bus_total,
               sum(coalesce(flow.car_total, flow.S05b_022, 0)) AS car_total,
               sum(coalesce(flow.walk_total, flow.S05b_034, 0)) AS walk_total
        """,
        {"ward_ids": ward_ids},
    )
    return {
        row["ward_id"]: {
            "od_total_trips": float(row.get("total_trips") or 0.0),
            "od_rail_total": float(row.get("rail_total") or 0.0),
            "od_bus_total": float(row.get("bus_total") or 0.0),
            "od_car_total": float(row.get("car_total") or 0.0),
            "od_walk_total": float(row.get("walk_total") or 0.0),
        }
        for row in rows
    }


def compute_spatial_stats(areas: list[AreaRow]) -> dict[str, dict[str, float]]:
    stats = {}
    for area in areas:
        geom = parse_wkt(area.geometry)
        if geom is None or geom.is_empty:
            stats[area.id] = {"area_m2": 0.0, "perimeter_m": 0.0, "compactness": 0.0}
            continue
        geom_m = project_geom(geom)
        area_m2 = float(geom_m.area)
        perimeter_m = float(geom_m.length)
        compactness = 0.0
        if perimeter_m > 0:
            compactness = float((4 * math.pi * area_m2) / (perimeter_m * perimeter_m))
        stats[area.id] = {
            "area_m2": area_m2,
            "perimeter_m": perimeter_m,
            "compactness": compactness,
        }
    return stats


def build_feature_matrix(
    areas: list[AreaRow],
    poi_counts: dict[str, Counter[str]],
    block_stats: dict[str, dict[str, float]],
    road_stats: dict[str, dict[str, float]],
    road_types: list[str],
    od_stats: dict[str, dict[str, float]],
    spatial_stats: dict[str, dict[str, float]],
    min_label_count: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int], list[str]]:
    categories = sorted({cate for counts in poi_counts.values() for cate in counts.keys()})
    labels = []
    label_names = []
    feature_names = [
        "area_m2",
        "perimeter_m",
        "compactness",
        "block_count",
        "block_poi_count",
        "poi_total",
    ]
    feature_names.extend([f"poi_count_{cate}" for cate in categories])
    feature_names.extend(["road_count_total", "road_length_total"])
    for road_type in road_types:
        feature_names.append(f"road_count_{road_type}")
        feature_names.append(f"road_length_{road_type}")
    feature_names.extend(
        [
            "od_total_trips",
            "od_rail_total",
            "od_bus_total",
            "od_car_total",
            "od_walk_total",
        ]
    )

    raw_label_names = []
    for area in areas:
        counts = poi_counts.get(area.id, Counter())
        raw_label_names.append(counts.most_common(1)[0][0] if counts else "__NO_LABEL__")

    label_frequency = Counter(raw_label_names)
    kept_labels = sorted(
        label for label, count in label_frequency.items() if label != "__NO_LABEL__" and count >= min_label_count
    )
    label_to_id = {label: idx for idx, label in enumerate(kept_labels)}

    features = np.zeros((len(areas), len(feature_names)), dtype=np.float32)
    y = np.full((len(areas),), -1, dtype=np.int64)

    for idx, area in enumerate(areas):
        values = []
        spatial = spatial_stats.get(area.id, {})
        blocks = block_stats.get(area.id, {})
        roads = road_stats.get(area.id, {})
        od = od_stats.get(area.ward_id or "", {})
        counts = poi_counts.get(area.id, Counter())
        poi_total = float(sum(counts.values()))

        values.extend(
            [
                spatial.get("area_m2", 0.0),
                spatial.get("perimeter_m", 0.0),
                spatial.get("compactness", 0.0),
                blocks.get("block_count", 0.0),
                blocks.get("block_poi_count", 0.0),
                poi_total,
            ]
        )
        values.extend([float(counts.get(cate, 0.0)) for cate in categories])
        values.extend([roads.get("road_count_total", 0.0), roads.get("road_length_total", 0.0)])
        for road_type in road_types:
            values.append(roads.get(f"road_count_{road_type}", 0.0))
            values.append(roads.get(f"road_length_{road_type}", 0.0))
        values.extend(
            [
                od.get("od_total_trips", 0.0),
                od.get("od_rail_total", 0.0),
                od.get("od_bus_total", 0.0),
                od.get("od_car_total", 0.0),
                od.get("od_walk_total", 0.0),
            ]
        )
        features[idx] = np.asarray(values, dtype=np.float32)

        label_name = raw_label_names[idx]
        if label_name in label_to_id:
            y[idx] = label_to_id[label_name]
        label_names.append(label_name)

    return features, y, feature_names, label_to_id, label_names


def normalize_features(features: np.ndarray) -> np.ndarray:
    log_features = np.log1p(np.maximum(features, 0.0))
    mean = log_features.mean(axis=0, keepdims=True)
    std = log_features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((log_features - mean) / std).astype(np.float32)


def build_normalized_adjacency(num_nodes: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for src, dst in edges:
        adjacency[src, dst] = 1.0
        adjacency[dst, src] = 1.0
    adjacency.fill_diagonal_(1.0)
    degree = adjacency.sum(dim=1)
    inv_sqrt_degree = torch.pow(degree.clamp(min=1.0), -0.5)
    return inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]


def hop2token(features: torch.Tensor, adjacency: torch.Tensor, hops: int) -> torch.Tensor:
    tokens = [features]
    current = features
    for _ in range(hops):
        current = adjacency @ current
        tokens.append(current)
    return torch.stack(tokens, dim=1)


def make_splits(labels: np.ndarray, seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[int]]:
    rng = random.Random(seed)
    train, val, test = [], [], []
    for label in sorted(set(int(v) for v in labels if v >= 0)):
        indices = [idx for idx, value in enumerate(labels) if int(value) == label]
        rng.shuffle(indices)
        n_train = max(1, int(len(indices) * train_ratio))
        n_val = max(1, int(len(indices) * val_ratio)) if len(indices) - n_train > 1 else 0
        train.extend(indices[:n_train])
        val.extend(indices[n_train : n_train + n_val])
        test.extend(indices[n_train + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract NAGphormer Hop2Token tensors from Neo4j UrbanKG.")
    parser.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hops", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-label-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kg = get_kg(args.env)
    try:
        areas = fetch_areas(kg, args.limit)
        area_ids = [area.id for area in areas]
        area_to_idx = {area_id: idx for idx, area_id in enumerate(area_ids)}
        ward_ids = sorted({area.ward_id for area in areas if area.ward_id})

        raw_edges = fetch_border_edges(kg, area_ids)
        edges = [
            (area_to_idx[src], area_to_idx[dst])
            for src, dst in raw_edges
            if src in area_to_idx and dst in area_to_idx
        ]
        poi_counts = fetch_poi_category_counts(kg, area_ids)
        block_stats = fetch_block_stats(kg, area_ids)
        road_stats, road_types = fetch_road_stats(kg, area_ids)
        od_stats = fetch_ward_od_stats(kg, ward_ids)
        spatial_stats = compute_spatial_stats(areas)

        features, labels, feature_names, label_to_id, raw_label_names = build_feature_matrix(
            areas=areas,
            poi_counts=poi_counts,
            block_stats=block_stats,
            road_stats=road_stats,
            road_types=road_types,
            od_stats=od_stats,
            spatial_stats=spatial_stats,
            min_label_count=args.min_label_count,
        )
        normalized_features = normalize_features(features)
        adjacency = build_normalized_adjacency(len(areas), edges)
        feature_tensor = torch.from_numpy(normalized_features)
        labels_tensor = torch.from_numpy(labels)
        hop_tokens = hop2token(feature_tensor, adjacency, args.hops)
        splits = make_splits(labels, args.seed, args.train_ratio, args.val_ratio)

        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hop_tokens": hop_tokens,
                "features": feature_tensor,
                "labels": labels_tensor,
                "adjacency": adjacency,
                "splits": {key: torch.tensor(value, dtype=torch.long) for key, value in splits.items()},
            },
            output_dir / "nagphormer_data.pt",
        )
        write_json(output_dir / "area_ids.json", area_ids)
        write_json(output_dir / "label_to_id.json", label_to_id)
        write_json(output_dir / "feature_names.json", feature_names)
        write_json(
            output_dir / "metadata.json",
            {
                "hops": args.hops,
                "num_nodes": len(areas),
                "num_edges": len(edges),
                "feature_dim": int(feature_tensor.shape[1]),
                "num_classes": len(label_to_id),
                "valid_label_nodes": int((labels >= 0).sum()),
                "split_sizes": {key: len(value) for key, value in splits.items()},
                "task": "dominant_poi_category_classification",
                "source_crs": SOURCE_CRS,
                "metric_crs": METRIC_CRS,
            },
        )
        write_json(
            output_dir / "area_labels.json",
            [
                {
                    "area_id": area.id,
                    "area_name": area.name,
                    "label": raw_label_names[idx],
                    "label_id": int(labels[idx]),
                }
                for idx, area in enumerate(areas)
            ],
        )

        print(f"Wrote NAGphormer data to {output_dir / 'nagphormer_data.pt'}")
        print(f"Nodes: {len(areas)}  Edges: {len(edges)}  Feature dim: {feature_tensor.shape[1]}")
        print(f"Classes: {len(label_to_id)}  Valid labels: {(labels >= 0).sum()}")
        print(f"Split sizes: { {key: len(value) for key, value in splits.items()} }")
    finally:
        kg.close()


if __name__ == "__main__":
    main()
