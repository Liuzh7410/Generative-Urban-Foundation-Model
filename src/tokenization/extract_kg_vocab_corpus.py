"""
Build a vocabulary-based KG token corpus from Neo4j UrbanKG.

This script is separate from the Area-centered memory tokenizers. It connects
to Neo4j, extracts Area-centered facts using the same spatial rules as V2, and
writes flat language-model-style token sequences plus vocabulary and ID files.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from extract_discrete_urban_tokens import attr_token, get_kg, id_token
from extract_spatial_discrete_urban_tokens_v2 import (
    METRIC_CRS,
    SOURCE_CRS,
    GeometryProjector,
    aggregate_poi_cates,
    compute_area_spatial,
    compute_neighbors,
    density,
    distance_bucket,
    fetch_anchors,
    fetch_areas,
    fetch_blocks,
    fetch_neighbors,
    fetch_od_flows,
    fetch_poi_points,
    fetch_road_types,
    make_binners,
    nearest_anchor,
    prepare_anchors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "token/vocab"

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]


def bin_token(label: str) -> str:
    return f"BIN_{label}"


def attr_bin(value: float | int | None, binner: Any) -> str:
    return bin_token(binner.label(value))


def poi_rank_token(index: int) -> str:
    if index == 1:
        return "RANK_PRIMARY"
    if index == 2:
        return "RANK_SECONDARY"
    if index == 3:
        return "RANK_TERTIARY"
    return f"RANK_{index}"


def road_rank_token(index: int) -> str:
    return poi_rank_token(index)


def sorted_poi_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -(row.get("poi_count") or 0),
            -(row.get("poi_density") or 0),
            row.get("avg_dist_m") if row.get("avg_dist_m") is not None else float("inf"),
            str(row.get("cate_name") or row.get("cate_id") or ""),
        ),
    )


def sorted_road_rows(rows: list[dict[str, Any]], spatial_area: Any | None) -> list[dict[str, Any]]:
    area_km2 = spatial_area.area_km2 if spatial_area else 0.0
    return sorted(
        rows,
        key=lambda row: (
            -(row.get("road_count") or 0),
            -(row.get("road_length") or 0),
            -density(row.get("road_length"), area_km2),
            str(row.get("road_type") or ""),
        ),
    )


def dominant_od_mode(row: dict[str, Any]) -> str:
    modes = {
        "RAIL": row.get("rail_total") or 0,
        "BUS": row.get("bus_total") or 0,
        "CAR": row.get("car_total") or 0,
        "WALK": row.get("walk_total") or 0,
    }
    mode, value = max(modes.items(), key=lambda item: item[1])
    if value <= 0:
        return "UNKNOWN"
    return mode


def inspect_schema(kg: Any) -> dict[str, Any]:
    node_counts = kg.query(
        """
        MATCH (n)
        UNWIND labels(n) AS label
        RETURN label, count(*) AS count
        ORDER BY count DESC
        """
    )
    rel_counts = kg.query(
        """
        MATCH ()-[r]->()
        RETURN type(r) AS relation, count(*) AS count
        ORDER BY count DESC
        """
    )
    property_counts = kg.query(
        """
        MATCH (n)
        UNWIND keys(n) AS property
        RETURN property, count(*) AS count
        ORDER BY count DESC
        """
    )
    return {
        "node_counts": node_counts,
        "relationship_counts": rel_counts,
        "node_property_counts": property_counts,
    }


def build_flat_sentence(
    area: Any,
    area_spatial: dict[str, Any],
    poi_cates: dict[str, list[dict[str, Any]]],
    road_types: dict[str, list[dict[str, Any]]],
    neighbors: dict[str, list[dict[str, Any]]],
    blocks: dict[str, dict[str, Any]],
    od_flows: dict[str, list[dict[str, Any]]],
    anchors: dict[str, list[Any]],
    binners: dict[str, Any],
    top_poi_cates: int,
    top_road_types: int,
    top_neighbors: int,
    top_od_flows: int,
) -> list[str]:
    area_tok = id_token("AREA", area.area_id)
    spatial = area_spatial.get(area.area_id)

    tokens = ["<BOS>", area_tok]

    if spatial is not None:
        tokens.append("<GEOMETRY>")
        tokens.extend(
            [
                "AREA_SIZE",
                attr_bin(spatial.area_m2, binners["area_m2"]),
                "COMPACTNESS",
                attr_bin(spatial.compactness, binners["compactness"]),
                "GRID",
                spatial.grid_token,
            ]
        )
        if spatial.ward_center_dist_m is not None and spatial.ward_relative_dir is not None:
            tokens.extend(
                [
                    "WARD_POS",
                    spatial.ward_relative_dir,
                    distance_bucket(spatial.ward_center_dist_m),
                ]
            )

        tokens.append("<ANCHOR>")
        for source, anchor_type in (("station", "STATION"), ("school", "SCHOOL")):
            nearest = nearest_anchor(spatial, anchors.get(source, []))
            if nearest is not None:
                tokens.extend(
                    [
                        "<ITEM>",
                        "ANCHOR_TYPE",
                        anchor_type,
                        "DIST",
                        distance_bucket(nearest["distance_m"]),
                        "DIR",
                        nearest["direction"],
                    ]
                )

    tokens.append("<ADMIN>")
    if area.ward_id:
        tokens.extend(["REL_BELONGS_TO", id_token("WARD", area.ward_id)])

    tokens.append("<LOCAL>")
    block_row = blocks.get(area.area_id, {})
    tokens.extend(
        [
            "BLOCK_COUNT",
            attr_bin(block_row.get("block_count", 0), binners["block_count"]),
            "BLOCK_POI_COUNT",
            attr_bin(block_row.get("block_poi_count", 0), binners["block_poi_count"]),
        ]
    )

    tokens.append("<POI>")
    for index, row in enumerate(sorted_poi_rows(poi_cates.get(area.area_id, []))[:top_poi_cates], start=1):
        tokens.extend(
            [
                "<ITEM>",
                poi_rank_token(index),
                id_token("CATE", row.get("cate_name"), row.get("cate_id")),
                "COUNT",
                attr_bin(row.get("poi_count"), binners["poi_count"]),
                "DENSITY",
                attr_bin(row.get("poi_density"), binners["poi_density"]),
                "DIST",
                distance_bucket(row.get("avg_dist_m")),
            ]
        )

    tokens.append("<ROAD>")
    spatial_area = area_spatial.get(area.area_id)
    for index, row in enumerate(sorted_road_rows(road_types.get(area.area_id, []), spatial_area)[:top_road_types], start=1):
        road_density = density(row.get("road_length"), spatial_area.area_km2 if spatial_area else 0.0)
        tokens.extend(
            [
                "<ITEM>",
                road_rank_token(index),
                id_token("ROADTYPE", row.get("road_type")),
                "COUNT",
                attr_bin(row.get("road_count"), binners["road_count"]),
                "LENGTH",
                attr_bin(row.get("road_length"), binners["road_length"]),
                "DENSITY",
                attr_bin(road_density, binners["road_density"]),
            ]
        )

    tokens.append("<NEIGHBOR>")
    for row in neighbors.get(area.area_id, [])[:top_neighbors]:
        tokens.extend(
            [
                "<ITEM>",
                "REL_BORDER_BY",
                id_token("AREA", row.get("neighbor_id")),
                "DIR",
                row.get("direction", "DIR_UNKNOWN"),
                "DIST",
                distance_bucket(row.get("centroid_dist_m")),
                "BORDER_LEN",
                attr_bin(row.get("border_len_m"), binners["border_len"]),
            ]
        )

    if area.ward_id:
        tokens.append("<MOBILITY>")
        od_rows = [
            row
            for row in od_flows.get(area.ward_id, [])
            if row.get("dest_ward_id") != area.ward_id
        ][:top_od_flows]
        for index, row in enumerate(od_rows, start=1):
            tokens.extend(
                [
                    "<ITEM>",
                    f"OD_RANK_{index}",
                    id_token("WARD", row.get("dest_ward_id")),
                    "TOTAL_TRIPS",
                    attr_bin(row.get("total_trips"), binners["total_trips"]),
                    "DOM_MODE",
                    dominant_od_mode(row),
                ]
            )

    tokens.append("<EOS>")
    return tokens


def build_vocab(corpus: list[dict[str, Any]], min_freq: int) -> dict[str, int]:
    counts = Counter(token for record in corpus for token in record["tokens"])
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for token, freq in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if token in vocab or freq < min_freq:
            continue
        vocab[token] = len(vocab)
    return vocab


def encode_corpus(corpus: list[dict[str, Any]], vocab: dict[str, int]) -> list[dict[str, Any]]:
    unk = vocab["<UNK>"]
    encoded = []
    for record in corpus:
        ids = [vocab.get(token, unk) for token in record["tokens"]]
        encoded.append(
            {
                "center": record["center"],
                "center_type": record["center_type"],
                "input_ids": ids[:-1],
                "labels": ids[1:],
                "length": len(ids),
            }
        )
    return encoded


def build_lm_samples(corpus: list[dict[str, Any]], vocab: dict[str, int], context_length: int) -> list[dict[str, Any]]:
    unk = vocab["<UNK>"]
    stride = max(1, context_length // 2)
    samples = []
    for record in corpus:
        ids = [vocab.get(token, unk) for token in record["tokens"]]
        if len(ids) <= context_length + 1:
            chunks = [(0, ids)]
        else:
            chunks = [
                (start, ids[start : start + context_length + 1])
                for start in range(0, len(ids) - 1, stride)
            ]

        for chunk_index, (_, chunk) in enumerate(chunks):
            if len(chunk) < 2:
                continue
            samples.append(
                {
                    "center": record["center"],
                    "center_type": record["center_type"],
                    "chunk_index": chunk_index,
                    "input_ids": chunk[:-1],
                    "labels": chunk[1:],
                    "length": len(chunk),
                }
            )
    return samples


def token_type_counts(vocab: dict[str, int]) -> dict[str, int]:
    prefixes = [
        "<",
        "OD_RANK_",
        "BIN_",
        "AREA_SIZE_",
        "AREA_",
        "WARD_",
        "POI_",
        "REL_",
        "CATE_",
        "ROADTYPE_",
        "COUNT_",
        "DENSITY_",
        "DIST_",
        "DIR_",
        "GRID_",
        "BORDER_LEN_",
        "COMPACTNESS_",
        "BLOCK_",
        "TOTAL_TRIPS_",
        "RAIL_",
        "BUS_",
        "CAR_",
        "WALK_",
        "LENGTH_",
    ]
    counts = {prefix: 0 for prefix in prefixes}
    counts["OTHER"] = 0
    for token in vocab:
        matched = False
        for prefix in prefixes:
            if token.startswith(prefix):
                counts[prefix] += 1
                matched = True
                break
        if not matched:
            counts["OTHER"] += 1
    return counts


def make_stats(
    corpus: list[dict[str, Any]],
    vocab: dict[str, int],
    lm_samples: list[dict[str, Any]],
    schema_summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    lengths = [len(record["tokens"]) for record in corpus]
    return {
        "tokenizer": "kg_vocab_v3_semidecoupled",
        "source": "neo4j",
        "num_sequences": len(corpus),
        "num_lm_samples": len(lm_samples),
        "vocab_size": len(vocab),
        "min_sequence_length": min(lengths) if lengths else 0,
        "max_sequence_length": max(lengths) if lengths else 0,
        "avg_sequence_length": round(sum(lengths) / len(lengths), 4) if lengths else 0,
        "median_sequence_length": statistics.median(lengths) if lengths else 0,
        "min_freq": args.min_freq,
        "context_length": args.context_length,
        "top_poi_cates": args.top_poi_cates,
        "top_road_types": args.top_road_types,
        "top_neighbors": args.top_neighbors,
        "top_od_flows": args.top_od_flows,
        "special_tokens": {token: vocab[token] for token in SPECIAL_TOKENS},
        "token_type_counts": token_type_counts(vocab),
        "schema_node_counts": schema_summary.get("node_counts", []),
        "schema_relationship_counts": schema_summary.get("relationship_counts", []),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semi-decoupled KG vocabulary corpus from Neo4j UrbanKG.")
    parser.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env", help="Path to .env file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of Area centers.")
    parser.add_argument("--min-freq", type=int, default=1, help="Minimum token frequency for vocabulary.")
    parser.add_argument("--context-length", type=int, default=256, help="LM sample context length.")
    parser.add_argument("--top-poi-cates", type=int, default=3, help="Top POI categories per Area.")
    parser.add_argument("--top-road-types", type=int, default=3, help="Top road types per Area.")
    parser.add_argument("--top-neighbors", type=int, default=5, help="Top neighbor Areas per Area.")
    parser.add_argument("--top-od-flows", type=int, default=1, help="Top non-self OD destination Wards per origin Ward.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kg = get_kg(args.env)
    projector = GeometryProjector()

    try:
        schema_summary = inspect_schema(kg)
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

        corpus = []
        for area in areas:
            center = id_token("AREA", area.area_id)
            corpus.append(
                {
                    "tokenizer": "kg_vocab_v3_semidecoupled",
                    "center": center,
                    "center_id": area.area_id,
                    "center_type": "Area",
                    "tokens": build_flat_sentence(
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
                    ),
                }
            )

        vocab = build_vocab(corpus, args.min_freq)
        id_to_token = {str(idx): token for token, idx in vocab.items()}
        encoded = encode_corpus(corpus, vocab)
        lm_samples = build_lm_samples(corpus, vocab, args.context_length)
        stats = make_stats(corpus, vocab, lm_samples, schema_summary, args)

        output_dir = args.output_dir
        write_json(output_dir / "kg_schema_summary.json", schema_summary)
        write_jsonl(output_dir / "urban_vocab_corpus.jsonl", corpus)
        write_json(output_dir / "urban_vocab.json", vocab)
        write_json(output_dir / "urban_id_to_token.json", id_to_token)
        write_jsonl(output_dir / "urban_vocab_corpus_ids.jsonl", encoded)
        write_jsonl(output_dir / "urban_vocab_lm_samples.jsonl", lm_samples)
        write_json(output_dir / "urban_vocab_stats.json", stats)

        print(f"Wrote {len(corpus)} KG vocabulary sequences to {output_dir / 'urban_vocab_corpus.jsonl'}")
        print(f"Vocabulary size: {len(vocab)}")
        print(f"LM samples: {len(lm_samples)}")
    finally:
        kg.close()


if __name__ == "__main__":
    main()
