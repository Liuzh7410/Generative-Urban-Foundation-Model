# UrbanToken Design

This document describes the current UrbanToken design used in this project.
The goal is to convert a general Urban Knowledge Graph (UrbanKG) into token
representations that can be used by Transformer-based models and GUFM.

The core pipeline is:

```text
Neo4j UrbanKG -> subgraph extraction -> UrbanToken JSONL -> Transformer / GUFM
```

UrbanToken is KG-aware tokenization. It does not compress urban knowledge into
plain text. Instead, it preserves entities, relations, attributes, spatial
context, and local graph structure as discrete tokens.

## Data Source

UrbanToken is generated from the actual UrbanKG stored in Neo4j. Connection
settings are read from `.env`.

The current UrbanKG contains:

- Entities: `Ward`, `Area`, `Block`, `POI`, `Road`, `Cate`
- Relations:
  - `Area -[:belongsTo]-> Ward`
  - `Block -[:belongsTo]-> Area`
  - `POI -[:locatesAt]-> Area`
  - `POI -[:locatesAt]-> Block`
  - `POI -[:hasType]-> Cate`
  - `Road -[:in]-> Area`
  - `Area -[:borderBy]-> Area`
  - `Ward -[:hasODFlow]-> Ward`
  - `POI -[:nearby]-> Road`

`assets/Triple_schema.csv` is only used for schema understanding and statistics.
Token extraction should query Neo4j directly.

## V1: Discrete KG Tokenization

V1 is the first executable UrbanToken design.

Script:

```text
src/tokenization/extract_discrete_urban_tokens.py
```

Output:

```text
output/urban_tokens_area_centered_v1.jsonl
```

### Purpose

V1 converts an Area-centered KG subgraph into a list of discrete tokens. It is
mainly designed to test whether the existing UrbanKG can be represented in a
stable, interpretable, Transformer-readable format.

### Center Node

The current V1 implementation uses `Area` as the center node.

Each JSONL record represents one Area:

```json
{
  "tokenizer": "discrete_v1",
  "center_type": "Area",
  "center": {
    "id": "area:...",
    "name": "...",
    "town_code": "...",
    "ward_id": "ward:...",
    "ward_name": "..."
  },
  "tokens": [],
  "token_count": 0
}
```

### Token Sections

V1 produces flat list tokens. Each token is a small list with a type marker and
its KG elements.

Main token types:

- `CENTER`: center Area identity
- `HIER`: Area to Ward hierarchy
- `ATTR`: block count and block POI count
- `POI`: top POI categories in the Area
- `ROAD`: top road types in the Area
- `NEIGHBOR`: adjacent Areas
- `OD`: Ward-level OD flow context

Example:

```json
[
  ["CENTER", "AREA_13101001001"],
  ["HIER", "AREA_13101001001", "REL_BELONGS_TO", "WARD_13101"],
  ["POI", "AREA_13101001001", "REL_HAS_POI_CATE", "CATE_BUSINESS", "COUNT_HIGH"],
  ["ROAD", "AREA_13101001001", "REL_HAS_ROAD_TYPE", "ROADTYPE_6", "COUNT_HIGH", "LENGTH_HIGH"],
  ["NEIGHBOR", "AREA_13101001001", "REL_BORDER_BY", "AREA_13101001002"],
  ["OD", "WARD_13101", "REL_HAS_OD_FLOW", "WARD_13103", "TOTAL_TRIPS_HIGH"]
]
```

### Discretization

Numeric values are converted into quantile bins:

```text
LOW / MED / HIGH
```

V1 therefore avoids raw numeric values and produces a compact symbolic
representation.

### Limitations

V1 is intentionally simple, but it has two main limitations:

- The subgraph is mostly flat: it is represented as independent list tokens.
- It only uses knowledge already stored as KG nodes, relations, and properties.
  It does not add derived spatial measurements such as distance, direction,
  density, area size, or boundary length.

## V2: Spatial Discrete Tokenization

V2 extends V1 by adding spatial information and a more structured output format.

Script:

```text
src/tokenization/extract_spatial_discrete_urban_tokens_v2.py
```

Output:

```text
output/urban_tokens_area_spatial_discrete_v2.jsonl
```

### Purpose

V2 is designed to make UrbanToken more spatially expressive. It still starts
from Neo4j UrbanKG, but it derives additional spatial features from geometry.

Neo4j geometry is read as WKT. Following the KG build script, geometries are
assumed to be stored in `EPSG:4326`. V2 projects geometries to `EPSG:32654`
before computing metric features for Tokyo.

### Structured Output

V2 outputs both:

- `structure`: sectioned token groups
- `tokens`: a linearized sequence for model input

The main structure is:

```json
{
  "structure": {
    "center": [],
    "hierarchy": [],
    "local_inventory": [],
    "spatial_neighbors": [],
    "mobility_context": []
  },
  "tokens": []
}
```

### V2 Sections

#### center

Describes the center Area itself and its spatial position.

Tokens include:

- `CENTER`
- `SPATIAL`: area size, compactness, grid token
- `WARD_REL_POS`: direction and distance to Ward center
- `ANCHOR`: nearest station and nearest school

Example:

```json
[
  ["CENTER", "AREA_13101001001"],
  ["SPATIAL", "AREA_13101001001", "AREA_SIZE_HIGH", "COMPACTNESS_LOW", "GRID_3568_13977"],
  ["WARD_REL_POS", "AREA_13101001001", "DIR_SE", "DIST_1_2KM"],
  ["ANCHOR", "AREA_13101001001", "NEAREST_STATION", "POI_xxx", "DIST_100_250M", "DIR_SE"]
]
```

#### hierarchy

Preserves administrative hierarchy.

Example:

```json
[["HIER", "AREA_13101001001", "REL_BELONGS_TO", "WARD_13101"]]
```

#### local_inventory

Summarizes local urban contents inside the Area.

Tokens include:

- `BLOCK`: block count and block POI count
- `POI_CATE`: POI category count, density, and average distance to Area center
- `ROAD_TYPE`: road count, road length, and road density

Example:

```json
[
  ["POI_CATE", "AREA_13101001001", "CATE_MEAL", "COUNT_HIGH", "DENSITY_HIGH", "DIST_100_250M"],
  ["ROAD_TYPE", "AREA_13101001001", "ROADTYPE_6", "COUNT_HIGH", "LENGTH_HIGH", "DENSITY_HIGH"]
]
```

#### spatial_neighbors

Adds structured spatial relations to adjacent Areas.

Tokens include:

- neighbor identity
- relative direction
- centroid distance
- shared boundary length

Example:

```json
[
  ["NEIGHBOR_HOP_1", "AREA_13101001001", "REL_BORDER_BY", "AREA_13101001002", "DIR_SW", "DIST_250_500M", "BORDER_LEN_HIGH"]
]
```

#### mobility_context

Preserves Ward-level OD flows.

Example:

```json
[
  ["OD", "WARD_13101", "REL_HAS_OD_FLOW", "WARD_13103", "TOTAL_TRIPS_HIGH", "RAIL_HIGH", "BUS_HIGH", "CAR_HIGH", "WALK_HIGH"]
]
```

### V2 Spatial Tokens

V2 adds the following spatial token families:

- `AREA_SIZE_LOW/MED/HIGH`
- `COMPACTNESS_LOW/MED/HIGH`
- `GRID_lat_lon`
- `DIR_N/NE/E/SE/S/SW/W/NW`
- `DIST_0_100M`, `DIST_100_250M`, `DIST_250_500M`,
  `DIST_500_1000M`, `DIST_1_2KM`, `DIST_2_5KM`, `DIST_5KM_PLUS`
- `DENSITY_LOW/MED/HIGH`
- `BORDER_LEN_LOW/MED/HIGH`

### Linearized Tokens

V2 keeps the sectioned structure but also provides a linearized version:

```json
[
  ["<URBAN>"],
  ["<CENTER>"],
  ["CENTER", "AREA_13101001001"],
  ["SPATIAL", "..."],
  ["<HIER>"],
  ["HIER", "..."],
  ["<LOCAL>"],
  ["POI_CATE", "..."],
  ["<NEIGHBOR>"],
  ["NEIGHBOR_HOP_1", "..."],
  ["<OD>"],
  ["OD", "..."],
  ["</URBAN>"]
]
```

## Summary

V1 and V2 are complementary:

- **V1** verifies that the existing Neo4j UrbanKG can be converted into stable
  discrete Area-centered tokens.
- **V2** strengthens V1 with spatial measurements and structured sections,
  making the token representation more suitable for spatial reasoning.

The current development direction is to use V2 as the stronger discrete
UrbanToken baseline, while leaving room for future vocabulary-based
tokenization methods inspired by BPE or KG-specific token learning.
