# KG Vocabulary Tokenization Design

This document describes the vocabulary-based UrbanToken method for converting
Neo4j UrbanKG into language-model-style training data.

This method is separate from the Area-centered discrete memory tokenization in
`URBAN_TOKEN_DESIGN.md`.

The memory method is designed for retrieval, conditioning, and explicit
reasoning over a centered urban subgraph. The vocabulary method is designed to
construct a KG-derived symbolic corpus for autoregressive pretraining.

The current final version is:

```text
kg_vocab_v3_semidecoupled
```

Implemented by:

```text
src/tokenization/extract_kg_vocab_corpus.py
```

Default output:

```text
output/token/vocab_structured/
```

## 1. Goal

The goal is to treat UrbanKG as a structured symbolic corpus:

```text
Neo4j UrbanKG
-> Area-centered fact extraction
-> KG-aware token sequence
-> vocabulary
-> token ID corpus
-> autoregressive LM samples
```

Unlike BPE or natural-language tokenization, this method does not split KG
identifiers into subwords. Entity, relation, section, slot, and discretized
attribute symbols are treated as atomic tokens.

Examples:

```text
AREA_13101001001
WARD_13101
CATE_BUSINESS
ROADTYPE_9
REL_BELONGS_TO
COUNT
BIN_HIGH
DIST_250_500M
DIR_NE
```

The intended learning target is not raw text. It is regularities in urban KG
structure, such as:

```text
primary POI category -> count bin -> density bin -> distance bin
road type -> length bin -> density bin
neighbor direction -> distance bin -> boundary length bin
ward mobility -> dominant mode
```

## 2. Data Source

The source of truth is Neo4j UrbanKG, not previously generated token files.

The script reads `.env`:

```text
NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
NEO4J_DATABASE=
```

The tokenizer queries the actual graph schema and stores a schema snapshot in:

```text
kg_schema_summary.json
```

Current UrbanKG entities include:

- `Ward`
- `Area`
- `Block`
- `POI`
- `Road`
- `Cate`

Current relations include:

- `Area -[:belongsTo]-> Ward`
- `Block -[:belongsTo]-> Area`
- `POI -[:locatesAt]-> Area`
- `POI -[:hasType]-> Cate`
- `Road -[:in]-> Area`
- `Area -[:borderBy]-> Area`
- `Ward -[:hasODFlow]-> Ward`

Geometry is read from Neo4j WKT. Following the KG build script, WKT is assumed
to be stored in `EPSG:4326`. Metric features are computed after projection to
`EPSG:32654`.

## 3. Output Files

The script generates:

```text
urban_vocab_corpus.jsonl
urban_vocab.json
urban_id_to_token.json
urban_vocab_corpus_ids.jsonl
urban_vocab_lm_samples.jsonl
urban_vocab_stats.json
kg_schema_summary.json
```

### 3.1 `urban_vocab_corpus.jsonl`

Human-readable token corpus.

Each line is one Area-centered KG sentence:

```json
{
  "tokenizer": "kg_vocab_v3_semidecoupled",
  "center": "AREA_13101001001",
  "center_id": "area:13101001001",
  "center_type": "Area",
  "tokens": ["<BOS>", "AREA_13101001001", "...", "<EOS>"]
}
```

### 3.2 `urban_vocab.json`

Token-to-ID vocabulary:

```json
{
  "<PAD>": 0,
  "<BOS>": 1,
  "<EOS>": 2,
  "<UNK>": 3,
  "AREA_13101001001": 4
}
```

Special token IDs are fixed:

```text
<PAD> = 0
<BOS> = 1
<EOS> = 2
<UNK> = 3
```

### 3.3 `urban_id_to_token.json`

Reverse mapping from ID to token. This is used for decoding model outputs and
debugging.

### 3.4 `urban_vocab_corpus_ids.jsonl`

ID-encoded corpus. Each line contains:

```json
{
  "center": "AREA_13101001001",
  "center_type": "Area",
  "input_ids": [1, 10, 20],
  "labels": [10, 20, 2],
  "length": 4
}
```

This follows autoregressive language modeling:

```text
input_ids = ids[:-1]
labels    = ids[1:]
```

### 3.5 `urban_vocab_lm_samples.jsonl`

Training-ready LM samples.

Currently most sequences fit within the default context length, so this file is
usually identical in count to `urban_vocab_corpus_ids.jsonl`. If future
sequences exceed `context_length`, this file will contain sliding-window
chunks.

### 3.6 `urban_vocab_stats.json`

Corpus and vocabulary statistics:

- number of sequences
- vocabulary size
- min / max / average / median sequence length
- token type counts
- top-k settings
- schema node counts
- schema relation counts

## 4. Development History

The vocabulary method went through three stages.

## 4.1 Initial Flat Vocabulary Corpus

The first version directly flattened Area-centered KG facts:

```text
<BOS>
AREA_13101001001
REL_HAS_AREA_SIZE AREA_SIZE_HIGH
REL_HAS_COMPACTNESS COMPACTNESS_LOW
REL_HAS_POI_CATE CATE_BUSINESS COUNT_HIGH DENSITY_HIGH DIST_250_500M
REL_BORDER_BY AREA_13101001002 DIR_SW DIST_250_500M BORDER_LEN_HIGH
REL_HAS_OD_FLOW WARD_13103 TOTAL_TRIPS_HIGH RAIL_HIGH BUS_HIGH CAR_HIGH WALK_HIGH
<EOS>
```

Problems:

- The sequence had many repeated relation tokens.
- The corpus was flat and had weak structural hints.
- OD tokens were repeated for all Areas in the same Ward.
- `TOTAL_TRIPS_HIGH`, `RAIL_HIGH`, etc. were often too similar across Areas.
- The model could overfit to repeated token patterns rather than learning
  useful structure.

## 4.2 Structured Compact Corpus

The second version added section markers and compressed multi-field attributes
into profile tokens:

```text
<BOS>
AREA_13101001001
<GEOMETRY>
GEO_PROFILE_AREA_SIZE_HIGH_COMPACTNESS_LOW
GRID_3568_13977
WARD_POS_DIR_SE_DIST_1_2KM
<POI>
POI_PRIMARY CATE_BUSINESS POI_PROFILE_COUNT_HIGH_DENSITY_HIGH_DIST_250_500M
<ROAD>
ROAD_PRIMARY ROADTYPE_6 ROAD_PROFILE_COUNT_MED_LENGTH_MED_DENSITY_MED
<MOBILITY>
OD_RANK_1 WARD_13103 OD_PROFILE_TOTAL_TRIPS_HIGH_OD_DOM_RAIL
<EOS>
```

Improvements:

- Section markers made the sequence easier for Transformers to parse.
- Sequence length decreased significantly.
- POI and Road items received rank tokens such as `POI_PRIMARY` and
  `ROAD_PRIMARY`.
- OD was compressed to fewer destination records.

Problems:

- Profile tokens were too tightly bound.
- Tokens such as `POI_PROFILE_COUNT_HIGH_DENSITY_HIGH_DIST_250_500M` created
  sparse combinations.
- The model could memorize profile tokens instead of learning reusable
  relations such as `COUNT -> BIN_HIGH` or `DENSITY -> BIN_HIGH`.

## 4.3 Final Version: V3 Semi-Decoupled Structured Corpus

The current final version keeps section structure and ranking but breaks
profile tokens into slot-value pairs.

Example:

```text
<BOS>
AREA_13101001001

<GEOMETRY>
AREA_SIZE BIN_HIGH
COMPACTNESS BIN_LOW
GRID GRID_3568_13977
WARD_POS DIR_SE DIST_1_2KM

<ANCHOR>
<ITEM> ANCHOR_TYPE STATION DIST DIST_100_250M DIR DIR_SE
<ITEM> ANCHOR_TYPE SCHOOL  DIST DIST_250_500M DIR DIR_SE

<ADMIN>
REL_BELONGS_TO WARD_13101

<LOCAL>
BLOCK_COUNT BIN_LOW
BLOCK_POI_COUNT BIN_HIGH

<POI>
<ITEM> RANK_PRIMARY   CATE_BUSINESS     COUNT BIN_HIGH DENSITY BIN_HIGH DIST DIST_250_500M
<ITEM> RANK_SECONDARY CATE_LIFE_RELATED COUNT BIN_HIGH DENSITY BIN_HIGH DIST DIST_100_250M
<ITEM> RANK_TERTIARY  CATE_MEAL         COUNT BIN_HIGH DENSITY BIN_HIGH DIST DIST_100_250M

<ROAD>
<ITEM> RANK_PRIMARY   ROADTYPE_6 COUNT BIN_MED LENGTH BIN_MED DENSITY BIN_MED
<ITEM> RANK_SECONDARY ROADTYPE_3 COUNT BIN_MED LENGTH BIN_MED DENSITY BIN_LOW
<ITEM> RANK_TERTIARY  ROADTYPE_9 COUNT BIN_MED LENGTH BIN_MED DENSITY BIN_LOW

<NEIGHBOR>
<ITEM> REL_BORDER_BY AREA_13101001002 DIR DIR_SW DIST DIST_250_500M BORDER_LEN BIN_HIGH

<MOBILITY>
<ITEM> OD_RANK_1 WARD_13103 TOTAL_TRIPS BIN_HIGH DOM_MODE RAIL

<EOS>
```

### Why V3 Is Preferred

V3 addresses three issues in the compact version.

#### 1. Avoid over-binding

Instead of:

```text
POI_PROFILE_COUNT_HIGH_DENSITY_HIGH_DIST_250_500M
```

V3 uses:

```text
COUNT BIN_HIGH DENSITY BIN_HIGH DIST DIST_250_500M
```

This lets the model separately learn slots and values.

#### 2. Reduce sparse combination tokens

Reusable values such as:

```text
BIN_LOW
BIN_MED
BIN_HIGH
DIR_NE
DIST_250_500M
```

appear across many sections and therefore receive denser embedding updates.

#### 3. Encourage learning regularities instead of memorization

The model can learn patterns such as:

```text
COUNT BIN_HIGH -> DENSITY BIN_HIGH
RANK_PRIMARY -> CATE_* -> COUNT BIN_*
DIR_* -> DIST_* -> BORDER_LEN BIN_*
```

rather than memorizing a single rare profile token.

## 5. Final Token Grammar

The current Area-centered sentence follows this order:

```text
<BOS>
AREA_xxx
<GEOMETRY>
<ANCHOR>
<ADMIN>
<LOCAL>
<POI>
<ROAD>
<NEIGHBOR>
<MOBILITY>
<EOS>
```

### 5.1 Geometry

```text
<GEOMETRY>
AREA_SIZE BIN_LOW|BIN_MED|BIN_HIGH
COMPACTNESS BIN_LOW|BIN_MED|BIN_HIGH
GRID GRID_xxx
WARD_POS DIR_* DIST_*
```

### 5.2 Anchor

Nearest station and school:

```text
<ANCHOR>
<ITEM> ANCHOR_TYPE STATION DIST DIST_* DIR DIR_*
<ITEM> ANCHOR_TYPE SCHOOL  DIST DIST_* DIR DIR_*
```

### 5.3 Admin

```text
<ADMIN>
REL_BELONGS_TO WARD_xxx
```

### 5.4 Local

```text
<LOCAL>
BLOCK_COUNT BIN_*
BLOCK_POI_COUNT BIN_*
```

### 5.5 POI

Top POI categories, sorted by:

```text
poi_count desc -> poi_density desc -> avg_dist asc -> cate_name
```

Grammar:

```text
<POI>
<ITEM> RANK_PRIMARY   CATE_xxx COUNT BIN_* DENSITY BIN_* DIST DIST_*
<ITEM> RANK_SECONDARY CATE_xxx COUNT BIN_* DENSITY BIN_* DIST DIST_*
<ITEM> RANK_TERTIARY  CATE_xxx COUNT BIN_* DENSITY BIN_* DIST DIST_*
```

Default:

```text
top_poi_cates = 3
```

### 5.6 Road

Top road types, sorted by:

```text
road_count desc -> road_length desc -> road_density desc -> road_type
```

Grammar:

```text
<ROAD>
<ITEM> RANK_PRIMARY   ROADTYPE_x COUNT BIN_* LENGTH BIN_* DENSITY BIN_*
<ITEM> RANK_SECONDARY ROADTYPE_x COUNT BIN_* LENGTH BIN_* DENSITY BIN_*
<ITEM> RANK_TERTIARY  ROADTYPE_x COUNT BIN_* LENGTH BIN_* DENSITY BIN_*
```

Default:

```text
top_road_types = 3
```

### 5.7 Neighbor

Neighbor Areas are sorted by:

```text
border_length desc -> centroid_distance asc -> neighbor_id
```

Grammar:

```text
<NEIGHBOR>
<ITEM> REL_BORDER_BY AREA_xxx DIR DIR_* DIST DIST_* BORDER_LEN BIN_*
```

Default:

```text
top_neighbors = 5
```

### 5.8 Mobility

OD is Ward-level, so it is intentionally compressed in the Area sentence.

Current rule:

- skip self-flow
- keep top non-self OD destinations
- keep only total trip bin and dominant mode

Grammar:

```text
<MOBILITY>
<ITEM> OD_RANK_1 WARD_xxx TOTAL_TRIPS BIN_* DOM_MODE RAIL|BUS|CAR|WALK|UNKNOWN
```

Default:

```text
top_od_flows = 1
```

This avoids repeating the same full Ward OD vector for every Area in the same
Ward.

## 6. Vocabulary Construction

After corpus generation:

1. Count all token frequencies.
2. Initialize special tokens:

```text
<PAD>, <BOS>, <EOS>, <UNK>
```

3. Add all tokens with frequency >= `min_freq`.
4. Save `urban_vocab.json` and `urban_id_to_token.json`.

Default:

```text
min_freq = 1
```

Because `AREA_xxx` tokens are meaningful even if each appears once.

## 7. ID Encoding and LM Samples

Each token sequence is converted to IDs:

```python
ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
```

Autoregressive training pairs:

```text
input_ids = ids[:-1]
labels    = ids[1:]
```

If a sequence exceeds `context_length`, it is split into sliding-window samples.

Default:

```text
context_length = 256
stride = context_length // 2
```

## 8. Current Run Command

Generate the final V3 semi-decoupled corpus:

```bash
python3 src/tokenization/extract_kg_vocab_corpus.py
```

Output:

```text
output/token/vocab_structured/
```

Useful optional arguments:

```bash
python3 src/tokenization/extract_kg_vocab_corpus.py \
  --top-poi-cates 3 \
  --top-road-types 3 \
  --top-neighbors 5 \
  --top-od-flows 1 \
  --context-length 256
```

## 9. Comparison with Memory Tokenization

### Memory Tokenization

Output:

```text
output/token/urban_tokens_area_centered_v2.jsonl
```

Characteristics:

- nested / sectioned memory
- better for retrieval and reasoning
- keeps explicit local subgraph-like context
- useful as external memory for downstream models

### Vocabulary Tokenization

Output:

```text
output/token/vocab_structured/
```

Characteristics:

- flat token sequence with section markers
- designed for autoregressive pretraining
- optimized to reduce sparse profile tokens
- better for learning reusable KG regularities

## 10. Future Extensions

Potential next steps:

1. Generate a pattern corpus by replacing entity IDs with abstract roles:

```text
AREA_SELF
WARD_PARENT
NEIGHBOR_RANK_1
OD_DEST_RANK_1
```

2. Generate separate Ward-level OD corpus instead of including OD in every Area
sentence.

3. Add path-based or random-walk KG corpus:

```text
Area -> POI -> Cate
Area -> Road -> Area
Ward -> Ward
```

4. Compare:

- memory tokens
- vocabulary tokens
- pattern vocabulary tokens
- hybrid pretraining + retrieval memory
