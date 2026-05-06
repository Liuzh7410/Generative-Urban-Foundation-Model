# GUFM 2026

![GUFM](assets/GUFM.png)

This project explores how to convert a general urban knowledge graph
(UrbanKG) into structured **UrbanToken** representations that can be used as
inputs for Transformer-based urban foundation models.

The current workspace focuses on Tokyo 23 wards. It builds and uses an
UrbanKG containing urban entities such as `Ward`, `Area`, `Block`, `POI`,
`Road`, and `Cate`, together with relationships such as spatial containment,
area adjacency, POI categories, road membership, and ward-level OD flows.

The main research direction is:

```text
Urban data -> UrbanKG -> UrbanToken -> Transformer / GUFM
```

## Tokenization Methods

This project currently contains three UrbanKG tokenization directions.

### 1. Area-centered Discrete UrbanToken

Script:

```bash
python3 src/tokenization/extract_spatial_discrete_urban_tokens.py
```

Output:

```text
output/token/discrete/urban_tokens_area_centered.jsonl
```

Format:

```json
{
  "tokenizer": "spatial_discrete_v2",
  "center_type": "Area",
  "center": {
    "id": "area:13101001001",
    "name": "...",
    "town_code": "...",
    "ward_id": "...",
    "ward_name": "..."
  },
  "crs": {
    "source": "EPSG:4326",
    "metric": "EPSG:32654"
  },
  "structure": {
    "center": [],
    "hierarchy": [],
    "local_inventory": [],
    "spatial_neighbors": [],
    "mobility_context": []
  },
  "tokens": [["<URBAN>"], ["<CENTER>"], "...", ["</URBAN>"]],
  "token_count": 34
}
```

This method represents each `Area` as a structured symbolic memory. The token
sequence keeps explicit sections for geometry, administrative hierarchy, local
POI/road inventory, spatial neighbors, and ward-level mobility context.

### 2. KG Vocabulary Corpus Tokenization

Script:

```bash
python3 src/tokenization/extract_kg_vocab_corpus.py
```

Output directory:

```text
output/token/vocab/
```

Main outputs:

```text
urban_vocab_corpus.jsonl
urban_vocab_corpus_ids.jsonl
urban_vocab_lm_samples.jsonl
urban_vocab.json
urban_id_to_token.json
urban_vocab_stats.json
```

Format:

```json
{
  "tokenizer": "kg_vocab_v3_semidecoupled",
  "center": "AREA_13101001001",
  "center_id": "area:13101001001",
  "center_type": "Area",
  "tokens": ["<BOS>", "AREA_13101001001", "<GEOMETRY>", "AREA_SIZE", "BIN_HIGH", "...", "<EOS>"]
}
```

This method treats the UrbanKG as a corpus for pre-training. It converts
Area-centered facts into semi-decoupled token sequences, then builds a vocabulary
and id-mapped corpus. Compared with fully coupled tokens, slot/value tokens such
as `COUNT`, `BIN_HIGH`, `DENSITY`, and `DIST_100_250M` reduce sparsity and make
patterns easier for a Transformer to learn.

### 3. NAGphormer Hop2Token Embedding

Extraction script:

```bash
/opt/anaconda3/bin/python src/tokenization/extract_nagphormer_tokens.py
```

Training script:

```bash
/opt/anaconda3/bin/python src/trainers/train_nagphormer.py
```

Output directory:

```text
output/token/nagphormer/
```

Main outputs:

```text
nagphormer_data.pt
metadata.json
area_ids.json
feature_names.json
label_to_id.json
area_labels.json
nagphormer_best_model.pt
train_history.json
```

Format:

```text
nagphormer_data.pt
+-- hop_tokens: [num_area, hops + 1, feature_dim]
+-- features:   [num_area, feature_dim]
+-- labels:     [num_area]
+-- adjacency:  [num_area, num_area]
+-- splits:     train / val / test indices
```

This method follows NAGphormer-style continuous tokenization. It builds an
Area graph from `borderBy`, extracts an Area feature matrix `X`, normalizes the
adjacency matrix, and generates Hop2Token sequences:

```text
[X, A_hat X, A_hat^2 X, ..., A_hat^K X]
```

The current proxy task is Area-level dominant POI category classification.
