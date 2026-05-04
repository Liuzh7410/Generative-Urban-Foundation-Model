# UrbanToken V1 Design (KG-based, Neo4j-driven)

## 1. Overview

This document defines the **UrbanToken V1 design**, aiming to transform an existing Urban Knowledge Graph (UrbanKG) into a structured token sequence suitable for Transformer-based models.

The purpose of UrbanToken is to:

- Provide a structured and learnable representation of urban knowledge
- Preserve graph semantics (entity, relation, attribute)
- Enable usage in:
  - generative models (GUFM)
  - retrieval-based pipelines
  - future pretraining tasks

## 2. Data Source Assumption

### Important Constraints

- UrbanToken must be constructed based on actual KG schema
- The KG is stored in Neo4j, accessed via `.env`
- Triple_schema.csv is only used for schema understanding and statistics

### Key Principle

Neo4j Query → Subgraph Extraction → Token Sequence

## 3. Design Principles

UrbanToken is based on three dimensions:

- Spatial (Where)
- Semantic (What)
- Relational (How)

## 4. KG-Aware Tokenization

UrbanToken encodes structured knowledge instead of compressing text.

## 5. Vocabulary Design (V1)

### Token Categories

#### Entity Tokens
AREA_xxx, WARD_xxx, CITY_xxx, POI_xxx, ROAD_xxx

#### Relation Tokens
REL_BELONGS_TO, REL_HAS_POI, REL_NEARBY, REL_CONNECTED

#### Attribute Tokens
LANDUSE_RESIDENTIAL, LANDUSE_COMMERCIAL  
POP_LOW, POP_MED, POP_HIGH

#### Structural Tokens
<URBAN>, </URBAN>, <CENTER>, <HIER>, <ATTR>, <POI>, <NEIGHBOR>

## 6. Centered Subgraph Representation

Center node can be:
- Area
- Ward
- POI cluster

## 7. UrbanToken Sequence Format

<URBAN>

<CENTER>
AREA_001

<HIER>
AREA_001 REL_BELONGS_TO WARD_01
WARD_01 REL_BELONGS_TO CITY_TOKYO

<ATTR>
AREA_001 LANDUSE_COMMERCIAL
AREA_001 POP_HIGH

<POI>
AREA_001 REL_HAS_POI POI_CAFE
AREA_001 REL_HAS_POI POI_RESTAURANT

<NEIGHBOR>
AREA_001 REL_NEARBY AREA_002
AREA_001 REL_NEARBY AREA_003

</URBAN>

## 8. Token Construction Pipeline

1. Query Neo4j
2. Extract subgraph
3. Convert to tokens
4. Assemble structured sequence

## 9. Discretization Strategy

Use quantile-based binning for attributes:
LOW / MED / HIGH

## 10. Token Size Control

- POI: top 5
- neighbors: top 5

## 11. Output Format

{
  "center": "AREA_001",
  "tokens": ["<URBAN>", "<CENTER>", "AREA_001", ...]
}

## 12. Validation

- Interpretability
- Distinguishability
- Token length

## 13. Summary

UrbanToken V1 enables:

UrbanKG → Token → Transformer
