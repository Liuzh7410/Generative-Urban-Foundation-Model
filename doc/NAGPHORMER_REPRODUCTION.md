# NAGphormer Reproduction Plan (UrbanKG Version)

## 0. Objective

Reproduce the core methodology of NAGphormer:
"Tokenized Graph Transformer with Neighborhood Aggregation"

Adapt it to Urban Knowledge Graph (UrbanKG) stored in Neo4j.

Key idea:
Each node (Area) is represented as a sequence of tokens derived from multi-hop neighborhood aggregation (Hop2Token), and processed by a Transformer.

---

## 1. Key Mapping: Paper → UrbanKG

| NAGphormer Concept | UrbanKG Implementation |
|------------------|----------------------|
| Node             | Area (中心实体) |
| Graph            | Neo4j UrbanKG |
| Feature Matrix X | Area attribute vectors |
| Adjacency A      | Area–Area spatial relations |
| Token            | k-hop aggregated embedding |
| Sequence         | [x0, x1, ..., xK] |
| Task             | Node-level prediction (proxy task) |

---

## 2. Data Extraction from Neo4j

### 2.1 Connect to Neo4j

Use `.env`:

NEO4J_URI=neo4j://localhost:7687  
NEO4J_USERNAME=neo4j  
NEO4J_PASSWORD=***

---

### 2.2 Extract Nodes (Area)

Cypher:

MATCH (a:Area)  
RETURN id(a) AS node_id, a.name AS name

---

### 2.3 Build Adjacency (Graph Structure)

MATCH (a:Area)-[:BORDER_BY]->(b:Area)  
RETURN id(a) AS src, id(b) AS dst  

- Build undirected graph  
- Add self-loop later  

---

### 2.4 Extract Node Features

Example (simple version):

MATCH (a:Area)  
OPTIONAL MATCH (a)<-[:LOCATED_IN]-(p:POI)  
RETURN id(a) AS node_id, count(p) AS poi_count  

Better version:

x_v = [
  #POI_total,
  #POI_food,
  #POI_transport,
  #roads,
  ...
]

---

## 3. Preprocessing

### 3.1 Feature Matrix

X ∈ R^(N × d)

---

### 3.2 Normalize Adjacency

A_hat = D^{-1/2} (A + I) D^{-1/2}

---

## 4. Hop2Token (CORE)

X_k = A_hat^k X

Implementation:

X0 = X  
X1 = A_hat @ X0  
X2 = A_hat @ X1  
...  
XK = A_hat^K @ X  

---

## 5. Token Sequence per Node

Sv = [x_v^0, x_v^1, ..., x_v^K]

Shape: (K+1, d)

---

## 6. Transformer Input

Linear projection:

Z_v^0 = [x0 E, x1 E, ..., xK E]

Transformer encoder:

- Multi-head attention  
- Feed-forward  
- LayerNorm  

---

## 7. Readout Function

Attention-based:

α_k = attention(x0, xk)

Z_out = x0 + Σ α_k xk

---

## 8. Task Design

Option A: Predict Area functional type  
Option B: Predict POI density class  
Option C: Predict OD cluster  

---

## 9. Training

Loss: CrossEntropyLoss  

Hyperparameters:

- K: 2–10  
- hidden dim: 128–512  
- layers: 1–3  

---

## 10. Output

Z_out ∈ R^(d_model)

---

## 11. Evaluation

- Accuracy  
- F1 score  

---

## 12. Comparison

1. Discrete Urban Token  
2. Vocabulary Token  
3. NAGphormer  

---

## 13. Notes

Do NOT:

- serialize KG into text  
- use BPE  
- use LLM  

Must:

- use matrix multiplication  
- train Transformer  
- use node supervision  

---

## 14. Expected Outcome

- embedding-based baseline  
- multi-hop representation  
- comparison with symbolic tokens  

---

## 15. Insight

NAGphormer: continuous tokenization  
UrbanToken: symbolic tokenization  

Github: https://github.com/JHL-HUST/NAGphormer