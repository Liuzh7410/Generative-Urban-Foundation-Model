# Generative Urban Foundation Model (GUFM) Design Note

## 1. Motivation

東京大学CSISの擬似人流は、人口分布生成、施設選択、活動生成、軌跡生成といった階層的な生成パイプラインを持つ。これは統計的に整合した大規模人流を生成するうえで強力である。一方、Generative Urban Foundation Model (GUFM) として発展させるためには、単に既存都市の人流を再現するだけでは不十分である。

GUFMの目的は、都市を入力として人流を生成する条件付き世界モデルを構築することである。つまり、都市構造や都市状態が変化したときに、人流がどのように応答するかを生成できるモデルを目指す。

従来モデルは次のように表せる。

```math
p(\text{mobility})
```

GUFMでは次の条件付き分布を学習する。

```math
p(\text{mobility} \mid \text{urban state}, \text{context}, \text{intervention})
```

さらに時系列的には次のように表せる。

```math
p(e_{t+1} \mid e_{\leq t}, U_t, z)
```

ここで、`e_t` は人流イベント、`U_t` は時刻 `t` の都市状態、`z` は個人・世帯属性を表す。

---

## 2. Core Concept

GUFMの本質は、都市を固定データや単なる特徴量として扱うのではなく、生成過程の条件そのものとして扱うことである。

### Conventional View

```text
urban features = [population_density, poi_count, road_length, ...]
mobility = model(urban features, person features)
```

この設計では、都市は単なる説明変数となり、空間構造や介入可能性が弱くなる。

### GUFM View

```text
Urban tokens + Person tokens + Mobility history tokens
→ Generative model
→ Next mobility event
```

都市をトークン列または構造化メモリとして表現し、人流トークンがその都市メモリを参照しながら次の行動を生成する。

---

## 3. Generalization, Transfer, and Intervention Response

GUFMでは、以下の3つの能力が重要になる。

### 3.1 Generalization

未知都市や未学習地域に対して、それらしい人流を生成できる能力である。

例:

```text
Train: Tokyo, Osaka
Test: Nagoya or another unseen city
```

重要なのは、都市名を覚えることではなく、土地利用、POI分布、交通接続、人口構造などから都市の機能を理解することである。

### 3.2 Transfer

ある条件で学習した行動知識を、別の条件へ転用する能力である。

例:

```text
weekday → weekend
sunny day → rainy day
Tokyo → regional city
```

ここで転移されるべきものは、単なるODパターンではなく、通勤、買い物、食事、混雑回避、寄り道といった抽象的な行動文法である。

### 3.3 Intervention Response

都市や環境への介入に対して、人流の変化を生成できる能力である。

例:

```text
new station added
road blocked
large event held
railway disruption
disaster occurs
```

GUFMでは、介入をモデル外の後処理として与えるのではなく、都市トークンの追加、削除、更新として入力に反映する。

---

## 4. Urban Token Design

都市トークンは、都市を生成条件として記述するための基本単位である。都市を単一のベクトルに圧縮するのではなく、空間、属性、関係、時間状態を持つトークン群として表現する。

### 4.1 Spatial Tokens

場所を表すトークンである。

```text
CITY_TOKYO
WARD_SHIBUYA
CELL_533945
H3_8FA23
STATION_SHIBUYA
```

GUFMでは、空間を単一粒度に固定するのではなく、階層的に扱うことが望ましい。

```text
City → Ward/District → Mesh/Cell → POI/Station
```

### 4.2 Semantic Attribute Tokens

場所の意味を表すトークンである。

```text
LANDUSE_COMMERCIAL
LANDUSE_RESIDENTIAL
POI_FOOD_HIGH
POI_OFFICE_MED
POI_SCHOOL_LOW
POP_DENSITY_HIGH
EMPLOYMENT_DENSITY_MED
```

### 4.3 Network and Relation Tokens

都市内の接続関係を表すトークンである。

```text
EDGE_CELL_1001_CELL_1002
CONNECTED_BY_ROAD
CONNECTED_BY_RAIL
WALK_5MIN
TRAIN_12MIN
ADJACENT_CELL
```

### 4.4 Dynamic Context Tokens

都市状態は時間により変化する。したがって、静的な都市属性だけでなく、動的状態もトークン化する。

```text
TIME_08_00
WEEKDAY
WEATHER_RAIN
CROWD_HIGH
EVENT_FIREWORK
DISRUPTION_RAIL_STOP
```

---

## 5. Mobility Token Design

人流トークンは、単なる緯度経度列ではなく、意思決定イベントの列として設計する。

### 5.1 Why Not Raw Coordinates

生の緯度経度列は以下の理由からGUFMには向かない。

1. 長すぎて計算効率が悪い。
2. 行動の意味が失われやすい。
3. 都市条件との接続が弱い。
4. 介入応答を表現しにくい。

### 5.2 Event-Based Mobility Tokens

人流は次のようなイベント列として表現する。

```text
[ACT_HOME_END, T=07:45, ORIG=CELL_1001]
[DEPART, MODE=WALK]
[BOARD, MODE=TRAIN, STATION=ST_21]
[ARRIVE, DEST=CELL_2203]
[ACT_WORK_START, DUR=4H_PLUS]
[ACT_WORK_END, T=12:00]
[ACT_EAT_START, DEST=CELL_2205, MODE=WALK, DUR=30_60]
```

### 5.3 Minimal Mobility Token Set

最小構成としては、以下の4種類が必要である。

```text
ACT_x
DEST_CELL_y
TIME_BIN_z
MODE_m
```

例:

```text
ACT_WORK
DEST_CELL_2203
TIME_09_00
MODE_TRAIN
DUR_4H_PLUS
```

### 5.4 Extended Mobility Tokens

より高度なモデルでは、以下のようなトークンを追加する。

```text
INTENT_COMMUTE
INTENT_LEISURE
DETOUR
DELAY
CANCEL
COMPANION_FAMILY
HABITUAL_DESTINATION
IRREGULAR_EVENT
```

これにより、寄り道、予定変更、遅延、同行行動などのlong-tail mobilityを表現しやすくなる。

---

## 6. Urban-Mobility Attention Design

GUFMの中核は、都市トークンと人流トークンの結合方法にある。単純に都市トークンと人流トークンを結合してTransformerに入力するだけでは弱い。より望ましいのは、人流トークンが都市トークンを参照するstructured cross-attentionである。

### 6.1 Basic Cross-Attention

```math
\text{Attn}(Q_m, K_u, V_u)
```

ここで、

- `Q_m`: mobility tokenから得られるquery
- `K_u`: urban tokenから得られるkey
- `V_u`: urban tokenから得られるvalue

を表す。

直感的には、現在の人流状態が「次にどの都市要素を見るべきか」を学習する。

### 6.2 Example

人流側:

```text
ACT_WORK_END
TIME_12_00
CURRENT_CELL_2203
```

都市側:

```text
CELL_2205 POI_FOOD_HIGH WALK_5MIN CROWD_MED
CELL_2210 POI_OFFICE_HIGH WALK_3MIN CROWD_HIGH
CELL_2300 POI_PARK_MED WALK_20MIN CROWD_LOW
```

このとき、昼食行動を生成する場合、モデルは `POI_FOOD_HIGH` や `WALK_5MIN` を持つセルに高いattentionを向けることが期待される。

### 6.3 Spatially Constrained Attention

全都市トークンを見ると計算量が大きく、また空間的に不自然な候補も含まれる。そのため、候補都市要素を近傍や到達可能性に基づいて制限する。

```math
\text{Attn}(Q, K) \cdot \mathbf{1}(\text{dist}(cell_i, cell_j) < r)
```

または、移動手段に応じた到達圏でmaskを構築する。

```text
WALK: within 1km
BIKE: within 3km
TRAIN: reachable stations within 30min
CAR: reachable road network within 30min
```

### 6.4 Hierarchical Attention

都市は階層構造を持つため、coarse-to-fineに目的地を生成する。

```text
Step 1: choose district
Step 2: choose cell within district
Step 3: choose POI or station within cell
```

数式的には以下のように分解できる。

```math
p(d) = p(\text{district}) \cdot p(\text{cell} \mid \text{district}) \cdot p(\text{poi} \mid \text{cell})
```

### 6.5 Activity-Conditioned Attention

活動タイプによって参照すべき都市要素は異なる。

```text
ACT_WORK → office, school, industrial area
ACT_EAT → restaurant, commercial area
ACT_SHOP → retail POI, shopping mall
ACT_LEISURE → park, entertainment, tourist spots
```

そのため、activity embeddingをqueryに加える。

```math
Q_m = h_m + e_{activity}
```

### 6.6 Dynamic Urban Memory

都市状態は人流によっても変化する。混雑、滞在人口、流入流出は次の人流生成に影響する。

```text
Generated mobility events
→ aggregate crowd state
→ update urban tokens
→ generate next events
```

これにより、GUFMは静的な条件付き生成モデルから、都市と人流が相互作用するworld modelに近づく。

---

## 7. Model Architecture

GUFMの基本構造は、Urban EncoderとMobility Decoderからなる。

```text
Urban Tokens
    ↓
Urban Encoder
    ↓
Structured Urban Memory
    ↓
Cross-Attention
    ↑
Mobility Decoder ← Person Tokens + Mobility History + Context Tokens
    ↓
Next Mobility Event
```

### 7.1 Urban Encoder

都市トークンを構造化メモリへ変換する。

入力:

```text
CELL_2203 LANDUSE_COMMERCIAL POI_OFFICE_HIGH RAIL_ACCESS_HIGH CROWD_HIGH
CELL_2205 LANDUSE_COMMERCIAL POI_FOOD_HIGH WALK_5MIN CROWD_MED
EDGE_CELL_2203_CELL_2205 WALK_5MIN
```

出力:

```text
urban memory embeddings
```

### 7.2 Mobility Decoder

過去の行動履歴を自己回帰的に処理し、次のイベントを生成する。

```text
history events → self-attention → cross-attention to urban memory → next event
```

### 7.3 Output Heads

次イベントは複数の要素からなるため、複数の出力headを用いる。

```text
activity head
destination head
mode head
departure time head
duration head
```

出力例:

```text
ACT_EAT
DEST_CELL_2205
MODE_WALK
START_12_05
DUR_30_60
```

---

## 8. Learning Data Design

1.2億人規模の擬似人流をGUFMに使う場合、重要なのは全軌跡をそのまま読むことではなく、都市行動の文法を抽出できる学習コーパスへ変換することである。

### 8.1 Three-Layer Data Structure

学習データは3層で整理する。

#### A. Urban Context

```text
cell / mesh / H3
land use
POI distribution
transport accessibility
crowd level
weather
event
disruption
```

#### B. Person and Household Context

```text
age_bin
sex
worker_type
household_type
home_cell
work_cell
school_cell
car_ownership
mobility_profile
```

#### C. Mobility Event Sequence

```text
activity
origin
destination
mode
start_time
end_time
duration
travel_time
```

---

## 9. Preprocessing Pipeline

### Stage 0: Raw Inputs

```text
Pseudo mobility trajectories
+ Census / household attributes
+ POI / land use / transport network
+ Weather / calendar / event logs
```

### Stage 1: Spatial-Temporal Standardization

```text
Raw lat-lon trajectories
→ map matching / stay-point detection
→ spatial discretization
→ time discretization
```

Recommended spatial units:

```text
500m mesh
H3 cell
station catchment
administrative area
```

Recommended time unit:

```text
15-minute time bin
```

### Stage 2: Semantic Event Extraction

```text
Stay segments → activity labeling
Move segments → mode labeling
```

Activity examples:

```text
HOME
WORK
SCHOOL
SHOP
EAT
LEISURE
HOSPITAL
BUSINESS
OTHER
```

Mode examples:

```text
WALK
BIKE
CAR
TRAIN
BUS
OTHER
```

### Stage 3: Urban Context Join

Each event is aligned with dynamic urban context.

```text
(person_id, day_id, step_id, cell_id, time_bin)
→ join urban_state_table
→ join person_table
→ join network candidates
```

### Stage 4: Sequence Construction

For each person-day:

```text
[PERSON TOKENS]
[URBAN CONTEXT TOKENS]
[HISTORY EVENT TOKENS]
→ [TARGET NEXT EVENT]
```

### Stage 5: Training Corpus Generation

Generate training samples for the following tasks:

```text
next activity prediction
next destination prediction
mode prediction
duration prediction
masked event completion
intervention-conditioned prediction
```

---

## 10. Recommended Data Schema

The training corpus can be stored as Parquet tables.

### 10.1 person.parquet

```text
person_id: int64
household_id: int64
pref_code: int16
city_code: int32

age_bin: int8
sex: int8
worker_type: int8
household_type: int8
car_ownership: int8
bike_ownership: int8

home_cell_id: int64
work_cell_id: int64
school_cell_id: int64

home_station_id: int32
work_station_id: int32

commute_distance_bin: int8
mobility_profile: int8
weight: float32
```

### 10.2 event.parquet

```text
person_id: int64
day_id: int32
step_id: int16

event_type: int8
activity_type: int8

orig_cell_id: int64
dest_cell_id: int64
orig_station_id: int32
dest_station_id: int32
poi_id: int64

mode: int8
route_type: int8

start_time_bin: int16
end_time_bin: int16
duration_bin: int8
travel_time_bin: int8
travel_distance_bin: int8

companion_flag: int8
is_recurrent: int8
is_weekend: int8
holiday_flag: int8
weather_code: int8
event_context_code: int8

prev_activity_type: int8
next_activity_type: int8
```

### 10.3 urban_state.parquet

```text
cell_id: int64
day_type: int8
time_bin: int16

pref_code: int16
city_code: int32
district_id: int32

landuse_major: int8
landuse_mix: int8
population_density_bin: int8
employment_density_bin: int8

poi_food_cnt_bin: int8
poi_retail_cnt_bin: int8
poi_office_cnt_bin: int8
poi_school_cnt_bin: int8
poi_hospital_cnt_bin: int8
poi_leisure_cnt_bin: int8
poi_total_cnt_bin: int8

rail_access_bin: int8
road_access_bin: int8
bus_access_bin: int8
parking_supply_bin: int8

crowd_level_bin: int8
inflow_bin: int8
outflow_bin: int8
stay_pop_bin: int8

weather_code: int8
rain_bin: int8
temperature_bin: int8

special_event_flag: int8
special_event_type: int8
disruption_flag: int8
```

### 10.4 edge.parquet

```text
src_cell_id: int64
dst_cell_id: int64
edge_type: int8
distance_bin: int8
travel_time_bin: int8
cost_bin: int8
transfer_penalty_bin: int8
capacity_bin: int8
is_adjacent: int8
is_same_station_area: int8
```

---

## 11. Training Sample Example

### Input

```text
<PERSON>
AGE_30_39 WORKER_OFFICE HH_WITH_CHILD HOME_CELL_1101 WORK_CELL_2203
</PERSON>

<URBAN_t>
CELL_2203 LANDUSE_COMMERCIAL POI_OFFICE_HIGH POI_FOOD_MED
RAIL_ACCESS_HIGH CROWD_HIGH TIME_12_00 WEEKDAY WEATHER_CLEAR
</URBAN_t>

<HISTORY>
ACT_START_HOME
ACT_END_HOME
DEPART MODE_WALK
BOARD MODE_TRAIN
ARRIVE DEST_CELL_2203
ACT_START_WORK
ACT_END_WORK
</HISTORY>
```

### Target

```text
ACT_START_EAT DEST_CELL_2205 MODE_WALK DUR_30_60
```

---

## 12. Training Objectives

GUFMでは、単純なnext-token predictionだけでは不十分である。以下の複数タスクを組み合わせる。

### 12.1 Next Event Prediction

```math
p(e_{t+1} \mid e_{\leq t}, U_t, z)
```

### 12.2 Next Destination Prediction

```math
p(d_{t+1} \mid e_{\leq t}, U_t, z)
```

### 12.3 Duration Prediction

```math
p(\tau_{t+1} \mid a_{t+1}, d_{t+1}, U_t, z)
```

### 12.4 Masked Event Completion

履歴の一部をmaskし、欠損イベントを復元する。

```text
ACT_HOME → [MASK] → ACT_WORK
```

### 12.5 Intervention-Conditioned Prediction

都市トークンを変更したとき、生成される人流がどのように変わるかを学習する。

```text
before: no station
after: STATION_NEW CONNECT_CELL_2203
target: changed mobility distribution
```

---

## 13. Sampling Strategy

1.2億人を均等にサンプリングすると、通勤などの多数派行動に偏る。そのため、以下のサンプリング戦略が必要である。

### 13.1 Demographic Balance

```text
age
sex
worker_type
household_type
region
```

### 13.2 Behavior Rarity

```text
detour
multi-stop trip
late-night movement
long-distance movement
many transfers
irregular destination
```

### 13.3 Urban Condition Rarity

```text
rain
holiday
event day
disaster
railway disruption
tourism peak
```

---

## 14. Evaluation Design

GUFMの評価は、単なる既存人流の再現ではなく、汎化、転移、介入応答を中心に設計する。

### 14.1 Reconstruction Evaluation

既知都市・既知期間での再現性能を測る。

```text
OD distribution similarity
activity distribution similarity
mode share accuracy
duration distribution similarity
trip distance distribution similarity
```

### 14.2 Generalization Evaluation

未知都市・未知地域での性能を測る。

```text
held-out city split
held-out district split
held-out station area split
```

### 14.3 Transfer Evaluation

条件を変えたときの性能を測る。

```text
weekday → weekend
sunny → rainy
metropolitan city → regional city
```

### 14.4 Intervention Evaluation

都市への介入に対する応答を測る。

```text
new station opening
road closure
rail disruption
large event
disaster scenario
POI addition / removal
```

### 14.5 Suggested Metrics

```text
OD KL divergence
Jensen-Shannon divergence
activity distribution error
mode share error
trip distance distribution error
destination ranking metrics
spatial flow correlation
counterfactual response consistency
```

---

## 15. Minimal Proof-of-Concept

最初のPoCでは、すべてを実装する必要はない。以下の最小構成から始めるのが現実的である。

### 15.1 Minimal Tables

#### person_min.parquet

```text
person_id
age_bin
worker_type
household_type
home_cell_id
work_cell_id
```

#### event_min.parquet

```text
person_id
day_id
step_id
activity_type
orig_cell_id
dest_cell_id
mode
start_time_bin
duration_bin
```

#### urban_min.parquet

```text
cell_id
time_bin
landuse_major
poi_food_cnt_bin
poi_office_cnt_bin
rail_access_bin
crowd_level_bin
weather_code
```

### 15.2 Minimal Model

```text
Person Encoder
Urban Encoder
Mobility Decoder
Cross-Attention
Multi-head Event Prediction
```

### 15.3 Minimal Task

```text
Given:
person attributes
current urban context
past event sequence

Predict:
next activity
next destination cell
next mode
next duration
```

---

## 16. Relationship to Existing Models

### MobGLM

MobGLM can be interpreted as modeling human mobility as a language-like sequence.

```text
mobility tokens → Transformer → next mobility token
```

It is useful for learning action sequence patterns, but urban structure is often implicit.

### MobilityGPT

MobilityGPT moves toward conditional mobility generation.

```text
urban/person features + mobility history → next mobility event
```

It is closer to GUFM, but may still treat urban information as features rather than structured urban memory.

### UrbanGPT

UrbanGPT can be seen as an interface layer for urban analysis using LLMs.

```text
urban data + natural language query → answer / analysis
```

It is useful as an interactive interface, but it is not necessarily a generative world model for mobility.

### GUFM

GUFM aims to model urban mobility as interaction between people and structured urban memory.

```text
Mobility tokens as query
Urban tokens as memory
Cross-attention as interaction
Next mobility event as output
```

---

## 17. Key Design Principles

### Principle 1: Urban as Memory

都市は固定特徴量ではなく、生成過程で参照される構造化メモリである。

### Principle 2: Mobility as Decision Events

人流は座標列ではなく、活動、目的地、時間、移動手段からなる意思決定イベント列である。

### Principle 3: Cross-Attention as Interaction

人流トークンが都市トークンを参照することで、都市条件付き生成を実現する。

### Principle 4: Intervention as Token Update

介入はモデル外の後処理ではなく、都市トークンの追加、削除、更新として扱う。

### Principle 5: Evaluation Beyond Reconstruction

GUFMの価値は、既存データの再現だけでなく、未知都市、未知条件、都市介入への応答で評価する。

---

## 18. One-Sentence Summary

GUFMとは、都市を構造化メモリ、人流を意思決定イベント列として表現し、人流トークンが都市トークンへcross-attentionすることで、汎化、転移、介入応答を可能にする都市生成基盤モデルである。
