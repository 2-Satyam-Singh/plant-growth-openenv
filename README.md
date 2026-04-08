---
title: Tomato Plant Growth Environment
emoji: 🍅
colorFrom: green
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - agriculture
  - plant-growth
  - reinforcement-learning
---

# 🍅 Tomato Plant Growth — OpenEnv Environment

A scientifically grounded simulation of **Solanum lycopersicum** (tomato) cultivation
for training and evaluating RL agents on real-world agricultural decision-making.

---

## Overview

The agent manages a 120-day tomato growing season, making **weekly decisions** about:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `irrigation_fraction` | 0.0 – 1.0 | Fraction of recommended water to apply |
| `fertilizer_amount` | 0.0 – 1.0 | Fraction of max fertilizer dose |
| `fertilizer_type` | categorical | none / inorganic / organic / combined |
| `substrate_type` | categorical | soil / sawdust / rice_husk (set at episode start) |

The environment is calibrated to **real agronomic data** from peer-reviewed studies:

| Source | Used For |
|--------|----------|
| Galaverni et al. (2025) | Irrigation regimes (I100/I60/I30), IWUE, BER rates |
| Olubanjo & Alade (2018) | Substrate growth multipliers (rice husk, sawdust, soil) |
| Sun et al. (2021) | TEP-based growth model, organ allocation |
| Lanki et al. (2023) | Poultry manure rates vs yield |
| Sabarinath et al. (2020) | Combined bio+inorganic fertilizer effects |

---

## Three Tasks

### Task 1 — Seedling Vigor *(Easy)*
**Goal:** Grow the healthiest seedling in 30 days.

```
score = 0.60 × (height / 35 cm)
      + 0.30 × (leaf_count / 50)
      + 0.10 × (1 - water_stress)
```

A simple irrigation + fertilization policy can score ~0.70.
An optimal agent should reach ~0.95+.

### Task 2 — Maximize Yield *(Medium)*
**Goal:** Maximize marketable fruit yield (t/ha) over 120 days.

```
score = min(1.0, marketable_yield / 99.27 t/ha)
```

Reference yields from Galaverni et al. (2025):
- I100 (full irrigation): 99.27 t/ha ← perfect score reference
- I60 (60% irrigation):   70.23 t/ha ← achievable with good strategy
- I30 (30% irrigation):   41.00 t/ha ← poor strategy baseline

### Task 3 — Efficient Farmer *(Hard)*
**Goal:** Maximize Irrigation Water Use Efficiency while maintaining quality.

```
score = 0.50 × min(1.0, IWUE / 35 kg/m³)
      + 0.30 × marketable_fraction
      + 0.20 × max(0, 1 - BER / 0.10)
```

Constraints: BER < 10%, marketable fraction > 70%.
The I60 regime achieves IWUE ≈ 29 kg/m³ — a good target.

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `day` | int | Episode day (0–120) |
| `growth_stage` | str | seedling / flowering / fruiting |
| `plant_height_cm` | float | Plant height (cm) |
| `leaf_count` | int | Number of leaves |
| `stem_girth_cm` | float | Stem girth (cm) |
| `fruit_count` | int | Fruits per plant |
| `fruit_weight_g` | float | Total fruit weight (g/plant) |
| `marketable_yield_t_ha` | float | Estimated marketable yield (t/ha) |
| `blossom_end_rot_fraction` | float | BER fraction [0, 1] |
| `soil_moisture` | float | Soil moisture [0, 1] |
| `temperature_c` | float | Daily temperature (°C) |
| `light_intensity_klux` | float | Light intensity (klux) |
| `nutrient_level` | float | Composite nutrient level [0, 1] |
| `water_stress_index` | float | Water stress [0, 1] |
| `chlorophyll_spad` | float | SPAD index (proxy for leaf health) |
| `accumulated_gdd` | float | Growing Degree Days accumulated |
| `accumulated_tep` | float | TEP accumulated (MJ/m²) |
| `total_water_applied_m3_ha` | float | Total irrigation applied |
| `iwue_kg_m3` | float | Irrigation Water Use Efficiency |
| `substrate` | str | Active growing substrate |

---

## Simulation Physics

### TEP Model (Sun et al., 2021)
```
PAR = light_klux × 1000 × 5.07e-4   [J/m²/h]
RTE = (T - T_base) / (T_opt - T_base)     if T < T_opt
    = (T_max - T) / (T_max - T_opt)       if T < T_max
DTEP = RTE × PAR × 36000 × 10 / 1e6     [MJ/m²/day]
```

### GDD Model (Galaverni et al., 2025)
```
GDD = T_avg - T_base   if T_base < T_avg < T_cutoff
T_base = 10°C,  T_cutoff = 32°C
```

### Substrate Effects (Olubanjo & Alade, 2018)
| Substrate | Height × | Leaves × | Yield × |
|-----------|----------|----------|---------|
| rice_husk | 1.54 | 1.79 | 1.12 |
| sawdust   | 1.19 | 1.39 | 1.10 |
| soil      | 1.00 | 1.00 | 1.00 |

### Water Stress & BER
Stress drives Blossom End Rot (Ca deficiency). At I30 regime:
BER reaches 15.83% vs 1.94% at I100 (Galaverni et al., 2025).

---

## Quick Start

```python
from plant_growth import PlantGrowthAction, PlantGrowthEnv

with PlantGrowthEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    obs = result.observation

    for week in range(18):
        if result.done:
            break

        # Your policy here
        action = PlantGrowthAction(
            irrigation_fraction=0.6,     # 60% — optimal per research
            fertilizer_amount=0.7,
            fertilizer_type="combined",  # bio-inoculants + NPK — best
            substrate_type="rice_husk",  # best growth substrate
        )

        result = env.step(action)
        obs = result.observation
        print(
            f"Week {week+1}: Day {obs.day} | "
            f"Height {obs.plant_height_cm:.1f}cm | "
            f"Yield {obs.marketable_yield_t_ha:.2f} t/ha | "
            f"IWUE {obs.iwue_kg_m3:.1f} kg/m³"
        )
```

---

## Running Locally

```bash
# Install dependencies
uv sync

# Start server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Run baseline inference (all 3 tasks)
export HF_TOKEN=your_token
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py --task all --episodes 1
```

## Docker

```bash
docker build -t plant_growth-env:latest -f server/Dockerfile .
docker run -p 8000:8000 plant_growth-env:latest
```

---

## Baseline Scores (LLM Agent)

| Task | Difficulty | Expected Score | Notes |
|------|-----------|---------------|-------|
| seedling_vigor | Easy | ~0.65–0.80 | Height-focused, easy to learn |
| maximize_yield | Medium | ~0.50–0.70 | Requires stage-aware strategy |
| efficient_farmer | Hard | ~0.40–0.60 | Deficit irrigation is counter-intuitive |

A random agent scores ~0.2–0.3 across all tasks.

## Project Structure

```
plant_growth/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # PlantGrowthEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── plant_growth_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
