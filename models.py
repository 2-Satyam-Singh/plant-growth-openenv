"""
Data models for the Tomato Plant Growth Environment.

Grounded in real agronomic research:
- Olubanjo & Alade (2018) - substrate effects (rice husk, sawdust, soil)
- Galaverni et al. (2025) - IoT irrigation study (100%, 60%, 30% regimes)
- Lanki et al. (2023) - organic vs conventional fertilization
- Sabarinath et al. (2020) - bio-inoculants + inorganic fertilizers
- Sun et al. (2021) - TEP-based growth model
"""

from typing import Literal
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class PlantGrowthAction(Action):
    """
    Weekly management action for the tomato farm.
    The agent makes decisions every 7 days about water and nutrients.
    """

    irrigation_fraction: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the recommended weekly irrigation to apply. "
            "0.0 = no water, 1.0 = full Irriframe recommendation (~191 m3/ha/week). "
            "Research shows 0.6 is optimal for marketable yield (Galaverni et al., 2025)."
        ),
    )

    fertilizer_amount: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the maximum recommended fertilizer dose to apply. "
            "0.0 = none, 1.0 = max (11 t/ha poultry manure equivalent or full NPK). "
            "Higher amounts increase yield but also cost."
        ),
    )

    fertilizer_type: Literal["none", "inorganic", "organic", "combined"] = Field(
        default="combined",
        description=(
            "Type of fertilizer applied this week. "
            "'none' = no fertilizer, "
            "'inorganic' = NPK (urea + superphosphate + muriate of potash), "
            "'organic' = poultry manure, "
            "'combined' = bioinoculants + inorganic NPK (best in Sabarinath et al., 2020). "
            "Combined type is most effective but most expensive."
        ),
    )

    substrate_type: Literal["soil", "sawdust", "rice_husk"] = Field(
        default="rice_husk",
        description=(
            "Growing substrate. Only effective on the FIRST action of an episode. "
            "rice_husk: best growth (height 42.75 cm, yield 410.6 g/plant), "
            "sawdust: intermediate, "
            "soil: conventional baseline (Olubanjo & Alade, 2018)."
        ),
    )


class PlantGrowthObservation(Observation):
    """
    Full state observation of the tomato plant and its environment.
    Updated every step (7 days). All values reflect real agronomic ranges.
    """

    # ── Time ──────────────────────────────────────────────────────────────────
    day: int = Field(default=0, description="Current day of the episode (0-120).")
    growth_stage: str = Field(
        default="seedling",
        description="Phenological stage: 'seedling' (0-30d), 'flowering' (31-50d), 'fruiting' (51-120d).",
    )

    # ── Plant morphology ───────────────────────────────────────────────────────
    plant_height_cm: float = Field(
        default=5.0,
        description="Plant height in cm. Target: 40-100 cm depending on substrate and management.",
    )
    leaf_count: int = Field(
        default=3,
        description="Number of leaves. Rice husk substrate reaches ~120 leaves at maturity.",
    )
    stem_girth_cm: float = Field(
        default=0.10,
        description="Stem girth in cm. Indicator of structural health.",
    )
    fresh_weight_g: float = Field(
        default=50.0,
        description="Total above-ground fresh weight in grams (stems + leaves + fruit).",
    )

    # ── Yield metrics ─────────────────────────────────────────────────────────
    fruit_count: int = Field(
        default=0,
        description="Number of fruits per plant. Research range: 14-38 fruits/plant.",
    )
    fruit_weight_g: float = Field(
        default=0.0,
        description="Total fruit fresh weight per plant in grams. Research range: 368-410 g/plant.",
    )
    marketable_yield_t_ha: float = Field(
        default=0.0,
        description=(
            "Estimated marketable yield in tonnes/ha. "
            "I60 regime gave 70.23 t/ha vs I100's 99.27 t/ha (Galaverni et al., 2025)."
        ),
    )
    blossom_end_rot_fraction: float = Field(
        default=0.0,
        description=(
            "Fraction of fruits with Blossom End Rot (calcium deficiency from water stress). "
            "I30 regime: 15.83%, I60: 6.29%, I100: 1.94% (Galaverni et al., 2025)."
        ),
    )

    # ── Soil and environment ───────────────────────────────────────────────────
    soil_moisture: float = Field(
        default=0.6,
        description="Volumetric soil moisture (0.0 = bone dry, 1.0 = saturated).",
    )
    temperature_c: float = Field(
        default=22.0,
        description="Daily average temperature in Celsius. Optimal range: 22-26°C.",
    )
    light_intensity_klux: float = Field(
        default=15.0,
        description="Light intensity in klux. Affects TEP and photosynthesis.",
    )
    nutrient_level: float = Field(
        default=0.5,
        description=(
            "Composite soil nutrient level (0.0 = depleted, 1.0 = optimal). "
            "Represents available N, P, K for plant uptake."
        ),
    )

    # ── Stress indicators ──────────────────────────────────────────────────────
    water_stress_index: float = Field(
        default=0.0,
        description=(
            "Water stress index (0.0 = no stress, 1.0 = lethal stress). "
            "Drives SPAD decline and BER formation per Galaverni et al. (2025)."
        ),
    )
    chlorophyll_spad: float = Field(
        default=55.0,
        description=(
            "SPAD chlorophyll index. Declines under water stress. "
            "I100: started 53.6, I30: dropped to 45.8 by late season (Galaverni et al., 2025)."
        ),
    )

    # ── Accumulated growth indicators ──────────────────────────────────────────
    accumulated_gdd: float = Field(
        default=0.0,
        description=(
            "Accumulated Growing Degree Days (base 10°C, cutoff 32°C). "
            "Fruit set occurs around 109 GDD, full flowering ~300 GDD (Galaverni et al., 2025)."
        ),
    )
    accumulated_tep: float = Field(
        default=0.0,
        description=(
            "Accumulated Thermal Effectiveness × PAR product (MJ/m²). "
            "Used in the Sun et al. (2021) organ allocation model. "
            "Seedling end: ~72, Flowering end: ~161, Maturity: ~263."
        ),
    )

    # ── Water use efficiency ───────────────────────────────────────────────────
    total_water_applied_m3_ha: float = Field(
        default=0.0,
        description="Total irrigation water applied so far (m³/ha). I100 season total: 3244 m³/ha.",
    )
    iwue_kg_m3: float = Field(
        default=0.0,
        description=(
            "Irrigation Water Use Efficiency = marketable yield (kg/ha) / water (m³/ha). "
            "Target: >25 kg/m³. Research range: 23-31 kg/m³ (Galaverni et al., 2025)."
        ),
    )

    # ── Growing substrate ──────────────────────────────────────────────────────
    substrate: str = Field(
        default="rice_husk",
        description="Active growing substrate selected at episode start.",
    )

    # ── OpenEnv standard fields ────────────────────────────────────────────────
    done: bool = Field(default=False, description="True when the episode (120-day season) is complete.")
    reward: float = Field(default=0.0, description="Reward from the most recent step.")