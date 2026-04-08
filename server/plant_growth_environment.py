"""
Tomato Plant Growth Environment — Core Simulation Engine

Implements a physics-grounded simulation of Solanum lycopersicum (tomato) growth
for reinforcement learning research.

Scientific sources:
  [1] Olubanjo & Alade (2018) — substrate effects (rice husk, sawdust, soil)
  [2] Galaverni et al. (2025) — IoT irrigation study, IWUE, water stress physiology
  [3] Lanki et al. (2023)    — poultry manure rates vs yield
  [4] Sabarinath et al. (2020)— bio-inoculants + inorganic fertilizers, T8 treatment
  [5] Sun et al. (2021)      — real-time weight & TEP-based growth model
  [6] White et al. (2022)    — functional-structural plant modelling (GreenLab-inspired)

Environment dynamics (per weekly step):
  - Temperature & light sampled from stage-appropriate distributions
  - TEP (Thermal Effectiveness × PAR) accumulated daily
  - GDD (Growing Degree Days) accumulated daily
  - Soil moisture updated by evapotranspiration + irrigation
  - Nutrient level updated by plant uptake + fertilizer
  - Plant height, leaf count, stem girth grown via TEP × stress × substrate model
  - Fruit count & weight accumulated in fruiting stage
  - Blossom End Rot increases with water stress in fruiting stage
  - IWUE and marketable yield tracked throughout
"""

import math
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PlantGrowthAction, PlantGrowthObservation
except ImportError:
    from models import PlantGrowthAction, PlantGrowthObservation


# ────────────────────────────────────────────────────────────────────────────
# Task registry
# ────────────────────────────────────────────────────────────────────────────

TASKS = {
    "seedling_vigor": {
        "name": "Seedling Vigor (Easy)",
        "description": (
            "Grow the tallest, healthiest seedling possible over the first 30 days. "
            "Score is based on plant height and leaf count at day 30."
        ),
        "max_days": 30,
        "difficulty": "easy",
    },
    "maximize_yield": {
        "name": "Maximize Fruit Yield (Medium)",
        "description": (
            "Run a full 120-day season. Maximize total marketable fruit yield (t/ha). "
            "Balance water and nutrients across all three growth stages."
        ),
        "max_days": 120,
        "difficulty": "medium",
    },
    "efficient_farmer": {
        "name": "Efficient Farmer (Hard)",
        "description": (
            "Maximize Irrigation Water Use Efficiency (IWUE = marketable kg / m³ water) "
            "while keeping marketable yield fraction >70% and BER <10%. "
            "Agent must master deficit irrigation and nutrient timing."
        ),
        "max_days": 120,
        "difficulty": "hard",
    },
}

# ────────────────────────────────────────────────────────────────────────────
# Substrate parameters — from Olubanjo & Alade (2018), Tables 1-4
# ────────────────────────────────────────────────────────────────────────────

SUBSTRATE_PARAMS = {
    "rice_husk": {
        # Growth multipliers relative to soil baseline
        "height_mult": 42.75 / 27.80,   # 1.538
        "leaf_mult":   120.60 / 67.40,  # 1.789
        "girth_mult":  0.5677 / 0.4155, # 1.366
        "yield_mult":  410.6  / 368.2,  # 1.115
        # Physical properties
        "water_retention": 0.76,  # WHC (%)
        "nutrient_efficiency": 1.35,
        "ph": 5.5,
        "organic_matter_g_kg": 12.2,
    },
    "sawdust": {
        "height_mult": 33.01 / 27.80,
        "leaf_mult":   93.50 / 67.40,
        "girth_mult":  0.4068 / 0.4155,
        "yield_mult":  406.0  / 368.2,
        "water_retention": 0.54,
        "nutrient_efficiency": 1.18,
        "ph": 6.1,
        "organic_matter_g_kg": 9.53,
    },
    "soil": {
        "height_mult": 1.0,
        "leaf_mult":   1.0,
        "girth_mult":  1.0,
        "yield_mult":  1.0,
        "water_retention": 0.14,
        "nutrient_efficiency": 1.0,
        "ph": 5.3,
        "organic_matter_g_kg": 2.98,
    },
}

# ────────────────────────────────────────────────────────────────────────────
# Fertilizer parameters — from Lanki et al. (2023) & Sabarinath et al. (2020)
# ────────────────────────────────────────────────────────────────────────────

FERTILIZER_PARAMS = {
    "none":      {"nutrient_release": 0.00, "growth_boost": 0.00, "cost_per_week": 0.0},
    "inorganic": {"nutrient_release": 0.06, "growth_boost": 0.85, "cost_per_week": 3.0},
    "organic":   {"nutrient_release": 0.05, "growth_boost": 0.90, "cost_per_week": 2.5},
    "combined":  {"nutrient_release": 0.07, "growth_boost": 1.00, "cost_per_week": 4.5},  # T8 best
}

# ────────────────────────────────────────────────────────────────────────────
# Irrigation reference — from Galaverni et al. (2025)
# Full season I100 = 3244.85 m3/ha over ~17 weeks
# ────────────────────────────────────────────────────────────────────────────

WEEKLY_IRRIFRAME_M3_HA = 3244.85 / 17.0   # ≈ 191 m3/ha/week at 100%

# ────────────────────────────────────────────────────────────────────────────
# Tomato temperature cardinal points — Sun et al. (2021), Table 1
# ────────────────────────────────────────────────────────────────────────────

T_BASE   = 10.0   # °C  minimum for growth
T_OPT    = 25.0   # °C  optimal
T_MAX    = 32.0   # °C  upper cutoff
T_LO     = 22.0   # °C  lower optimal (NHH curve)
T_UO     = 26.0   # °C  upper optimal (NHH curve)

# ────────────────────────────────────────────────────────────────────────────
# Reference marketable yield targets — Galaverni et al. (2025), Table 2
# ────────────────────────────────────────────────────────────────────────────

YIELD_I100_T_HA = 99.27    # I100 marketable yield
YIELD_I60_T_HA  = 70.23    # I60  marketable yield (60% irrigation)
YIELD_I30_T_HA  = 41.00    # I30  marketable yield

# ────────────────────────────────────────────────────────────────────────────
# Environment
# ────────────────────────────────────────────────────────────────────────────

class PlantGrowthEnvironment(Environment):
    """
    OpenEnv environment simulating tomato (Solanum lycopersicum) growth.

    Three tasks of escalating difficulty let agents learn:
      1. Basic irrigation/nutrition for seedling health (easy)
      2. Full-season yield maximization (medium)
      3. Water-efficient farming with marketable quality constraints (hard)

    State transitions follow the TEP-based model of Sun et al. (2021),
    calibrated to the empirical yield and stress data of Galaverni et al. (2025).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ── Days per action step ───────────────────────────────────────────────
    STEP_DAYS = 7

    def __init__(self, task_id: str = "maximize_yield"):
        if task_id not in TASKS:
            task_id = "maximize_yield"
        self._task_id   = task_id
        self._task_cfg  = TASKS[task_id]
        self._max_days  = self._task_cfg["max_days"]
        self._state     = State(episode_id=str(uuid4()), step_count=0)
        self._obs: PlantGrowthObservation | None = None
        self._substrate_locked = False

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(self) -> PlantGrowthObservation:
        """Reset environment to transplanting day (day 0)."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._substrate_locked = False

        self._obs = PlantGrowthObservation(
            day=0,
            growth_stage="seedling",
            # Transplant-size seedling (3-leaf stage, from Sabarinath et al.)
            plant_height_cm=5.0,
            leaf_count=3,
            stem_girth_cm=0.10,
            fresh_weight_g=50.0,
            fruit_count=0,
            fruit_weight_g=0.0,
            marketable_yield_t_ha=0.0,
            blossom_end_rot_fraction=0.0,
            soil_moisture=0.60,
            temperature_c=22.0,
            light_intensity_klux=15.0,
            nutrient_level=0.35,
            water_stress_index=0.0,
            chlorophyll_spad=55.0,
            accumulated_gdd=0.0,
            accumulated_tep=0.0,
            total_water_applied_m3_ha=0.0,
            iwue_kg_m3=0.0,
            substrate="rice_husk",
            done=False,
            reward=0.0,
        )
        return self._obs

    # ── Step ───────────────────────────────────────────────────────────────

    def step(self, action: PlantGrowthAction) -> PlantGrowthObservation:
        """Advance the simulation by one week (7 days) given the agent's action."""
        assert self._obs is not None, "Call reset() before step()"
        self._state.step_count += 1

        # Lock substrate on the first step of the episode
        if not self._substrate_locked:
            self._obs = self._obs.model_copy(
                update={"substrate": action.substrate_type}
            )
            self._substrate_locked = True

        new_obs, reward = self._simulate_week(action, self._obs)
        new_obs = new_obs.model_copy(update={"reward": reward})
        self._obs = new_obs
        return self._obs

    # ── State ──────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    # ──────────────────────────────────────────────────────────────────────
    # SIMULATION CORE
    # ──────────────────────────────────────────────────────────────────────

    def _simulate_week(
        self,
        action: PlantGrowthAction,
        obs: PlantGrowthObservation,
    ) -> tuple[PlantGrowthObservation, float]:
        """Run STEP_DAYS of daily simulation and return the new state + reward."""
        sub = SUBSTRATE_PARAMS[obs.substrate]
        fert = FERTILIZER_PARAMS[action.fertilizer_type]

        accumulated_water = 0.0  # m3/ha applied this week

        for _ in range(self.STEP_DAYS):
            if obs.day >= self._max_days:
                break

            # ── Environmental sample ───────────────────────────────────────
            temp, light = self._sample_environment(obs.day)

            # ── Compute daily TEP and GDD ──────────────────────────────────
            daily_tep = self._calc_tep(temp, light)
            daily_gdd = self._calc_gdd(temp)

            # ── Update soil moisture ───────────────────────────────────────
            et_daily = 0.018 + 0.004 * (light / 10.0)  # evapotranspiration
            water_added = action.irrigation_fraction * (WEEKLY_IRRIFRAME_M3_HA / 7) / 1000  # normalised 0-1 scale
            soil_moisture = max(
                0.0,
                min(1.0, obs.soil_moisture - et_daily + water_added * sub["water_retention"]),
            )
            accumulated_water += action.irrigation_fraction * (WEEKLY_IRRIFRAME_M3_HA / 7)

            # ── Update nutrient level ──────────────────────────────────────
            uptake_rate = 0.008 * max(0.1, obs.plant_height_cm / 30)
            release_rate = action.fertilizer_amount * fert["nutrient_release"] * sub["nutrient_efficiency"]
            nutrient_level = max(0.0, min(1.0, obs.nutrient_level - uptake_rate + release_rate))

            # ── Stress indices ─────────────────────────────────────────────
            water_stress = self._calc_water_stress(soil_moisture, action.irrigation_fraction)
            nutrient_effect = self._calc_nutrient_effect(nutrient_level, fert["growth_boost"])

            # ── SPAD (chlorophyll) dynamics ────────────────────────────────
            # Increases slightly mid-season, declines under stress (Galaverni et al.)
            spad_delta = -0.05 * water_stress - 0.02 * max(0, 0.3 - nutrient_level)
            chlorophyll_spad = max(30.0, min(70.0, obs.chlorophyll_spad + spad_delta))

            # ── Growth efficiency composite ────────────────────────────────
            tep_factor   = min(2.0, 1.0 + daily_tep * 8.0)
            stress_factor = (1.0 - water_stress * 0.75) * nutrient_effect

            # ── Stage-specific base growth rates ───────────────────────────
            stage = self._growth_stage(obs.day)
            # Rates calibrated so that at optimal conditions over 120 days
            # we reach ~42 cm height, ~120 leaves (rice_husk baseline from [1])
            if stage == "seedling":
                r_height = 0.65  # cm/day
                r_leaves = 0.28
                r_girth  = 0.0028
            elif stage == "flowering":
                r_height = 0.70
                r_leaves = 0.45
                r_girth  = 0.0035
            else:  # fruiting
                r_height = 0.40  # growth slows as assimilates shift to fruit
                r_leaves = 0.15
                r_girth  = 0.0018

            # Apply all modifiers
            dh = r_height * sub["height_mult"] * stress_factor * tep_factor
            dl = r_leaves * sub["leaf_mult"]   * stress_factor * tep_factor
            dg = r_girth  * sub["girth_mult"]  * stress_factor

            new_height = min(140.0, obs.plant_height_cm + dh)
            new_leaves = min(200,   obs.leaf_count + dl)
            new_girth  = min(2.0,   obs.stem_girth_cm + dg)

            # ── Fruit development ──────────────────────────────────────────
            new_fruit_count  = obs.fruit_count
            new_fruit_weight = obs.fruit_weight_g
            new_ber          = obs.blossom_end_rot_fraction

            if stage == "fruiting":
                # Fruit set rate from paper range 14-38 fruits / plant over ~70 days
                fruit_set_rate = 0.40 * stress_factor * sub["yield_mult"]
                new_fruit_count = min(40, obs.fruit_count + fruit_set_rate)

                # Weight accumulation per fruit per day (max ~420 g / plant)
                if new_fruit_count > 0:
                    weight_per_fruit_rate = 4.5 * stress_factor
                    new_fruit_weight = min(
                        new_fruit_count * 430,
                        obs.fruit_weight_g + weight_per_fruit_rate * new_fruit_count * 0.08,
                    )

                # BER increases with water stress (Ca deficiency, from [2])
                # I100 = 1.94%, I60 = 6.29%, I30 = 15.83%
                ber_increase = 0.002 * water_stress ** 2
                new_ber = min(1.0, obs.blossom_end_rot_fraction + ber_increase)

            # ── Marketable yield estimate ──────────────────────────────────
            # Scale plant-level to ha level (35,556 plants/ha @ 75×75 cm spacing)
            PLANTS_PER_HA = 35_556
            # Marketable = fruit_weight * (1 - BER fraction) * plants/ha / 1e6 (to t)
            marketable_yield_t_ha = (
                new_fruit_weight
                * (1.0 - new_ber)
                * PLANTS_PER_HA
                / 1_000_000
            )

            # ── Water-use efficiency ───────────────────────────────────────
            new_total_water = obs.total_water_applied_m3_ha + accumulated_water / self.STEP_DAYS
            iwue = 0.0
            if new_total_water > 0:
                iwue = (marketable_yield_t_ha * 1000) / new_total_water  # kg/m3

            # ── Fresh weight proxy ─────────────────────────────────────────
            fresh_weight = new_height * 4.5 + new_leaves * 1.8 + new_fruit_weight

            # ── Advance day ────────────────────────────────────────────────
            new_day = obs.day + 1
            obs = PlantGrowthObservation(
                day=new_day,
                growth_stage=self._growth_stage(new_day),
                plant_height_cm=new_height,
                leaf_count=int(new_leaves),
                stem_girth_cm=new_girth,
                fresh_weight_g=fresh_weight,
                fruit_count=int(new_fruit_count),
                fruit_weight_g=new_fruit_weight,
                marketable_yield_t_ha=marketable_yield_t_ha,
                blossom_end_rot_fraction=new_ber,
                soil_moisture=soil_moisture,
                temperature_c=temp,
                light_intensity_klux=light,
                nutrient_level=nutrient_level,
                water_stress_index=water_stress,
                chlorophyll_spad=chlorophyll_spad,
                accumulated_gdd=obs.accumulated_gdd + daily_gdd,
                accumulated_tep=obs.accumulated_tep + daily_tep,
                total_water_applied_m3_ha=new_total_water,
                iwue_kg_m3=iwue,
                substrate=obs.substrate,
                done=(new_day >= self._max_days),
                reward=0.0,
            )

        reward = self._compute_reward(obs, action)
        return obs, reward

    # ──────────────────────────────────────────────────────────────────────
    # TASK GRADERS  (all return float in [0, 1])
    # ──────────────────────────────────────────────────────────────────────

    def grade_seedling_vigor(self, obs: PlantGrowthObservation) -> float:
        """
        Easy task grader.
        Score = 0.6 × (height / 35 cm) + 0.3 × (leaves / 50) + 0.1 × (1 - stress)
        Perfect score needs height ≥ 35 cm and 50 leaves at day 30.
        Rice husk + good irrigation can reliably hit this.
        """
        height_score  = min(1.0, obs.plant_height_cm / 35.0)
        leaf_score    = min(1.0, obs.leaf_count / 50.0)
        stress_score  = 1.0 - obs.water_stress_index
        return float(0.60 * height_score + 0.30 * leaf_score + 0.10 * stress_score)

    def grade_maximize_yield(self, obs: PlantGrowthObservation) -> float:
        """
        Medium task grader.
        Score = marketable yield / I100 reference yield (99.27 t/ha).
        Perfect agent approaches or exceeds the I100 reference from [2].
        """
        score = min(1.0, obs.marketable_yield_t_ha / YIELD_I100_T_HA)
        return float(score)

    def grade_efficient_farmer(self, obs: PlantGrowthObservation) -> float:
        """
        Hard task grader.
        Maximise IWUE while keeping marketable fraction ≥ 70% and BER ≤ 10%.

        Score breakdown:
          - 50% IWUE component: target 30 kg/m³ (excellent)
          - 30% marketable fraction (fruit not lost to BER or stress)
          - 20% penalty-free quality (BER < 10% gives full marks here)

        A naive max-irrigation agent scores ~0.60 (low IWUE).
        Optimal deficit irrigation (60%) should reach ~0.85+.
        """
        # IWUE component — I60 achieves ~29 kg/m3, perfect ~35
        max_iwue = 35.0
        iwue_score = min(1.0, obs.iwue_kg_m3 / max_iwue)

        # Marketable fraction — how much of grown fruit is sellable
        if obs.fruit_weight_g > 0:
            marketable_frac = 1.0 - obs.blossom_end_rot_fraction
        else:
            marketable_frac = 0.0
        mkt_score = min(1.0, marketable_frac / 0.82)  # I60 achieves 81.8%

        # BER quality penalty — < 10% BER gives full score
        ber_score = max(0.0, 1.0 - obs.blossom_end_rot_fraction / 0.10)
        ber_score = min(1.0, ber_score)

        score = 0.50 * iwue_score + 0.30 * mkt_score + 0.20 * ber_score
        return float(score)

    def grade(self, obs: PlantGrowthObservation | None = None) -> float:
        """Grade the current (or provided) observation for the active task."""
        if obs is None:
            obs = self._obs
        if obs is None:
            return 0.0
        if self._task_id == "seedling_vigor":
            return self.grade_seedling_vigor(obs)
        elif self._task_id == "maximize_yield":
            return self.grade_maximize_yield(obs)
        else:
            return self.grade_efficient_farmer(obs)

    # ──────────────────────────────────────────────────────────────────────
    # REWARD SHAPING  (partial-progress signal throughout episode)
    # ──────────────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        obs: PlantGrowthObservation,
        action: PlantGrowthAction,
    ) -> float:
        """
        Shaped reward providing signal at every step, not just episode end.
        Penalises clearly harmful behaviour (severe stress, over-watering).
        """
        stage = obs.growth_stage
        r = 0.0

        if self._task_id == "seedling_vigor":
            # Reward height progress relative to expected trajectory
            expected = 0.85 * obs.day  # cm at this day under good conditions
            r  = 0.50 * min(1.0, obs.plant_height_cm / max(1.0, expected))
            r += 0.25 * min(1.0, obs.leaf_count / max(1, obs.day * 0.4))
            r += 0.25 * (1.0 - obs.water_stress_index)

        elif self._task_id == "maximize_yield":
            if stage == "seedling":
                r  = 0.30 * min(1.0, obs.plant_height_cm / 25.0)
                r += 0.20 * (1.0 - obs.water_stress_index)
            elif stage == "flowering":
                r  = 0.30 * min(1.0, obs.leaf_count / 70.0)
                r += 0.20 * obs.nutrient_level
            else:
                r  = 0.60 * min(1.0, obs.marketable_yield_t_ha / YIELD_I100_T_HA)
                r += 0.10 * (1.0 - obs.blossom_end_rot_fraction)

        else:  # efficient_farmer
            if stage == "fruiting":
                r  = 0.40 * min(1.0, obs.iwue_kg_m3 / 30.0)
                r += 0.30 * (1.0 - obs.blossom_end_rot_fraction)
                r += 0.20 * min(1.0, obs.marketable_yield_t_ha / YIELD_I60_T_HA)
            else:
                r  = 0.50 * (1.0 - obs.water_stress_index)
                r += 0.30 * obs.nutrient_level

        # ── Penalties for harmful behaviour ────────────────────────────────
        # Severe water stress (like I30 regime)
        if obs.water_stress_index > 0.65:
            r -= 0.25

        # Nutrient crash
        if obs.nutrient_level < 0.15:
            r -= 0.15

        # Chronic over-watering wastes resources and hurts IWUE task
        if action.irrigation_fraction > 0.95 and self._task_id == "efficient_farmer":
            r -= 0.05

        return float(max(-1.0, min(1.0, r)))

    # ──────────────────────────────────────────────────────────────────────
    # PHYSICS HELPERS
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _growth_stage(day: int) -> str:
        if day <= 30:
            return "seedling"
        elif day <= 50:
            return "flowering"
        return "fruiting"

    @staticmethod
    def _sample_environment(day: int) -> tuple[float, float]:
        """
        Sample realistic temperature and light intensity for a given day.
        Based on seasonal patterns observed in [2] (Parma, Italy field trial)
        and [5] (Chinese solar greenhouse).
        """
        stage = PlantGrowthEnvironment._growth_stage(day)
        if stage == "seedling":
            mu_t, mu_l = 22.0, 18.0
        elif stage == "flowering":
            mu_t, mu_l = 18.5, 14.0
        else:
            mu_t, mu_l = 15.0, 11.0

        temp  = max(5.0,  min(38.0, random.gauss(mu_t, 3.0)))
        light = max(0.5,  min(40.0, random.gauss(mu_l, 4.0)))
        return temp, light

    @staticmethod
    def _calc_tep(temp: float, light_klux: float) -> float:
        """
        Daily TEP (MJ/m²) — Sun et al. (2021) Equations 3-7.
        TEP = RTE × PAR × seconds_of_daylight / 1e6
        """
        mu = 5.07e-4  # lux-to-PAR conversion factor
        PAR_Jm2h = light_klux * 1000 * mu

        if temp <= T_BASE:
            RTE = 0.0
        elif temp < T_OPT:
            RTE = (temp - T_BASE) / (T_OPT - T_BASE)
        elif temp <= T_MAX:
            RTE = (T_MAX - temp) / (T_MAX - T_OPT)
        else:
            RTE = 0.0

        HTEP = RTE * PAR_Jm2h * 3600  # J/m2 over one hour
        DTEP = HTEP * 10 / 1_000_000   # ~10 daylight hours → MJ/m2/day
        return max(0.0, DTEP)

    @staticmethod
    def _calc_gdd(temp: float) -> float:
        """Daily GDD — Galaverni et al. (2025) Equation 1."""
        if temp <= T_BASE:
            return 0.0
        elif temp >= T_MAX:
            return float(T_MAX - T_BASE)
        return float(temp - T_BASE)

    @staticmethod
    def _calc_water_stress(soil_moisture: float, irr_frac: float) -> float:
        """
        Water stress index [0, 1].
        Calibrated so that irr_frac ≈ 1.0 → very low stress (I100 ≈ 1.94% BER),
        irr_frac ≈ 0.6 → mild stress (I60 ≈ 6.29% BER),
        irr_frac ≈ 0.3 → severe stress (I30 ≈ 15.83% BER).
        """
        # Reference soil moisture at each irrigation level
        ref = 0.35 + 0.50 * irr_frac  # 0.35 to 0.85
        deficit = max(0.0, ref - soil_moisture)
        stress = (deficit / ref) * (1.0 - 0.4 * irr_frac)
        return float(min(1.0, stress))

    @staticmethod
    def _calc_nutrient_effect(nutrient_level: float, fert_boost: float) -> float:
        """
        Nutrient availability effect on growth [0.3, 1.0].
        Combined bio+inorganic (T8) gave 96.45 cm height vs 
        control that gave ~30 cm — Sabarinath et al. (2020).
        """
        effective = min(1.0, nutrient_level + fert_boost * 0.15)
        if effective < 0.15:
            return 0.30
        elif effective < 0.35:
            return 0.30 + 1.75 * (effective - 0.15)
        elif effective < 0.80:
            return 1.00
        else:
            # Luxury consumption / slight toxicity above optimum
            return max(0.85, 1.0 - 0.5 * (effective - 0.80))