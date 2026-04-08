"""Plant Growth Environment — WebSocket Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import PlantGrowthAction, PlantGrowthObservation


class PlantGrowthEnv(
    EnvClient[PlantGrowthAction, PlantGrowthObservation, State]
):
    """
    Client for the Tomato Plant Growth Environment.

    Maintains a persistent WebSocket connection for low-latency multi-step episodes.

    Example:
        >>> with PlantGrowthEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     obs = result.observation
        ...     print(f"Day {obs.day}: height={obs.plant_height_cm:.1f} cm")
        ...
        ...     action = PlantGrowthAction(
        ...         irrigation_fraction=0.6,
        ...         fertilizer_amount=0.7,
        ...         fertilizer_type="combined",
        ...         substrate_type="rice_husk",
        ...     )
        ...     result = env.step(action)
        ...     obs = result.observation
        ...     print(f"Day {obs.day}: height={obs.plant_height_cm:.1f} cm, reward={result.reward:.3f}")
    """

    def _step_payload(self, action: PlantGrowthAction) -> Dict:
        return {
            "irrigation_fraction": action.irrigation_fraction,
            "fertilizer_amount":   action.fertilizer_amount,
            "fertilizer_type":     action.fertilizer_type,
            "substrate_type":      action.substrate_type,
        }

    def _parse_result(self, payload: Dict) -> StepResult[PlantGrowthObservation]:
        obs_data = payload.get("observation", {})
        observation = PlantGrowthObservation(
            day=obs_data.get("day", 0),
            growth_stage=obs_data.get("growth_stage", "seedling"),
            plant_height_cm=obs_data.get("plant_height_cm", 5.0),
            leaf_count=obs_data.get("leaf_count", 3),
            stem_girth_cm=obs_data.get("stem_girth_cm", 0.1),
            fresh_weight_g=obs_data.get("fresh_weight_g", 50.0),
            fruit_count=obs_data.get("fruit_count", 0),
            fruit_weight_g=obs_data.get("fruit_weight_g", 0.0),
            marketable_yield_t_ha=obs_data.get("marketable_yield_t_ha", 0.0),
            blossom_end_rot_fraction=obs_data.get("blossom_end_rot_fraction", 0.0),
            soil_moisture=obs_data.get("soil_moisture", 0.6),
            temperature_c=obs_data.get("temperature_c", 22.0),
            light_intensity_klux=obs_data.get("light_intensity_klux", 15.0),
            nutrient_level=obs_data.get("nutrient_level", 0.5),
            water_stress_index=obs_data.get("water_stress_index", 0.0),
            chlorophyll_spad=obs_data.get("chlorophyll_spad", 55.0),
            accumulated_gdd=obs_data.get("accumulated_gdd", 0.0),
            accumulated_tep=obs_data.get("accumulated_tep", 0.0),
            total_water_applied_m3_ha=obs_data.get("total_water_applied_m3_ha", 0.0),
            iwue_kg_m3=obs_data.get("iwue_kg_m3", 0.0),
            substrate=obs_data.get("substrate", "rice_husk"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )