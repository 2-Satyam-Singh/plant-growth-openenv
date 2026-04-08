"""
inference.py — FINAL VERSION FOR HACKATHON
Fully async-safe + rule-based agent (no API key needed)
"""
import argparse
import asyncio
import json
import os
import sys

# Import the environment
try:
    from plant_growth.client import PlantGrowthEnv
    from plant_growth.models import PlantGrowthAction, PlantGrowthObservation
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from client import PlantGrowthEnv
    from models import PlantGrowthAction, PlantGrowthObservation


TASK_MAX_STEPS = {
    "seedling_vigor": 5,
    "maximize_yield": 18,
    "efficient_farmer": 18,
}

TASK_MAX_DAYS = {
    "seedling_vigor": 30,
    "maximize_yield": 120,
    "efficient_farmer": 120,
}


def rule_based_action(obs: PlantGrowthObservation, task_id: str) -> PlantGrowthAction:
    """Simple but strong policy based on the research papers."""
    stage = obs.growth_stage

    if task_id == "efficient_farmer":
        irr = 0.70 if stage == "seedling" else 0.65 if stage == "flowering" else 0.60
    elif task_id == "seedling_vigor":
        irr = 0.75
    else:
        irr = 0.80 if stage == "seedling" else 0.75 if stage == "flowering" else 0.70

    # Avoid over-watering
    if obs.soil_moisture > 0.85:
        irr = max(0.0, irr - 0.20)

    return PlantGrowthAction(
        irrigation_fraction=round(irr, 2),
        fertilizer_amount=0.70,
        fertilizer_type="combined",
        substrate_type="rice_husk",
    )


def grade(obs: PlantGrowthObservation, task_id: str) -> float:
    """Mirror the environment's graders."""
    if task_id == "seedling_vigor":
        return min(1.0, 0.60 * min(1.0, obs.plant_height_cm / 35.0) +
                   0.30 * min(1.0, obs.leaf_count / 50.0) +
                   0.10 * (1.0 - obs.water_stress_index))
    elif task_id == "maximize_yield":
        return min(1.0, obs.marketable_yield_t_ha / 99.27)
    else:  # efficient_farmer
        iwue_score = min(1.0, obs.iwue_kg_m3 / 35.0)
        mkt_score = min(1.0, (1.0 - obs.blossom_end_rot_fraction) / 0.82)
        ber_score = min(1.0, max(0.0, 1.0 - obs.blossom_end_rot_fraction / 0.10))
        return 0.50 * iwue_score + 0.30 * mkt_score + 0.20 * ber_score


async def run_episode(env, task_id: str, episode: int):
    max_steps = TASK_MAX_STEPS[task_id]
    print(f"\n{'='*70}")
    print(f"Task: {task_id} | Episode {episode+1} | Max steps: {max_steps}")
    print(f"{'='*70}")

    result = await env.reset()
    obs: PlantGrowthObservation = result.observation
    total_reward = 0.0

    for step in range(1, max_steps + 1):
        if result.done or obs.day >= TASK_MAX_DAYS[task_id]:
            print(f"→ Done at step {step}, day {obs.day}")
            break

        action = rule_based_action(obs, task_id)
        result = await env.step(action)          # ← await is required
        obs = result.observation
        reward = result.reward or 0.0
        total_reward += reward

        print(f"Step {step:2d} | Day {obs.day:3d} | Height {obs.plant_height_cm:5.1f}cm | "
              f"Yield {obs.marketable_yield_t_ha:5.2f} | Reward {reward:+.3f}")

    final_score = grade(obs, task_id)

    print(f"\nFinal Task Score: {final_score:.4f} (0.0–1.0)")

    return {
        "task_id": task_id,
        "episode": episode + 1,
        "final_score": round(final_score, 4),
        "total_reward": round(total_reward, 4),
        "final_height_cm": round(obs.plant_height_cm, 2),
        "final_yield_t_ha": round(obs.marketable_yield_t_ha, 4),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["seedling_vigor", "maximize_yield", "efficient_farmer", "all"])
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    env = PlantGrowthEnv(base_url="http://localhost:8000")

    tasks = ["seedling_vigor", "maximize_yield", "efficient_farmer"] if args.task == "all" else [args.task]
    all_results = []

    try:
        for task_id in tasks:
            for ep in range(args.episodes):
                result = await run_episode(env, task_id, ep)
                all_results.append(result)
    finally:
        await env.close()          # ← Fixed: now awaited

    # Save for submission
    with open("baseline_results.json", "w") as f:
        json.dump({"agent": "rule_based", "results": all_results}, f, indent=2)

    print("\n=== BASELINE RESULTS ===")
    for r in all_results:
        print(f"{r['task_id']:20} → Score: {r['final_score']:.4f} | Yield: {r['final_yield_t_ha']:.2f} t/ha")

    print("\n✅ baseline_results.json created successfully!")


if __name__ == "__main__":
    asyncio.run(main())