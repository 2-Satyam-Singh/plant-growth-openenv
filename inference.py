"""
Inference Script — Tomato Plant Growth OpenEnv
===============================================

MANDATORY environment variables:
  API_BASE_URL  The API endpoint for the LLM.
  MODEL_NAME    The model identifier to use for inference.
  HF_TOKEN      Your Hugging Face / API key.

Usage:
  python inference.py                      # LLM agent, all tasks
  python inference.py --agent rule         # rule-based (no API key needed)
  python inference.py --task seedling_vigor
  python inference.py --episodes 3
"""

import argparse
import asyncio
import json
import os
import sys
import textwrap

from openai import OpenAI

try:
    from plant_growth.client import PlantGrowthEnv
    from plant_growth.models import PlantGrowthAction, PlantGrowthObservation
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from client import PlantGrowthEnv
    from models import PlantGrowthAction, PlantGrowthObservation

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

TEMPERATURE = 0.2
MAX_TOKENS  = 512

TASK_IDS = ["seedling_vigor", "maximize_yield", "efficient_farmer"]

TASK_MAX_STEPS = {"seedling_vigor": 5, "maximize_yield": 18, "efficient_farmer": 18}
TASK_MAX_DAYS  = {"seedling_vigor": 30, "maximize_yield": 120, "efficient_farmer": 120}

# ────────────────────────────────────────────────────────────────────────────
# Rule-based agent
# ────────────────────────────────────────────────────────────────────────────

def rule_based_action(obs: PlantGrowthObservation, task_id: str) -> PlantGrowthAction:
    stage = obs.growth_stage

    if task_id == "efficient_farmer":
        irr = {"seedling": 0.70, "flowering": 0.65, "fruiting": 0.60}[stage]
    elif task_id == "seedling_vigor":
        irr = 0.75
    else:
        irr = {"seedling": 0.80, "flowering": 0.75, "fruiting": 0.70}[stage]

    if obs.water_stress_index > 0.40:
        irr = min(1.0, irr + 0.15)
    if obs.soil_moisture > 0.85:
        irr = max(0.0, irr - 0.20)

    fert_type = "combined"
    fert_amt  = {"seedling": 0.60, "flowering": 0.75, "fruiting": 0.65}[stage]

    if obs.nutrient_level < 0.25:
        fert_amt = min(1.0, fert_amt + 0.20)
    if task_id == "efficient_farmer" and obs.nutrient_level > 0.55:
        fert_amt = max(0.3, fert_amt - 0.15)

    return PlantGrowthAction(
        irrigation_fraction=round(irr, 2),
        fertilizer_amount=round(fert_amt, 2),
        fertilizer_type=fert_type,
        substrate_type="rice_husk",
    )

# ────────────────────────────────────────────────────────────────────────────
# LLM agent helpers
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert agronomist managing a tomato greenhouse farm.
Reply ONLY with a valid JSON object — no other text:

{
  "irrigation_fraction": 0.6,
  "fertilizer_amount": 0.7,
  "fertilizer_type": "combined",
  "substrate_type": "rice_husk",
  "reasoning": "one line rationale"
}

Key facts:
- irrigation_fraction 0.0-1.0: 0.60 is optimal (same yield, 40% less water).
- fertilizer_type: "combined" (bio-inoculants + NPK) gives best growth.
- substrate_type: "rice_husk" gives 54% more height than "soil". Set once at start.
- Water stress > 0.4 causes Blossom End Rot. Keep soil moisture adequate.
""").strip()


def build_llm_prompt(step: int, obs: PlantGrowthObservation, task_id: str) -> str:
    task_goal = {
        "seedling_vigor":   "Maximize height+leaves by day 30.",
        "maximize_yield":   "Maximize marketable yield (t/ha) over 120 days. Target >= 99 t/ha.",
        "efficient_farmer": "Maximize IWUE (kg/m3). Keep BER < 10%, marketable fraction > 70%.",
    }[task_id]
    return (
        f"TASK: {task_goal}\n"
        f"Week {step} | Day {obs.day} | Stage: {obs.growth_stage}\n"
        f"Height: {obs.plant_height_cm:.1f}cm | Leaves: {obs.leaf_count} | Fruits: {obs.fruit_count}\n"
        f"Soil moisture: {obs.soil_moisture:.2f} | Water stress: {obs.water_stress_index:.2f}\n"
        f"Nutrients: {obs.nutrient_level:.2f} | Yield: {obs.marketable_yield_t_ha:.2f} t/ha\n"
        f"BER: {obs.blossom_end_rot_fraction*100:.1f}% | IWUE: {obs.iwue_kg_m3:.2f} kg/m3\n"
        "Reply with JSON only."
    )


def parse_llm_action(text: str) -> PlantGrowthAction:
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        return PlantGrowthAction(
            irrigation_fraction=float(data.get("irrigation_fraction", 0.6)),
            fertilizer_amount=float(data.get("fertilizer_amount",   0.6)),
            fertilizer_type=str(data.get("fertilizer_type",   "combined")),
            substrate_type=str(data.get("substrate_type",    "rice_husk")),
        )
    except Exception:
        return PlantGrowthAction(
            irrigation_fraction=0.6, fertilizer_amount=0.6,
            fertilizer_type="combined", substrate_type="rice_husk",
        )

# ────────────────────────────────────────────────────────────────────────────
# Graders
# ────────────────────────────────────────────────────────────────────────────

def grade(obs: PlantGrowthObservation, task_id: str) -> float:
    if task_id == "seedling_vigor":
        return min(1.0, (
            0.60 * min(1.0, obs.plant_height_cm / 35.0) +
            0.30 * min(1.0, obs.leaf_count / 50.0) +
            0.10 * (1.0 - obs.water_stress_index)
        ))
    elif task_id == "maximize_yield":
        return min(1.0, obs.marketable_yield_t_ha / 99.27)
    else:
        iwue_score = min(1.0, obs.iwue_kg_m3 / 35.0)
        mkt_score  = min(1.0, (1.0 - obs.blossom_end_rot_fraction) / 0.82)
        ber_score  = min(1.0, max(0.0, 1.0 - obs.blossom_end_rot_fraction / 0.10))
        return 0.50 * iwue_score + 0.30 * mkt_score + 0.20 * ber_score

# ────────────────────────────────────────────────────────────────────────────
# Async episode runner
# ────────────────────────────────────────────────────────────────────────────

async def run_episode(env: PlantGrowthEnv, task_id: str, episode: int,
                      agent: str, llm_client=None) -> dict:
    max_steps = TASK_MAX_STEPS[task_id]
    max_days  = TASK_MAX_DAYS[task_id]

    print(f"[START] task={task_id} episode={episode+1} agent={agent}", flush=True)

    result = await env.reset()
    obs: PlantGrowthObservation = result.observation
    total_reward = 0.0
    step = 0

    for step in range(1, max_steps + 1):
        if result.done or obs.day >= max_days:
            break

        # Choose action
        if agent == "rule" or llm_client is None:
            action = rule_based_action(obs, task_id)
        else:
            prompt = build_llm_prompt(step, obs, task_id)
            try:
                completion = llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response_text = completion.choices[0].message.content or ""
                action = parse_llm_action(response_text)
            except Exception as e:
                print(f"[WARN] LLM error step {step}: {e}. Using rule fallback.", flush=True)
                action = rule_based_action(obs, task_id)

        result = await env.step(action)
        obs    = result.observation
        reward = result.reward or 0.0
        total_reward += reward

        # Required structured output
        print(
            f"[STEP] task={task_id} step={step} day={obs.day} "
            f"stage={obs.growth_stage} "
            f"reward={reward:.4f} "
            f"height={obs.plant_height_cm:.2f} "
            f"leaves={obs.leaf_count} "
            f"fruits={obs.fruit_count} "
            f"yield={obs.marketable_yield_t_ha:.4f} "
            f"ber={obs.blossom_end_rot_fraction:.4f} "
            f"iwue={obs.iwue_kg_m3:.4f} "
            f"water_stress={obs.water_stress_index:.4f} "
            f"soil_moisture={obs.soil_moisture:.4f} "
            f"nutrient={obs.nutrient_level:.4f} "
            f"action_irr={action.irrigation_fraction:.2f} "
            f"action_fert={action.fertilizer_amount:.2f} "
            f"action_fert_type={action.fertilizer_type} "
            f"substrate={obs.substrate}",
            flush=True,
        )

    final_score = grade(obs, task_id)

    print(
        f"[END] task={task_id} episode={episode+1} "
        f"score={final_score:.4f} "
        f"steps={step} "
        f"total_reward={total_reward:.4f} "
        f"final_day={obs.day} "
        f"height={obs.plant_height_cm:.2f} "
        f"leaves={obs.leaf_count} "
        f"fruits={obs.fruit_count} "
        f"yield={obs.marketable_yield_t_ha:.4f} "
        f"ber={obs.blossom_end_rot_fraction:.4f} "
        f"iwue={obs.iwue_kg_m3:.4f} "
        f"water_used={obs.total_water_applied_m3_ha:.2f} "
        f"substrate={obs.substrate}",
        flush=True,
    )

    return {
        "task_id":               task_id,
        "episode":               episode + 1,
        "agent":                 agent,
        "steps":                 step,
        "final_day":             obs.day,
        "score":                 round(final_score, 4),
        "total_reward":          round(total_reward, 4),
        "final_height_cm":       round(obs.plant_height_cm, 2),
        "final_leaf_count":      obs.leaf_count,
        "final_fruit_count":     obs.fruit_count,
        "fruit_weight_g":        round(obs.fruit_weight_g, 2),
        "marketable_yield_t_ha": round(obs.marketable_yield_t_ha, 4),
        "ber_pct":               round(obs.blossom_end_rot_fraction * 100, 3),
        "water_m3_ha":           round(obs.total_water_applied_m3_ha, 2),
        "iwue_kg_m3":            round(obs.iwue_kg_m3, 4),
        "substrate":             obs.substrate,
    }

# ────────────────────────────────────────────────────────────────────────────
# Async main
# ────────────────────────────────────────────────────────────────────────────

async def async_main(args) -> None:
    # LLM client (sync OpenAI is fine — it's called from async context)
    llm_client = None
    if args.agent == "llm":
        if not API_KEY:
            print("[WARN] HF_TOKEN/API_KEY not set — falling back to rule-based agent.", flush=True)
            args.agent = "rule"
        else:
            llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            print(f"[INFO] LLM agent: {MODEL_NAME} via {API_BASE_URL}", flush=True)

    if args.agent == "rule":
        print("[INFO] Agent: rule-based (no API key needed)", flush=True)

    tasks_to_run = TASK_IDS if args.task == "all" else [args.task]
    all_results  = []

    for task_id in tasks_to_run:
        async with PlantGrowthEnv(base_url=args.server) as env:
            for ep in range(args.episodes):
                r = await run_episode(env, task_id, ep, args.agent, llm_client)
                all_results.append(r)

    # Save results
    output = {
        "agent":   args.agent,
        "model":   MODEL_NAME if args.agent == "llm" else "rule_based",
        "results": all_results,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== BASELINE RESULTS ===", flush=True)
    for r in all_results:
        print(
            f"{r['task_id']} → Score: {r['score']:.4f} | "
            f"Yield: {r['marketable_yield_t_ha']:.2f} t/ha | "
            f"IWUE: {r['iwue_kg_m3']:.2f} kg/m3 | "
            f"BER: {r['ber_pct']:.1f}% | "
            f"Reward: {r['total_reward']:.4f}",
            flush=True,
        )
    print("baseline_results.json saved.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tomato Plant Growth — inference baseline")
    parser.add_argument("--agent",    default="llm", choices=["rule", "llm"],
                        help="'llm' uses OpenAI client (requires HF_TOKEN); 'rule' = no key needed")
    parser.add_argument("--task",     default="all", choices=TASK_IDS + ["all"])
    parser.add_argument("--episodes", default=1,     type=int)
    parser.add_argument("--server",   default="http://localhost:8000")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()