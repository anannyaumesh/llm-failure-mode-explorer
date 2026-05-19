"""
run_eval_v2.py
--------------
Full evaluation runner for the LLM Failure Mode Explorer dataset (v7) using
evaluator_v2.

Upgrades over v1:
  - 6-model registry (adds DeepSeek-V3 and Llama-4-Scout via Together.ai)
  - v7 schema: 'category' instead of 'type'; new fields (analysis_scope,
    constraint_complexity, anchor_topic, pair_id, failure_mode)
  - evaluator_v2 integration: captures sub_constraint_results (JSON string),
    hallucination_similarity, consistency_failure, consistency_details,
    injection_success, injection_details, evaluator_warnings, evaluator_version
  - Resume from checkpoint: skips (model, row_id) pairs already evaluated
  - Multi-seed mode: --seeds 1,2,3 + --temperature 0.7 for calibration subset
  - Cost tracking: token usage logged per call; running cost estimate
  - Smoke test: --smoke N runs N prompts on one model
  - Sample mode preserved: --sample N for subset runs

Usage:
  # Full evaluation
  python run_eval_v2.py --dataset data/questions_v7.csv

  # Smoke test (5 prompts on llama-4-scout)
  python run_eval_v2.py --dataset data/questions_v7.csv --smoke 5 --model llama-4-scout

  # Multi-seed calibration subset
  python run_eval_v2.py --dataset data/questions_v7.csv --sample 50 \\
                       --seeds 1,2,3 --temperature 0.7

  # Resume from checkpoint
  python run_eval_v2.py --dataset data/questions_v7.csv --resume
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Anthropic uses its own SDK (different request/response schema vs OpenAI)
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None

from prompts import get_prompt
from evaluator_v2 import evaluate_response, EVALUATOR_VERSION

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────

MAX_TOKENS = 1024  # bumped from 512 per audit — tier 4/5 prompts can exceed 512
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, exponential backoff applied
SLEEP_BETWEEN_CALLS = 0.3
CHECKPOINT_EVERY = 50
RESULTS_DIR = Path("results")
DEFAULT_CHECKPOINT = RESULTS_DIR / "results_partial.csv"
DEFAULT_FINAL = RESULTS_DIR / "results_v7.csv"

# ── API clients ────────────────────────────────────────────────────────────

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

mistral_client = OpenAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    base_url="https://api.mistral.ai/v1",
)

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

together_client = OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url="https://api.together.ai/v1",
)

# Gemini uses OpenAI-compatible endpoint — works with the standard OpenAI client
gemini_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

anthropic_client = None
if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Provider markers — used to dispatch the right call shape
PROVIDER_OPENAI = "openai_compat"  # all clients that mimic OpenAI's API
PROVIDER_ANTHROPIC = "anthropic"

# ── Model registry ─────────────────────────────────────────────────────────
# Schema: (client, api_model_id, cost_per_1k_input, cost_per_1k_output, provider) in USD
# Prices as of May 2026; update via provider pricing pages
#
# Model-tag pinning (per audit): floating tags like "mistral-small-latest" are
# replaced with dated checkpoints. For providers that don't expose dated tags
# (DeepSeek, Llama on Together), log the actual model ID returned by the API
# on the first call — see _log_first_call_model_id.

# Model registry — updated May 2026 per audit:
#   - gpt-4o-mini → gpt-4.1-mini-2025-04-14 (better IF, IFEval 84.1% vs 81%)
#   - mistral-small-2503 → mistral-small-2603 (Mistral Small 4; cheaper, current)
#   - DeepSeek-V3 → DeepSeek-V3.1 on Together (current stable release)
#   - Gemini 2.5 Flash added for Google representation in the model suite
#   - Llama-3.x dense models kept for size-controlled comparison vs Llama-4-Scout MoE
MODEL_REGISTRY = {
    "gpt-4.1-mini":     (openai_client,    "gpt-4.1-mini-2025-04-14",                   0.000400, 0.001600, PROVIDER_OPENAI),
    "mistral-small":    (mistral_client,   "mistral-small-2603",                        0.000070, 0.000300, PROVIDER_OPENAI),
    "gemini-2.5-flash": (gemini_client,    "gemini-2.5-flash",                          0.000300, 0.002500, PROVIDER_OPENAI),
    "llama-3.1-8b":     (groq_client,      "llama-3.1-8b-instant",                      0.000050, 0.000080, PROVIDER_OPENAI),
    "llama-3.3-70b":    (groq_client,      "llama-3.3-70b-versatile",                   0.000590, 0.000790, PROVIDER_OPENAI),
    "deepseek-v3":      (together_client,  "deepseek-ai/DeepSeek-V3.1",                 0.001250, 0.001250, PROVIDER_OPENAI),
    "llama-4-scout":    (together_client,  "meta-llama/Llama-4-Scout-17B-16E-Instruct", 0.000180, 0.000600, PROVIDER_OPENAI),
    "claude-haiku-4-5": (anthropic_client, "claude-haiku-4-5-20251001",                 0.001000, 0.005000, PROVIDER_ANTHROPIC),
}

# Models requiring reasoning_effort=none to disable hybrid reasoning mode.
# (Mistral Small 4 is hybrid instruct+reasoning+coding; Gemini 2.5 Flash is
# a thinking model by default. Disable both for fair non-reasoning comparison.)
REASONING_DISABLED_MODELS = {"mistral-small", "gemini-2.5-flash"}

DEFAULT_MODELS = list(MODEL_REGISTRY.keys())

# Track first-call model IDs for reproducibility logging
_LOGGED_MODEL_IDS = {}


# ── Logging ────────────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a"),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


# ── Output parsing (preserved from v1) ─────────────────────────────────────

def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def attempt_partial_json_recovery(text: str) -> dict | None:
    """Pull answer/confidence out of truncated JSON. Preserves v1 logic."""
    answer_match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    conf_match = re.search(r'"confidence"\s*:\s*"(high|medium|low)"', text)
    if answer_match:
        return {
            "answer": answer_match.group(1),
            "confidence": conf_match.group(1) if conf_match else "low",
            "_recovered": True,
        }
    return None


def parse_model_output(raw: str) -> tuple[dict | None, bool, str]:
    """Returns (parsed_dict, format_error, parse_note)."""
    cleaned = strip_markdown_fences(raw)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed, False, "ok"
        return None, True, f"JSON parsed but missing 'answer' key: {list(parsed.keys())}"
    except json.JSONDecodeError:
        pass

    recovered = attempt_partial_json_recovery(cleaned)
    if recovered:
        return recovered, False, "recovered_from_truncated_json"

    return None, True, f"Unparseable output: {raw[:80]}"


# ── API call with retry, token tracking, and provider dispatch ─────────────

def _log_first_call_model_id(model_name: str, response, log) -> None:
    """
    Log the actual model ID returned by the provider on the first call.
    Audit fix: floating-tag providers (mistral, together) may resolve aliases
    to specific versions; logging this makes the run reproducible.
    """
    if model_name in _LOGGED_MODEL_IDS:
        return
    actual_id = getattr(response, "model", None)
    if actual_id:
        _LOGGED_MODEL_IDS[model_name] = actual_id
        if log:
            log.info(f"  Resolved model ID for {model_name}: {actual_id}")


def _call_openai_compat(client, api_model, messages, temperature, seed, model_name=""):
    """
    OpenAI-compatible call. Sets top_p=1.0 explicitly (audit fix).
    For hybrid reasoning models (e.g., Mistral Small 4), passes
    reasoning_effort='none' to disable reasoning mode for fair comparison
    with non-reasoning models.
    """
    kwargs = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": MAX_TOKENS,
    }
    if seed is not None:
        kwargs["seed"] = seed
    if model_name in REASONING_DISABLED_MODELS:
        # Pass through extra_body for non-standard params on OpenAI-compatible APIs
        kwargs["extra_body"] = {"reasoning_effort": "none"}
    return client.chat.completions.create(**kwargs)


def _call_anthropic(client, api_model, messages, temperature, seed):
    """
    Anthropic call. Differences from OpenAI:
      - system message is a separate parameter, not part of `messages`
      - response.content is a list of content blocks; we extract text
      - no native seed param; we ignore the seed (T=0 should be deterministic)
      - usage fields are input_tokens / output_tokens (not prompt/completion)
    """
    # Split off system messages
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    system_text = "\n\n".join(system_parts) if system_parts else None

    kwargs = {
        "model": api_model,
        "messages": non_system,
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": MAX_TOKENS,
    }
    if system_text:
        kwargs["system"] = system_text

    response = client.messages.create(**kwargs)

    # Wrap in a shim object so caller code can use uniform attribute access
    class _Shim:
        pass
    shim = _Shim()
    # Extract text from content blocks (Anthropic returns list[ContentBlock])
    text_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    shim.text = "".join(text_parts)
    shim.prompt_tokens = response.usage.input_tokens if response.usage else 0
    shim.completion_tokens = response.usage.output_tokens if response.usage else 0
    shim.model = response.model
    return shim


def call_model(
    model_name: str,
    messages: list[dict],
    temperature: float = 0.0,
    seed: int | None = None,
    log: logging.Logger | None = None,
) -> tuple[str, str, dict]:
    """
    Returns (output_text, error_string, usage_dict).
    usage_dict has 'prompt_tokens', 'completion_tokens', 'total_tokens'.
    """
    client, api_model, _, _, provider = MODEL_REGISTRY[model_name]

    if client is None:
        return "", f"No client for {model_name} (missing API key or SDK)", {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider == PROVIDER_ANTHROPIC:
                shim = _call_anthropic(client, api_model, messages, temperature, seed)
                _log_first_call_model_id(model_name, shim, log)
                usage = {
                    "prompt_tokens": shim.prompt_tokens,
                    "completion_tokens": shim.completion_tokens,
                    "total_tokens": shim.prompt_tokens + shim.completion_tokens,
                }
                return shim.text.strip(), "", usage
            else:
                response = _call_openai_compat(client, api_model, messages, temperature, seed, model_name)
                _log_first_call_model_id(model_name, response, log)
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
                return response.choices[0].message.content.strip(), "", usage
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * (2 ** (attempt - 1))  # exponential backoff
                if log:
                    log.warning(f"  [{model_name}] attempt {attempt} failed: {e}. Retry in {wait}s")
                time.sleep(wait)
            else:
                return "", str(e), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return "", "Max retries exceeded", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def estimate_cost(model_name: str, usage: dict) -> float:
    """Return cost in USD for a single call."""
    _, _, cost_in, cost_out, _ = MODEL_REGISTRY[model_name]
    return (usage["prompt_tokens"] * cost_in / 1000) + (usage["completion_tokens"] * cost_out / 1000)


# ── Row utilities ──────────────────────────────────────────────────────────

def safe_str(value) -> str:
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).strip() if value is not None else ""


def _normalize_seed(value) -> str:
    """
    Normalize a seed value (read from CSV) to a canonical string.

    CSV roundtrip can convert int seeds to floats: "1" → 1.0 → "1.0".
    This normalizer makes "1", 1, 1.0, "1.0" all → "1", and missing → "".
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    # If it looks like a float-encoded int, strip the trailing ".0"
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return s
    except (ValueError, TypeError):
        return s


def load_existing_results(checkpoint_path: Path) -> tuple[list[dict], set[tuple]]:
    """
    Load already-evaluated (model, row_id, seed) tuples for resume.
    Returns (existing_rows, set of (model, row_id, seed) keys).
    """
    if not checkpoint_path.exists():
        return [], set()
    df = pd.read_csv(checkpoint_path)
    done_keys = set()
    for _, row in df.iterrows():
        seed_str = _normalize_seed(row.get("seed", ""))
        done_keys.add((str(row["model"]), str(row["id"]), seed_str))
    return df.to_dict("records"), done_keys


# ── Main evaluation loop ───────────────────────────────────────────────────

def run_evaluation(
    dataset_path: str,
    models_to_run: list[str],
    sample_n: int | None = None,
    smoke_n: int | None = None,
    seeds: list[int] | None = None,
    temperature: float = 0.0,
    resume: bool = False,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
    random_state: int = 42,
):
    """Run the full evaluation pipeline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log = setup_logger(RESULTS_DIR / "eval.log")
    log.info(f"Evaluator: {EVALUATOR_VERSION}")
    log.info(f"Dataset: {dataset_path}")
    log.info(f"Models: {models_to_run}")
    log.info(f"Temperature: {temperature}, Seeds: {seeds or [None]}")

    checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
    output_path = output_path or DEFAULT_FINAL

    df = pd.read_csv(dataset_path)
    assert df["id"].is_unique, "Duplicate IDs in dataset"

    if smoke_n:
        df = df.head(smoke_n)
        log.info(f"SMOKE TEST: first {smoke_n} rows")
    elif sample_n:
        df = df.sample(sample_n, random_state=random_state)
        log.info(f"Sampled {sample_n} rows (seed={random_state})")
    else:
        log.info(f"Full run on {len(df)} rows")

    seeds = seeds or [None]
    total_calls = len(models_to_run) * len(df) * len(seeds)
    log.info(f"Total API calls planned: {total_calls}")

    # Resume support
    existing_results, done_keys = [], set()
    if resume:
        existing_results, done_keys = load_existing_results(checkpoint_path)
        log.info(f"Resume: {len(done_keys)} (model, id, seed) tuples already done")

    results = existing_results.copy()
    call_count = 0
    skipped = 0
    total_cost = 0.0
    per_model_cost = {m: 0.0 for m in models_to_run}

    for model_name in models_to_run:
        log.info(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")

        for seed in seeds:
            for _, row in df.iterrows():
                call_count += 1

                row_id = safe_str(row.get("id", ""))
                question = safe_str(row.get("question", ""))
                category = safe_str(row.get("category", ""))
                subcategory = safe_str(row.get("subcategory", ""))
                difficulty = safe_str(row.get("difficulty", ""))
                expected = safe_str(row.get("expected_answer", ""))
                # v7 extra fields preserved for downstream analysis
                anchor_topic = safe_str(row.get("anchor_topic", ""))
                pair_id = safe_str(row.get("pair_id", ""))
                failure_mode = safe_str(row.get("failure_mode", ""))
                analysis_scope = safe_str(row.get("analysis_scope", ""))
                constraint_complexity = safe_str(row.get("constraint_complexity", ""))

                if not question:
                    log.warning(f"Skipping row {row_id}: empty question")
                    continue

                seed_str = _normalize_seed(seed) if seed is not None else ""
                if (model_name, row_id, seed_str) in done_keys:
                    skipped += 1
                    continue

                log.info(f"[{call_count}/{total_calls}] {model_name} id={row_id} "
                         f"{category}/{subcategory}" + (f" seed={seed}" if seed else ""))

                # API call
                messages = get_prompt(question)
                raw_output, api_error, usage = call_model(
                    model_name, messages, temperature=temperature, seed=seed, log=log,
                )

                cost = estimate_cost(model_name, usage)
                total_cost += cost
                per_model_cost[model_name] += cost

                # Base row for both success and failure paths
                base_row = {
                    "model": model_name,
                    "model_id_resolved": _LOGGED_MODEL_IDS.get(model_name, ""),
                    "seed": seed_str,
                    "id": row_id,
                    "question": question,
                    "category": category,
                    "subcategory": subcategory,
                    "difficulty": difficulty,
                    "expected": expected,
                    "anchor_topic": anchor_topic,
                    "pair_id": pair_id,
                    "failure_mode": failure_mode,
                    "analysis_scope": analysis_scope,
                    "constraint_complexity": constraint_complexity,
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "cost_usd": round(cost, 6),
                }

                if api_error:
                    log.error(f"API error: {api_error}")
                    results.append({
                        **base_row,
                        "raw_output": "",
                        "answer": "",
                        "confidence": "",
                        "format_error": True,
                        "parse_note": api_error,
                        "constraint_violation": False,
                        "constraint_details": "",
                        "sub_constraint_results": "{}",
                        "hallucination": False,
                        "hallucination_details": "",
                        "hallucination_similarity": 1.0,
                        "incomplete": True,
                        "incomplete_details": f"API error: {api_error}",
                        "sycophancy": False,
                        "sycophancy_details": "",
                        "injection_success": False,
                        "injection_details": "",
                        "consistency_failure": False,
                        "consistency_details": "",
                        "evaluator_warnings": "[]",
                        "evaluator_version": EVALUATOR_VERSION,
                        "error": api_error,
                    })

                    if len(results) % CHECKPOINT_EVERY == 0:
                        pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                        log.info(f"Checkpoint saved at {len(results)} rows")
                    time.sleep(SLEEP_BETWEEN_CALLS)
                    continue

                # Parse
                parsed, format_error, parse_note = parse_model_output(raw_output)
                if format_error:
                    log.warning(f"Parse failed: {parse_note}")
                    answer, confidence = "", ""
                else:
                    answer = safe_str(parsed.get("answer", ""))
                    confidence = safe_str(parsed.get("confidence", ""))

                # Evaluate
                eval_metrics = evaluate_response(
                    answer=answer,
                    expected=expected,
                    category=category,
                    subcategory=subcategory,
                    question=question,
                    row_id=row_id,
                )

                # Convert sub_constraint_results to JSON string for CSV
                sub_results_json = json.dumps(eval_metrics["sub_constraint_results"])
                warnings_json = json.dumps(eval_metrics["evaluator_warnings"])

                # Inline logging of violations
                if eval_metrics["constraint_violation"]:
                    log.info(f"  ⚠ CONSTRAINT: {eval_metrics['constraint_details']}")
                if eval_metrics["hallucination"]:
                    log.info(f"  ⚠ HALLUC: {eval_metrics['hallucination_details']}")
                if eval_metrics["sycophancy"]:
                    log.info(f"  ⚠ SYCO: {eval_metrics['sycophancy_details']}")
                if eval_metrics["injection_success"]:
                    log.info(f"  ⚠ INJECT: {eval_metrics['injection_details']}")
                if eval_metrics["consistency_failure"]:
                    log.info(f"  ⚠ CONSIST: {eval_metrics['consistency_details']}")

                results.append({
                    **base_row,
                    "raw_output": raw_output,
                    "answer": answer,
                    "confidence": confidence,
                    "format_error": format_error,
                    "parse_note": parse_note,
                    "constraint_violation": eval_metrics["constraint_violation"],
                    "constraint_details": eval_metrics["constraint_details"],
                    "sub_constraint_results": sub_results_json,
                    "hallucination": eval_metrics["hallucination"],
                    "hallucination_details": eval_metrics["hallucination_details"],
                    "hallucination_similarity": eval_metrics["hallucination_similarity"],
                    "incomplete": eval_metrics["incomplete"],
                    "incomplete_details": eval_metrics["incomplete_details"],
                    "sycophancy": eval_metrics["sycophancy"],
                    "sycophancy_details": eval_metrics["sycophancy_details"],
                    "injection_success": eval_metrics["injection_success"],
                    "injection_details": eval_metrics["injection_details"],
                    "consistency_failure": eval_metrics["consistency_failure"],
                    "consistency_details": eval_metrics["consistency_details"],
                    "evaluator_warnings": warnings_json,
                    "evaluator_version": eval_metrics["evaluator_version"],
                    "error": "",
                })

                if len(results) % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                    log.info(f"  Checkpoint: {len(results)} rows; cost so far ${total_cost:.2f}")

                time.sleep(SLEEP_BETWEEN_CALLS)

    # Final save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    log.info(f"\nSaved {len(results_df)} rows to {output_path}")
    log.info(f"Skipped (resume): {skipped}")
    log.info(f"Total cost: ${total_cost:.4f}")
    for m, c in per_model_cost.items():
        log.info(f"  {m}: ${c:.4f}")

    _print_summary(results_df)
    return results_df


def _print_summary(df: pd.DataFrame):
    """Per-model breakdown of all failure flags."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    flags = [
        "format_error", "constraint_violation", "hallucination",
        "incomplete", "sycophancy", "injection_success", "consistency_failure",
    ]
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        n = len(sub)
        print(f"\n{model} (n={n})")
        for flag in flags:
            if flag in sub.columns:
                count = sub[flag].astype(str).str.lower().eq("true").sum()
                pct = 100 * count / n if n else 0
                print(f"  {flag:<25} {count:4d}/{n} ({pct:5.1f}%)")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/questions_v7.csv")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="Comma-separated model names")
    parser.add_argument("--model", default=None,
                        help="Single model (overrides --models). Used for smoke tests")
    parser.add_argument("--sample", type=int, default=None,
                        help="Random subset of N prompts")
    parser.add_argument("--smoke", type=int, default=None,
                        help="Smoke test: first N prompts on a single model")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seeds (e.g. '1,2,3') — enables multi-seed mode")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from results_partial.csv, skipping completed rows")
    parser.add_argument("--output", default=str(DEFAULT_FINAL),
                        help="Output CSV path")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                        help="Checkpoint CSV path")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.model:
        models = [args.model]
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip() in MODEL_REGISTRY]

    seeds = None
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    run_evaluation(
        dataset_path=args.dataset,
        models_to_run=models,
        sample_n=args.sample,
        smoke_n=args.smoke,
        seeds=seeds,
        temperature=args.temperature,
        resume=args.resume,
        output_path=Path(args.output),
        checkpoint_path=Path(args.checkpoint),
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
