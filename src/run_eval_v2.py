"""
run_eval_v2.py
"""

import pandas as pd
import json
import os
import re
import time
import logging
from dotenv import load_dotenv
from openai import OpenAI

from prompts import get_prompt
from evaluator_v2 import evaluate_response

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/eval_v2.log"),
    ]
)
log = logging.getLogger(__name__)

MAX_TOKENS          = 512
MAX_RETRIES         = 3
RETRY_DELAY         = 2
SLEEP_BETWEEN_CALLS = 0.3
CHECKPOINT_EVERY    = 50
CHECKPOINT_PATH     = "results/results_v2_partial.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Clients
# ─────────────────────────────────────────────────────────────────────────────

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

mistral_client = OpenAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    base_url="https://api.mistral.ai/v1",
)

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

gemini_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

together_client = OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    # Original 4 models
    "gpt-4o-mini":      (openai_client,   "gpt-4o-mini"),
    "mistral-small":    (mistral_client,  "mistral-small-latest"),
    "llama-3.1-8b":     (groq_client,     "llama-3.1-8b-instant"),
    "llama-3.3-70b":    (groq_client,     "llama-3.3-70b-versatile"),
    # New models v2
    "gemini-1.5-flash": (gemini_client,   "gemini-1.5-flash"),
    "qwen-2.5-72b":     (together_client, "Qwen/Qwen2.5-72B-Instruct-Turbo"),
}

MODELS_TO_RUN = list(MODEL_REGISTRY.keys())

# ─────────────────────────────────────────────────────────────────────────────
# JSON parser
# ─────────────────────────────────────────────────────────────────────────────

def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def attempt_partial_json_recovery(text: str) -> dict | None:
    """Recover from truncated JSON (Groq bug)."""
    answer_match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    conf_match   = re.search(r'"confidence"\s*:\s*"(high|medium|low)"', text)
    if answer_match:
        return {
            "answer":     answer_match.group(1),
            "confidence": conf_match.group(1) if conf_match else "low",
            "_recovered": True,
        }
    return None


def parse_model_output(raw: str, model_name: str) -> tuple:
    """
    Returns (parsed_dict | None, format_error: bool, parse_note: str).
    Always strips fences defensively regardless of model.
    """
    cleaned = strip_markdown_fences(raw)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed, False, "ok"
        return None, True, f"JSON ok but no 'answer' key: {list(parsed.keys())}"
    except json.JSONDecodeError:
        pass

    recovered = attempt_partial_json_recovery(cleaned)
    if recovered:
        return recovered, False, "recovered_from_truncated_json"

    return None, True, f"Unparseable: {raw[:80]}"


# ─────────────────────────────────────────────────────────────────────────────
# API call with retry + exponential backoff
# ─────────────────────────────────────────────────────────────────────────────

def call_model(model_name: str, messages: list) -> tuple:
    """Returns (output_text: str, error: str). error is '' on success."""
    client, api_model = MODEL_REGISTRY[model_name]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=api_model,
                messages=messages,
                temperature=0,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content.strip(), ""
        except Exception as e:
            err_str = str(e)
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                log.warning(
                    f"  [{model_name}] attempt {attempt} failed: {err_str[:80]}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                return "", err_str

    return "", "Max retries exceeded"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_str(value) -> str:
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).strip() if value is not None else ""


def _empty_result(model_name, row_id, question, category, subcategory,
                  difficulty, expected, constraint_complexity,
                  constraint_priority, attack_intent, error_msg):
    return {
        "model":                   model_name,
        "id":                      row_id,
        "question":                question,
        "category":                category,
        "subcategory":             subcategory,
        "difficulty":              difficulty,
        "expected":                expected,
        "constraint_complexity":   constraint_complexity,
        "constraint_priority":     constraint_priority,
        "attack_intent":           attack_intent,
        "raw_output":              "",
        "answer":                  "",
        "confidence":              "",
        "format_error":            True,
        "parse_note":              error_msg,
        "constraint_violation":    False,
        "constraint_details":      "",
        "sub_constraint_results":  "{}",
        "hallucination":           False,
        "hallucination_details":   "",
        "incomplete":              True,
        "incomplete_details":      f"API error: {error_msg}",
        "sycophancy":              False,
        "sycophancy_details":      "",
        "injection_success":       False,
        "system_prompt_leak":      False,
        "injection_details":       "",
        "error":                   error_msg,
    }


def _maybe_checkpoint(results):
    if len(results) % CHECKPOINT_EVERY == 0:
        pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
        log.info(f"Checkpoint saved ({len(results)} rows)")


def _print_summary(df):
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    flags = [
        "format_error", "constraint_violation", "hallucination",
        "incomplete", "sycophancy", "injection_success",
    ]
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        n   = len(sub)
        print(f"\n{model} (n={n})")
        for flag in flags:
            if flag in sub.columns:
                count = sub[flag].astype(str).str.lower().eq("true").sum()
                pct   = 100 * count // n if n else 0
                print(f"  {flag}: {count}/{n} ({pct}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    dataset_path: str = "data/questions_v5.csv",
    output_path:  str = "results/results_v2.csv",
    sample_n:     int | None = None,
    random_state: int = 42,
    models:       list | None = None,
    skip_ids:     set | None = None,
):
    """
    Run the full evaluation pipeline.

    Parameters
    ----------
    dataset_path : path to questions CSV
    output_path  : where to write results
    sample_n     : if set, only run this many rows (for smoke-testing)
    random_state : seed for reproducible sampling
    models       : list of model keys to run; defaults to all 6
    skip_ids     : set of row IDs to skip (for targeted reruns)
    """
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(dataset_path)

    # Column name compatibility — v4 uses "type" and "expected_answer"
    if "category" not in df.columns and "type" in df.columns:
        df = df.rename(columns={"type": "category"})
    if "expected" not in df.columns and "expected_answer" in df.columns:
        df = df.rename(columns={"expected_answer": "expected"})

    assert df["id"].is_unique, (
        f"Duplicate IDs detected: {df[df['id'].duplicated()]['id'].tolist()}"
    )

    if skip_ids:
        before = len(df)
        df = df[~df["id"].astype(str).isin({str(i) for i in skip_ids})]
        log.info(f"Skipping {before - len(df)} already-completed rows")

    if sample_n:
        df = df.sample(min(sample_n, len(df)), random_state=random_state)
        log.info(f"Sampled {len(df)} rows")

    models_to_run = models or MODELS_TO_RUN
    total_calls   = len(models_to_run) * len(df)
    log.info(f"Models: {models_to_run}")
    log.info(f"Rows per model: {len(df)}  |  Total API calls: {total_calls}")

    results    = []
    call_count = 0

    for model_name in models_to_run:
        log.info(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")

        for _, row in df.iterrows():
            call_count += 1

            row_id                 = safe_str(row.get("id", ""))
            question               = safe_str(row.get("question", ""))
            category               = safe_str(row.get("category", ""))
            subcategory            = safe_str(row.get("subcategory", ""))
            difficulty             = safe_str(row.get("difficulty", ""))
            expected               = safe_str(row.get("expected", ""))
            constraint_complexity  = safe_str(row.get("constraint_complexity", ""))
            constraint_priority    = safe_str(row.get("constraint_priority", ""))
            attack_intent          = safe_str(row.get("attack_intent", ""))

            if not question:
                log.warning(f"Skipping row {row_id}: empty question")
                continue

            log.info(
                f"[{call_count}/{total_calls}] "
                f"id={row_id} model={model_name} "
                f"cat={category}/{subcategory}"
            )

            messages = get_prompt(question)

            # ── API call ──────────────────────────────────────────────────────
            raw_output, api_error = call_model(model_name, messages)

            if api_error:
                log.error(f"API error id={row_id}: {api_error[:120]}")
                results.append(
                    _empty_result(
                        model_name, row_id, question, category, subcategory,
                        difficulty, expected, constraint_complexity,
                        constraint_priority, attack_intent, api_error
                    )
                )
                _maybe_checkpoint(results)
                time.sleep(SLEEP_BETWEEN_CALLS)
                continue

            # ── Parse ─────────────────────────────────────────────────────────
            parsed, format_error, parse_note = parse_model_output(
                raw_output, model_name
            )

            if format_error:
                log.warning(f"Parse failed id={row_id}: {parse_note}")
                answer     = ""
                confidence = ""
            else:
                answer     = safe_str(parsed.get("answer", ""))
                confidence = safe_str(parsed.get("confidence", ""))

            # ── Evaluate ──────────────────────────────────────────────────────
            eval_metrics = evaluate_response(
                answer=answer,
                expected=expected,
                category=category,
                subcategory=subcategory,
                question=question,
                row_id=row_id,
            )

            # Log key signals
            if eval_metrics["constraint_violation"]:
                log.info(f"  CV: {eval_metrics['constraint_details']}")
                log.info(f"  SUB: {eval_metrics['sub_constraint_results']}")
            if eval_metrics["hallucination"]:
                log.info(f"  HALL: {eval_metrics['hallucination_details']}")
            if eval_metrics["injection_success"]:
                log.info(f"  INJECTION: {eval_metrics['injection_details']}")
            if eval_metrics["sycophancy"]:
                log.info(f"  SYCO: {eval_metrics['sycophancy_details']}")

            # Serialise sub_constraint_results dict → JSON string for CSV
            sub_results_str = json.dumps(
                eval_metrics.pop("sub_constraint_results")
            )

            results.append({
                "model":                  model_name,
                "id":                     row_id,
                "question":               question,
                "category":               category,
                "subcategory":            subcategory,
                "difficulty":             difficulty,
                "expected":               expected,
                "constraint_complexity":  constraint_complexity,
                "constraint_priority":    constraint_priority,
                "attack_intent":          attack_intent,
                "raw_output":             raw_output,
                "answer":                 answer,
                "confidence":             confidence,
                "format_error":           format_error,
                "parse_note":             parse_note,
                "sub_constraint_results": sub_results_str,
                **eval_metrics,
                "error": "",
            })

            _maybe_checkpoint(results)
            time.sleep(SLEEP_BETWEEN_CALLS)

    # ── Final save ────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    log.info(f"\nSaved {len(results_df)} rows → {output_path}")
    _print_summary(results_df)
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/questions_v5.csv")
    parser.add_argument("--output",  default="results/results_v2.csv")
    parser.add_argument("--sample",  type=int, default=None,
                        help="Run on N rows only (smoke test)")
    parser.add_argument("--models",  nargs="+", default=None,
                        help="Specific models e.g. --models gemini-1.5-flash qwen-2.5-72b")
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        sample_n=args.sample,
        models=args.models,
    )