import pandas as pd
import json
import os
import re
import time
import logging
from dotenv import load_dotenv
from openai import OpenAI

from prompts import get_prompt
from evaluator import evaluate_response

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/eval.log"),
    ]
)
log = logging.getLogger(__name__)

MAX_TOKENS = 512

# Retry config
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Rate limiting (Fix 6)
SLEEP_BETWEEN_CALLS = 0.3  # seconds

# Checkpoint saving (Fix 5)
CHECKPOINT_EVERY = 50
CHECKPOINT_PATH = "results/results_partial.csv"

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

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
#─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "gpt-4o-mini":            (openai_client,  "gpt-4o-mini"),
    "mistral-small":          (mistral_client, "mistral-small-latest"),
    "llama-3.1-8b":           (groq_client,    "llama-3.1-8b-instant"),
    "llama-3.3-70b":          (groq_client,    "llama-3.3-70b-versatile"),
}

MODELS_TO_RUN = list(MODEL_REGISTRY.keys())

# ─────────────────────────────────────────────────────────────────────────────
# JSON parser 
# ─────────────────────────────────────────────────────────────────────────────

def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    # remove opening fence (```json or ```)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    # remove closing fence
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def attempt_partial_json_recovery(text: str) -> dict | None:
    answer_match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    conf_match   = re.search(r'"confidence"\s*:\s*"(high|medium|low)"', text)
    if answer_match:
        return {
            "answer":     answer_match.group(1),
            "confidence": conf_match.group(1) if conf_match else "low",
            "_recovered": True,
        }
    return None


def parse_model_output(raw: str) -> tuple[dict | None, bool, str]:
    """
    Parse model output into a structured dict.
    Returns (parsed_dict, format_error, parse_note).
    
    format_error=True means we could not extract a valid answer at all.
    parse_note gives a human-readable explanation for logging.
    """
    cleaned = strip_markdown_fences(raw)  # Bug B1

    # primary parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed, False, "ok"
        return None, True, f"JSON parsed but missing 'answer' key: {list(parsed.keys())}"
    except json.JSONDecodeError:
        pass

    # partial recovery (Bug B2 — truncated Groq response)
    recovered = attempt_partial_json_recovery(cleaned)
    if recovered:
        return recovered, False, "recovered_from_truncated_json"

    return None, True, f"Unparseable output: {raw[:80]}"


# ─────────────────────────────────────────────────────────────────────────────
# API call with retry
# ─────────────────────────────────────────────────────────────────────────────

def call_model(model_name: str, messages: list[dict]) -> tuple[str, str]:
    """
    Call the model API with retry on transient errors.
    Returns (output_text, error_string).
    error_string is "" on success.
    """
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
            if attempt < MAX_RETRIES:
                log.warning(f"  [{model_name}] attempt {attempt} failed: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                return "", str(e)

    return "", "Max retries exceeded"


# ─────────────────────────────────────────────────────────────────────────────
# Safe row field accessor 
# ─────────────────────────────────────────────────────────────────────────────

def safe_str(value) -> str:
    """
    Safely convert a dataset cell to string.
    Handles cases where pandas returns a Series or list instead of scalar. 
    """
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).strip() if value is not None else ""


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(dataset_path: str, sample_n: int | None = None, random_state: int = 42):
    """
    Run the full evaluation pipeline.
    """
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(dataset_path)

    # Ensure dataset integrity
    assert df["id"].is_unique, f"Duplicate IDs found in dataset: {df[df['id'].duplicated()]['id'].tolist()}"

    if sample_n:
        df = df.sample(sample_n, random_state=random_state)
        log.info(f"Sampled {sample_n} rows (random_state={random_state})")
    else:
        log.info(f"Running full evaluation on {len(df)} rows")

    total_calls = len(MODELS_TO_RUN) * len(df)
    log.info(f"Models: {MODELS_TO_RUN}")
    log.info(f"Total API calls: {total_calls}")

    results = []
    call_count = 0

    for model_name in MODELS_TO_RUN:
        log.info(f"\n{'='*60}")
        log.info(f"Model: {model_name}")
        log.info(f"{'='*60}")

        for _, row in df.iterrows():
            call_count += 1

            row_id       = safe_str(row.get("id", ""))
            question     = safe_str(row.get("question", ""))
            category     = safe_str(row.get("type", ""))
            subcategory  = safe_str(row.get("subcategory", ""))
            difficulty   = safe_str(row.get("difficulty", ""))
            expected     = safe_str(row.get("expected_answer", ""))

            if not question:
                log.warning(f"Skipping row {row_id}: empty question")
                continue

            log.info(f"[{call_count}/{total_calls}] id={row_id} cat={category}/{subcategory}")

            messages = get_prompt(question)

            # ─────────────────────────────────────────
            # API CALL
            # ─────────────────────────────────────────
            raw_output, api_error = call_model(model_name, messages)

            if api_error:
                log.error(f"API error: {api_error}")

                results.append({
                    "model":      model_name,
                    "id":         row_id,
                    "question":   question,
                    "category":   category,
                    "subcategory": subcategory,
                    "difficulty": difficulty,
                    "expected":   expected,
                    "raw_output": "",
                    "answer":     "",
                    "confidence": "",
                    "format_error": True,
                    "parse_note": api_error,
                    "constraint_violation": False,
                    "constraint_details": "",
                    "hallucination": False,
                    "hallucination_details": "",
                    "incomplete": True,
                    "incomplete_details": f"API error: {api_error}",
                    "sycophancy": False,
                    "sycophancy_details": "",
                    "error": api_error,
                })

                if len(results) % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
                    log.info(f"Checkpoint saved at {len(results)} rows")

                time.sleep(SLEEP_BETWEEN_CALLS)

                continue

            # ─────────────────────────────────────────
            # PARSING
            # ─────────────────────────────────────────
            parsed, format_error, parse_note = parse_model_output(raw_output)

            if format_error:
                log.warning(f"Parse failed: {parse_note}")
                answer = ""
                confidence = ""
            else:
                answer = safe_str(parsed.get("answer", ""))
                confidence = safe_str(parsed.get("confidence", ""))

            # ─────────────────────────────────────────
            # EVALUATION
            # ─────────────────────────────────────────
            eval_metrics = evaluate_response(
                answer=answer,
                expected=expected,
                category=category,
                subcategory=subcategory,
                question=question,
            )

            if eval_metrics["constraint_violation"]:
                log.info(f"CONSTRAINT VIOLATION: {eval_metrics['constraint_details']}")
            if eval_metrics["hallucination"]:
                log.info(f"HALLUCINATION: {eval_metrics['hallucination_details']}")
            if eval_metrics["sycophancy"]:
                log.info(f"SYCOPHANCY: {eval_metrics['sycophancy_details']}")

            # ─────────────────────────────────────────
            # SAVE RESULT
            # ─────────────────────────────────────────
            results.append({
                "model":       model_name,
                "id":          row_id,
                "question":    question,
                "category":    category,
                "subcategory": subcategory,
                "difficulty":  difficulty,
                "expected":    expected,
                "raw_output":  raw_output,
                "answer":      answer,
                "confidence":  confidence,
                "format_error": format_error,
                "parse_note":   parse_note,
                **eval_metrics,
                "error": "",
            })

            if len(results) % CHECKPOINT_EVERY == 0:
                pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
                log.info(f"Checkpoint saved at {len(results)} rows")

            time.sleep(SLEEP_BETWEEN_CALLS)

    # ─────────────────────────────────────────
    # FINAL SAVE
    # ─────────────────────────────────────────
    results_df = pd.DataFrame(results)
    out_path = "results/results.csv"
    results_df.to_csv(out_path, index=False)

    log.info(f"\nSaved {len(results_df)} rows to {out_path}")

    _print_summary(results_df)
    return results_df


def _print_summary(df: pd.DataFrame):
    """Print a quick per-model breakdown after the run."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    flags = ["format_error", "hallucination", "constraint_violation", "incomplete", "sycophancy"]
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        n = len(sub)
        print(f"\n{model} (n={n})")
        for flag in flags:
            if flag in sub.columns:
                count = sub[flag].astype(str).str.lower().eq("true").sum()
                print(f"  {flag}: {count}/{n} ({100*count//n}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluation(
        dataset_path="data/questions_v4.csv",
        sample_n=None,       
        random_state=42,
    )
