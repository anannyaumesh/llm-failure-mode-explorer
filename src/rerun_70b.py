"""
rerun_70b.py
------------
Re-runs ONLY the 69 failed Llama-3.3-70B rows (IDs 363-432).
Run this the morning after the full run once the Groq TPD resets at 00:00 UTC.

Usage:
    python src/rerun_70b.py

Output:
    results/results_final.csv   — merged complete dataset (1728 rows, all valid)

Token estimate for this re-run: ~18,000 tokens (well within 100k daily limit).
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
from evaluator import evaluate_response

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/rerun_70b.log"),
    ]
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

FAILED_IDS = [
    363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376,
    377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390,
    391, 392, 393, 394, 395, 396, 397, 398, 400, 401, 402, 403, 404, 405,
    406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419,
    420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432,
]

MAX_TOKENS  = 512
MAX_RETRIES = 3
RETRY_DELAY = 3

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# ── Parser (same as run_eval.py) ─────────────────────────────────────────────

def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
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
    cleaned = strip_markdown_fences(raw)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed, False, "ok"
        return None, True, f"JSON parsed but missing 'answer' key"
    except json.JSONDecodeError:
        pass
    recovered = attempt_partial_json_recovery(cleaned)
    if recovered:
        return recovered, False, "recovered_from_truncated_json"
    return None, True, f"Unparseable output: {raw[:80]}"


def safe_str(value) -> str:
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).strip() if value is not None else ""


def call_model(messages: list[dict]) -> tuple[str, str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content.strip(), ""
        except Exception as e:
            err = str(e)
            if "rate_limit_exceeded" in err and "tokens per day" in err:
                # TPD still not reset — tell user clearly
                log.error("DAILY TOKEN LIMIT STILL ACTIVE. Wait until 00:00 UTC and retry.")
                raise SystemExit(1)
            if attempt < MAX_RETRIES:
                log.warning(f"  Attempt {attempt} failed: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                return "", str(e)
    return "", "Max retries exceeded"


# ── Main ─────────────────────────────────────────────────────────────────────

def run_rerun():
    os.makedirs("results", exist_ok=True)

    # Load full dataset to get the questions for failed IDs
    df_dataset = pd.read_csv("data/questions_v4.csv")
    df_failed  = df_dataset[df_dataset["id"].isin(FAILED_IDS)].copy()

    log.info(f"Re-running {len(df_failed)} rows for llama-3.3-70b")
    log.info(f"Categories: {dict(df_failed['type'].value_counts())}")
    log.info("Estimated tokens: ~18,000 (well within 100k daily limit)")

    new_results = []
    total = len(df_failed)

    for i, (_, row) in enumerate(df_failed.iterrows(), 1):
        row_id      = safe_str(row.get("id", ""))
        question    = safe_str(row.get("question", ""))
        category    = safe_str(row.get("type", ""))
        subcategory = safe_str(row.get("subcategory", ""))
        difficulty  = safe_str(row.get("difficulty", ""))
        expected    = safe_str(row.get("expected_answer", ""))

        log.info(f"  [{i}/{total}] id={row_id} cat={category}/{subcategory}")

        messages = get_prompt(question)
        raw_output, api_error = call_model(messages)

        if api_error:
            log.error(f"    API error: {api_error}")
            new_results.append({
                "model": "llama-3.3-70b", "id": row_id, "question": question,
                "category": category, "subcategory": subcategory,
                "difficulty": difficulty, "expected": expected,
                "raw_output": "", "answer": "", "confidence": "",
                "format_error": True, "parse_note": api_error,
                "constraint_violation": False, "constraint_details": "",
                "hallucination": False, "hallucination_details": "",
                "incomplete": True, "incomplete_details": f"API error: {api_error}",
                "sycophancy": False, "sycophancy_details": "",
                "injection_success": False, "system_prompt_leak": False,
                "injection_details": "", "error": api_error,
            })
            continue

        parsed, format_error, parse_note = parse_model_output(raw_output)

        if format_error:
            log.warning(f"    Parse failed: {parse_note}")
            answer = confidence = ""
        else:
            answer     = safe_str(parsed.get("answer", ""))
            confidence = safe_str(parsed.get("confidence", ""))

        eval_metrics = evaluate_response(
            answer=answer, expected=expected,
            category=category, subcategory=subcategory,
            question=question, row_id=row_id,
        )

        for flag, detail in [
            ("constraint_violation", eval_metrics.get("constraint_details","")),
            ("hallucination",        eval_metrics.get("hallucination_details","")),
            ("sycophancy",           eval_metrics.get("sycophancy_details","")),
            ("injection_success",    eval_metrics.get("injection_details","")),
            ("system_prompt_leak",   eval_metrics.get("injection_details","")),
        ]:
            if eval_metrics.get(flag):
                log.info(f"    {flag.upper()}: {detail}")

        new_results.append({
            "model": "llama-3.3-70b", "id": row_id, "question": question,
            "category": category, "subcategory": subcategory,
            "difficulty": difficulty, "expected": expected,
            "raw_output": raw_output, "answer": answer, "confidence": confidence,
            "format_error": format_error, "parse_note": parse_note,
            **eval_metrics, "error": "",
        })

    # ── Merge with existing results ───────────────────────────────────────────
    log.info("\nMerging with existing results.csv...")
    df_old     = pd.read_csv("results/results.csv")
    df_new     = pd.DataFrame(new_results)

    # Drop the failed 70B rows from old results
    df_old_clean = df_old[
        ~((df_old["model"] == "llama-3.3-70b") &
          (df_old["id"].isin(FAILED_IDS)))
    ].copy()

    # Add new signal columns to old data if missing (backward compatibility)
    for col in ["injection_success", "system_prompt_leak", "injection_details"]:
        if col not in df_old_clean.columns:
            df_old_clean[col] = False if "success" in col or "leak" in col else ""

    df_final = pd.concat([df_old_clean, df_new], ignore_index=True)
    df_final = df_final.sort_values(["model", "id"]).reset_index(drop=True)

    out_path = "results/results_final.csv"
    df_final.to_csv(out_path, index=False)

    log.info(f"Saved {len(df_final)} rows to {out_path}")
    log.info(f"  Old clean rows: {len(df_old_clean)}")
    log.info(f"  New 70B rows:   {len(df_new)}")

    # Quick summary of new results
    print("\n=== RE-RUN SUMMARY (llama-3.3-70b new rows) ===")
    flags = ["format_error","constraint_violation","hallucination",
             "incomplete","sycophancy","injection_success","system_prompt_leak"]
    n = len(df_new)
    for flag in flags:
        count = df_new[flag].astype(str).str.lower().eq("true").sum()
        print(f"  {flag}: {count}/{n}")

    return df_final


if __name__ == "__main__":
    run_rerun()
