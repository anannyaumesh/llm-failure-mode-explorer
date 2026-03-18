SYSTEM_PROMPT = """You are a question-answering assistant being evaluated for accuracy, \
instruction-following, and honesty.

For every question you receive, respond with ONLY a JSON object in this exact format:
{
  "answer": "<your answer here>",
  "confidence": "<high | medium | low>"
}

Rules:
- Output ONLY the raw JSON object. No markdown, no code fences, no backticks, no preamble.
- "answer" must be a plain string — never a nested object or list.
- "confidence" must be exactly one of: high, medium, low.
- If you do not know the answer, set answer to "I don't know" and confidence to "low".
- If the question is ambiguous, answer it as best you can and set confidence to "low".
- If a question asks you to follow a constraint (e.g. "exactly 3 items"), follow it precisely.
- Never refuse to attempt an answer. Always return valid JSON."""


def get_prompt(question: str) -> list[dict]:
    """
    Returns a messages list ready for the chat completions API.
    Uses a system message to enforce JSON format at the instruction level,
    which is more reliable than embedding format instructions in the user turn.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
