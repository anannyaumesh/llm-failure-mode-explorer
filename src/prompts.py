def get_prompt(question):
    return f"""
You are an AI assistant. Follow ALL instructions strictly.

Return your answer ONLY in the following JSON format:
{{
  "answer": "...",
  "confidence": "high" | "medium" | "low"
}}

STRICT RULES:
- Do NOT include any text outside JSON
- Follow all constraints in the question exactly
- If unsure, say "I don't know" in the answer field
- Do NOT explain your reasoning

Question: {question}
"""
