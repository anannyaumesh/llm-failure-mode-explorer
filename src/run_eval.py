import pandas as pd
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from prompts import get_prompt
from evaluator import evaluate_response

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Models to compare
models = ["gpt-4o-mini", "gpt-4o"]

# Load dataset
df = pd.read_csv("data/questions.csv")

results = []

# Loop through models and questions
for model_name in models:
    print(f"\nRunning evaluation for model: {model_name}\n")

    for _, row in df.iterrows():
        question = row["question"]
        q_type = row["type"]
        expected = str(row["expected_answer"])

        prompt = get_prompt(question)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            output_text = response.choices[0].message.content.strip()

            # Try parsing JSON
            try:
                parsed = json.loads(output_text)
                answer = parsed.get("answer", "")
                confidence = parsed.get("confidence", "")
                format_error = False
            except:
                answer = ""
                confidence = ""
                format_error = True

            # Evaluate response
            eval_metrics = evaluate_response(answer, expected, question)

            results.append({
                "model": model_name,
                "question": question,
                "type": q_type,
                "expected": expected,
                "raw_output": output_text,
                "answer": answer,
                "confidence": confidence,
                "format_error": format_error,
                **eval_metrics
            })

        except Exception as e:
            results.append({
                "model": model_name,
                "question": question,
                "type": q_type,
                "error": str(e)
            })

# Save results
os.makedirs("results", exist_ok=True)
pd.DataFrame(results).to_csv("results/results.csv", index=False)

print("\nEvaluation complete. Results saved to results/results.csv\n")