from sentence_transformers import SentenceTransformer, util
import re

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')


def evaluate_response(answer, expected, question):
    constraint_violation = False
    hallucination = False
    incomplete = False

    # --- Semantic similarity (better hallucination detection) ---
    if expected and expected != "nan":
        try:
            emb1 = model.encode(answer, convert_to_tensor=True)
            emb2 = model.encode(expected, convert_to_tensor=True)

            similarity = util.cos_sim(emb1, emb2).item()

            if similarity < 0.6:
                hallucination = True
        except:
            hallucination = True

    # --- Constraint: "exactly N" ---
    if "exactly" in question.lower():
        numbers = re.findall(r'\d+', question)
        if numbers:
            expected_count = int(numbers[0])

            items = answer.split("\n")
            items = [i for i in items if i.strip() != ""]

            if len(items) != expected_count:
                constraint_violation = True

    # --- Incomplete ---
    if len(answer.strip()) < 5:
        incomplete = True

    return {
        "constraint_violation": constraint_violation,
        "hallucination": hallucination,
        "incomplete": incomplete
    }