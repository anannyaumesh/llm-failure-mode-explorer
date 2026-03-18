# Evaluating Failure Modes in LLM-Based Question Answering Systems

## Goal

To evaluate how modern large language models behave under realistic usage conditions, with a focus on:

- Structured output reliability
- Instruction-following
- Constraint adherence
- Cross-model behavioral differences

---

## Method

- Designed a dataset covering diverse query types:
  - factual, reasoning, constraint-based, ambiguous, and format-heavy prompts
- Forced models to return strictly structured JSON outputs
- Evaluated responses using:
  - semantic similarity (for correctness)
  - rule-based constraint checks
- Compared performance across multiple models

---

## Models Evaluated

- GPT-4o-mini
- GPT-4o

---

## Key Results

- Near-zero format errors across both models
- Persistent constraint violations (~20–30%)
- Reduced but non-zero hallucination rates using semantic evaluation
- Observable differences in model behavior across task types

---

## Key Insights

### 1. Structured outputs ensure format compliance, not correctness

Both models consistently returned valid JSON, but semantic correctness and constraint adherence remained inconsistent.

### 2. Instruction-following is a primary failure mode

Even with explicit constraints (e.g., “exactly N items”), both models failed to comply reliably.

### 3. Errors are subtle rather than obvious

With semantic evaluation, failures appear as small deviations or partial compliance rather than blatant hallucinations.

### 4. Constraint-heavy prompts reduce reliability

Combining multiple constraints (format + count + brevity) significantly increased failure rates.

### 5. Model behavior differs across tasks

While both models performed well overall, differences emerged in reasoning accuracy and consistency of instruction-following.  
One model showed slightly better numerical reasoning, while the other demonstrated more consistent adherence to output structure, indicating that performance varies by task rather than overall capability.

### 6. Evaluation design significantly impacts conclusions

Naive heuristics and semantic similarity thresholds can misclassify correct responses as failures, particularly for:

- unit-based answers (e.g., "100°C" vs "100")
- abstentions (e.g., "I don't know")

This highlights the importance of carefully designed evaluation frameworks when assessing LLM behavior.

---

## Observed Failure Types

- Constraint violations (incorrect number of items, ignored limits)
- Partial instruction compliance
- Overconfident incorrect answers
- Subtle semantic mismatches

---

## Example Failures

Example 1:

Prompt:  
"Give exactly 3 uses of AI in healthcare."

Model Output:  
Returned 4 items → constraint violation

Example 2:

Prompt:  
"What is (23 \* 47) + 18?"

Model Output:  
Returned incorrect numerical result → reasoning failure

---

## Limitations

- Evaluation relies on heuristic rules and semantic similarity thresholds
- Correct answers with different representations (e.g., units or phrasing) may be misclassified
- Abstentions may be incorrectly flagged as hallucinations
- Dataset size is limited

---

## Project Structure

- `data/` – evaluation dataset
- `src/` – prompting and evaluation pipeline
- `results/` – model outputs (excluded from repository)
- `analysis.ipynb` – analysis and insights

---

## Future Work

- Expand evaluation to additional models (e.g., Mistral, Claude)
- Improve scoring using task-specific metrics
- Extend framework to retrieval-augmented (RAG) systems
- Build automated evaluation benchmarks
