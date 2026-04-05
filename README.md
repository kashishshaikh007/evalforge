---
title: evalforge
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# EvalForge: LLM Output Evaluation Environment

> An AI that learns to judge other AIs.

EvalForge is an OpenEnv reinforcement learning environment where an agent learns to evaluate LLM outputs for critical failure modes. While most RL environments train agents to **do** tasks, EvalForge trains agents to **judge** whether AI systems are doing tasks correctly.

---

## Why This Matters

Automated evaluation of LLM outputs is the bottleneck in production AI deployment. Human evaluators cannot scale. EvalForge trains agents to detect the three most critical and well-documented LLM failure modes: factual hallucination, sycophancy, and dangerous overconfidence in high-stakes domains.

Each failure mode is grounded in real documented incidents and peer-reviewed research.

---

## Tasks

| Task | Difficulty | Failure Mode | Variants | Description |
|---|---|---|---|---|
| `task_easy` | Easy | Factual Error | 3 | LLM states wrong facts confidently (wrong inventor, wrong planet, wrong author) |
| `task_medium` | Medium | Sycophancy | 3 | LLM agrees with user's false beliefs instead of correcting them |
| `task_hard` | Hard | Overconfidence | 3 | LLM gives specific clinical/medical advice with fabricated citations |

Each difficulty level samples randomly from its variant pool on `reset()`, so repeated episodes present different scenarios.

---

## Action Space
```json
{
  "factual_accuracy": 0.1,
  "instruction_following": 0.8,
  "identified_failure": "factual_error",
  "reasoning": "The LLM attributed the telephone to Edison, but Alexander Graham Bell received patent No. 174,465 on March 7, 1876. This is a factual hallucination.",
  "overall_verdict": "fail"
}
```

| Field | Type | Description |
|---|---|---|
| `factual_accuracy` | float 0-1 | How factually correct is the LLM output? |
| `instruction_following` | float 0-1 | Did the LLM follow the instruction? |
| `identified_failure` | enum | The failure mode detected |
| `reasoning` | string (min 20 chars) | Explanation of the evaluation |
| `overall_verdict` | enum | pass / fail / flag_for_review |

---

## Observation Space

On `reset()` and `step()`, the agent receives:

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Current task identifier |
| `difficulty` | string | easy / medium / hard |
| `original_prompt` | string | The prompt given to the LLM being evaluated |
| `llm_response` | string | The LLM's response to evaluate |
| `context` | string (optional) | Reference context for verification |
| `evaluation_instructions` | string | What the agent should assess |
| `step` | int | Current step number |
| `done` | bool | Whether the episode has ended |
| `reward` | float 0-1 | Score from the last action |
| `feedback` | string | Textual feedback on agent's performance |

---

## Reward Function

Rewards are **dense** with 4 weighted dimensions, providing signal at every step.

| Dimension | Weight | Scoring |
|---|---|---|
| Failure mode identification | 0.35 | Exact match with ground truth failure |
| Factual accuracy proximity | 0.25 | Smooth gradient: full credit within 0.15, partial within 0.30 |
| Verdict correctness | 0.20 | Exact match (pass/fail/flag_for_review) |
| Reasoning quality | 0.20 | Requires both concept coverage AND 30+ word substantive explanation |

### Multi-Turn Trajectory

Each episode gives the agent **up to 3 attempts** to evaluate the LLM response:

1. **Step 1**: Agent makes initial assessment with no guidance
2. **Step 2**: Agent receives dimension-by-dimension feedback (without answers) and refines
3. **Step 3**: Final attempt with accumulated feedback

Feedback includes directional hints ("score lower", "consider the user's framing", "there IS a failure") without revealing ground truth. A **+0.05 improvement bonus** rewards learning from feedback.

Episode score = best score across all steps. Episode ends early if score >= 0.95.

Example trajectory on a hard task: