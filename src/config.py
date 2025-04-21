from dotenv import load_dotenv

load_dotenv()

SEED=42


# Model configuration
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
JUDGE_MODEL_NAME = BASE_MODEL_NAME
ADAPTER_CLASS = "lora"

# Dataset configuration
DATASET_NAME = "nvidia/HelpSteer2"
PROMPT_COLUMN = "prompt"
RESPONSE_COLUMN = "response"
EXPLAINABLE_DIMENSIONS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

REWARD_SCORE_MIN = 0
REWARD_SCORE_MAX = 4
REWARD_AGGREGATION = "mean"

DIMENSION_DESCRIPTIONS = {
    "helpfulness": f"Evaluates how well the response directly and completely addresses the user's prompt and provides useful information ({REWARD_SCORE_MIN}=unhelpful, {REWARD_SCORE_MAX}=very helpful).",
    "correctness": f"Evaluates the factual accuracy of the response, ensuring it includes pertinent facts without errors ({REWARD_SCORE_MIN}=major errors, {REWARD_SCORE_MAX}=accurate).",
    "coherence": f"Evaluates the consistency, clarity, logical flow, and organization of the response ({REWARD_SCORE_MIN}=incoherent, {REWARD_SCORE_MAX}=very coherent).",
    "verbosity": f"Evaluates if the response provides an appropriate amount of detail relative to the prompt, avoiding excessive or insufficient information ({REWARD_SCORE_MIN}=too little/much detail, {REWARD_SCORE_MAX}=appropriately detailed).",
    "complexity": f"Evaluates the intellectual depth or domain expertise required to write the response ({REWARD_SCORE_MIN}=basic competency, {REWARD_SCORE_MAX}=deep expertise).",
}

EVAL_DATASET_NAME = "Anthropic/hh-rlhf"


# Reward model configuration
REWARD_JUDGE_PROMPT_TEMPLATE = f"""Please evaluate the following response based ONLY on the dimension of '{{dimension}}'. Use the provided definition and scoring guide. Respond with a single integer score from {REWARD_SCORE_MIN} to {REWARD_SCORE_MAX}, where {REWARD_SCORE_MIN} is the lowest quality and {REWARD_SCORE_MAX} is the highest quality for this specific dimension. Do not provide any explanation, just the score.

Dimension: {{dimension}}

Definition and Scoring Guide:
{{dimension_description}}

Prompt:
{{prompt}}

Response:
{{response}}

Score ({REWARD_SCORE_MIN}-{REWARD_SCORE_MAX}):"""
# how to use template
# print(REWARD_JUDGE_PROMPT_TEMPLATE.format(dimension="helpfulness", dimension_description=DIMENSION_DESCRIPTIONS["helpfulness"], prompt="Test prompt", response="Test response"))
