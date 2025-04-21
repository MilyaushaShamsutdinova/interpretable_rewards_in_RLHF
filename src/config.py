# Model configuration
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
JUDGE_MODEL_NAME = BASE_MODEL_NAME

# Dataset configuration
DATASET_NAME = "nvidia/HelpSteer2"
PROMPT_COLUMN = "prompt"
RESPONSE_COLUMN = "response"
EXPLAINABLE_DIMENSIONS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
