from dotenv import load_dotenv
import torch
import os

load_dotenv()

SEED=42
HF_TOKEN = os.getenv('HF_TOKEN')
WANDB_API = os.getenv('WANDB_API')


# Model config
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
JUDGE_MODEL_NAME = BASE_MODEL_NAME
ADAPTER_CLASS = "lora"
MAX_SEQ_LENGTH=256


# Dataset config
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

# EVAL_DATASET_NAME = "Anthropic/hh-rlhf"  # ?


# Reward model config
REWARD_JUDGE_PROMPT_TEMPLATE = f"""Please evaluate the following response based ONLY on the dimension of '{{dimension}}'. Use the provided definition and scoring guide. Respond with a single integer score from {REWARD_SCORE_MIN} to {REWARD_SCORE_MAX}, where {REWARD_SCORE_MIN} is the lowest quality and {REWARD_SCORE_MAX} is the highest quality for this specific dimension. Do not provide any explanation, just the score.

Dimension: {{dimension}}

Definition and Scoring Guide:
{{dimension_description}}

Prompt:
{{prompt}}

Response:
{{response}}

Score ({REWARD_SCORE_MIN}-{REWARD_SCORE_MAX}):"""


# Training config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results"
LOG_WITH = "wandb"
SAVE_FREQ = 100

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# RL config
RL_BATCH_SIZE = 1
RL_MAX_NEW_TOKENS = MAX_SEQ_LENGTH
RL_TEMPERATURE = 0.2
RL_TOP_K = 0.0
RL_TOP_P = 1.0
KL_PENALTY_BETA = 0.1

# PPO config
PPO_MINI_BATCH_SIZE = 1
PPO_GRAD_ACCUMULATION_PPO = 4
PPO_NUM_EPOCHS=1
PPO_LEARNING_RATE = 1e-5
PPO_WARMUP_RATIO = 0.1


# REINFORCE config
REINFORCE_LEARNING_RATE = 5e-6
REINFORCE_GRAD_ACCUMULATION = 2
REINFORCE_BASELINE_TYPE = "moving_average"
REINFORCE_BASELINE_ALPHA = 0.99
REINFORCE_NUM_EPOCHS = 1


# HF Hub config
HF_HUB_USERNAME = "MilyaShams"
PPO_HUB_REPO_ID = f"{HF_HUB_USERNAME}/{BASE_MODEL_NAME.split('/')[-1]}-ppo-explainable"
REINFORCE_HUB_REPO_ID = f"{HF_HUB_USERNAME}/{BASE_MODEL_NAME.split('/')[-1]}-reinforce-explainable"
# REWARD_MODEL_HUB_REPO_ID = f"{HF_HUB_USERNAME}/{BASE_MODEL_NAME.split('/')[-1]}-explainable-RM"


# Evaluation config
EVAL_BATCH_SIZE = 8
EVAL_MAX_SAMPLES = 500
