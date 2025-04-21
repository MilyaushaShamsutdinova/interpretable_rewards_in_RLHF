from dotenv import load_dotenv
import torch
import os

load_dotenv()

SEED=42
HF_TOKEN = os.getenv('HF_TOKEN')


# Model config
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
JUDGE_MODEL_NAME = BASE_MODEL_NAME
ADAPTER_CLASS = "lora"
MAX_SEQ_LENGTH=2048


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

# EVAL_DATASET_NAME = "Anthropic/hh-rlhf"


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
# how to use template
# print(REWARD_JUDGE_PROMPT_TEMPLATE.format(dimension="helpfulness", dimension_description=DIMENSION_DESCRIPTIONS["helpfulness"], prompt="Test prompt", response="Test response"))


# Training config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results"
LOG_WITH = "wandb"
# MAX_RL_STEPS = 200 # Define total optimization steps for RL training
# SAVE_FREQ = 50 # Save checkpoints every N steps

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# RL config
RL_BATCH_SIZE = 8
RL_MAX_NEW_TOKENS = 128
RL_TEMPERATURE = 0.7
RL_TOP_K = 50
RL_TOP_P = 0.9
KL_PENALTY_BETA = 0.1

# PPO config
PPO_LEARNING_RATE = 1e-5
PPO_MINI_BATCH_SIZE = 4
PPO_GRAD_ACCUMULATION_PPO = 2
PPO_CLIP_EPSILON = 0.2
PPO_VALUE_CLIP = 0.2
PPO_GAMMA = 1.0
PPO_LAM = 0.95
PPO_EPOCHS_PER_RL_STEP = 4

# REINFORCE config
REINFORCE_LEARNING_RATE = 5e-6
REINFORCE_GRAD_ACCUMULATION = 2
REINFORCE_BASELINE_TYPE = "moving_average"
REINFORCE_BASELINE_ALPHA = 0.99


# HF Hub config
HF_HUB_USERNAME = "MilyaShams"
HF_TOKEN = os.getenv("HF_TOKEN")
PPO_HUB_REPO_ID = f"{HF_HUB_USERNAME}/{BASE_MODEL_NAME.split('/')[-1]}-ppo-explainable"
REINFORCE_HUB_REPO_ID = f"{HF_HUB_USERNAME}/{BASE_MODEL_NAME.split('/')[-1]}-reinforce-explainable"
# REWARD_MODEL_HUB_REPO_ID = f"{HF_HUB_USERNAME}/{BASE_MODEL_NAME.split('/')[-1]}-explainable-RM"


# Evaluation config
EVAL_BATCH_SIZE = 8
EVAL_MAX_SAMPLES = 500
