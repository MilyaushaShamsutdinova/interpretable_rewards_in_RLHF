import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import logging
import os
from huggingface_hub import HfApi, create_repo, ModelCard, ModelCardData
from huggingface_hub.utils import HfHubHTTPError
from src import config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name, adapter_path=None, load_4bit=True, add_lora=False, mode="causal"):
    """Loads model and tokenizer, optionally adds NEW LoRA layers (if add_lora=True and adapter_path=None)."""
    logger.info(f"Loading model: {model_name} for mode: {mode}")
    bnb_config = None
    if load_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        logger.info("4-bit quantization enabled.")
    else:
        logger.info("Loading in default precision (float16/bfloat16).")

    load_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "quantization_config": bnb_config,
        "torch_dtype": torch.bfloat16 if bnb_config else torch.float16,
        "device_map": "auto",
        "trust_remote_code": True
    }

    if mode == "causal":
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        task_type = "CAUSAL_LM"
    elif mode == "reward":
        model = AutoModelForSequenceClassification.from_pretrained(num_labels=1, **load_kwargs)
        task_type = "SEQ_CLS"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        logger.info("Set pad_token to eos_token.")

    if load_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        logger.info("Prepared model for 4-bit training.")

    # Adapter handling
    if adapter_path and os.path.isdir(adapter_path):
        logger.info(f"Loading PEFT adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        logger.info("PEFT adapter loaded successfully.")
    elif add_lora and config.ADAPTER_CLASS == "lora":
        logger.info("Initializing new LoRA layers for training.")
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            target_modules=config.LORA_TARGET_MODULES,
            bias="none",
            task_type=task_type,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif add_lora:
        logger.warning(f"add_lora=True but adapter class '{config.ADAPTER_CLASS}' not 'lora' or not recognized.")

    logger.info("Model and tokenizer loading complete.")
    return model, tokenizer

def get_device():
    return torch.device(config.DEVICE)

# def push_to_hub(model, tokenizer, repo_id, commit_message="Push trained adapter"):
#     """Pushes PEFT adapter and tokenizer files to the Hugging Face Hub."""
#     if not config.HF_TOKEN:
#         logger.error("Hugging Face Hub token not found in config or .env. Cannot push.")
#         return
#     try:
#         api = HfApi(token=config.HF_TOKEN)
#         create_repo(repo_id, repo_type="model", exist_ok=True, token=config.HF_TOKEN)
#         logger.info(f"Repo '{repo_id}' created or already exists.")

#         # Push adapter and tokenizer
#         model.push_to_hub(repo_id, token=config.HF_TOKEN, commit_message=commit_message)
#         tokenizer.push_to_hub(repo_id, token=config.HF_TOKEN, commit_message="Push tokenizer")

#         # Create a basic model card
#         card_content = f"""
# Model trained with explainable RLHF ({repo_id.split('-')[-2].upper()})
# This repository contains LoRA adapters trained using {repo_id.split('-')[-2].upper()} with an explainable reward signal derived from {config.JUDGE_MODEL_NAME} scoring multiple dimensions ({', '.join(config.EXPLAINABLE_DIMENSIONS)}).
# The base model is {config.BASE_MODEL_NAME}.
# """
#         card = ModelCard(card_content)
#         card.push_to_hub(repo_id, token=config.HF_TOKEN)

#         logger.info(f"Successfully pushed adapter, tokenizer, and model card to '{repo_id}'")

#     except HfHubHTTPError as e:
#         logger.error(f"HTTP Error pushing to Hub: {e}")
#     except Exception as e:
#         logger.error(f"An error occurred pushing to Hub: {e}")

