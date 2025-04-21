import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import re
import logging
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
import time
from src import config, utils

logger = logging.getLogger(__name__)


class ExplainableRewardModel:
    """
    Calculates rewards based on an LLM-as-a-Judge scoring multiple dimensions.
    """
    def __init__(self, model_name=config.JUDGE_MODEL_NAME, device=None):
        self.model_name = model_name
        self.device = device if device else utils.get_device()
        logger.info(f"Initializing explainable RM using judge: {self.model_name} on device {self.device}")

        # Load judge model
        self.model, self.tokenizer = utils.load_model_and_tokenizer(
            self.model_name,
            adapter_path=None,
            load_4bit=True,
            add_lora=False,
            mode='causal',
        )
        self.model.eval()
        logger.info("Judge model loaded and set to evaluation mode.")

        # Generation config for potentially more deterministic scoring
        self.generation_config = GenerationConfig(
            max_new_tokens=10,
            min_new_tokens=1,
            # temperature=0.01,
            # top_k=1,
            # top_p=0.9,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        logger.info("Explainable RM initialized.")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5), reraise=True)
    def _get_score_from_judge(self, input_text: str) -> int:
        """
        Generates score using the judge model with retry mechanism. Handles potential errors during generation or parsing.
        """
        logger.debug(f"Sending text to judge: ...{input_text[-300:]}")
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH - self.generation_config.max_new_tokens - 10,
            padding=False,
        ).to(self.device)

        generated_text = "[ERROR: Generation Failed]"
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            generated_text = self.tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            logger.debug(f"Judge raw output: '{generated_text}'")

            match = re.search(fr'\b([{config.REWARD_SCORE_MIN}-{config.REWARD_SCORE_MAX}])\b', generated_text)
            if not match:
                 match = re.search(r'\d+', generated_text)

            if match:
                score_str = match.group(0)
                try:
                    score = int(score_str)
                    score = max(config.REWARD_SCORE_MIN, min(config.REWARD_SCORE_MAX, score))
                    logger.debug(f"Parsed score: {score}")
                    return score
                except ValueError:
                    logger.warning(f"Could not convert matched string '{score_str}' to int. Judge output: '{generated_text}'")
                    raise ValueError("Score parsing failed (conversion).")
            else:
                logger.warning(f"Could not parse score (no digit found) from judge output: '{generated_text}'. Retrying...")
                raise ValueError("Score parsing failed (no digit).")

        except Exception as e:
             logger.error(f"Exception during judge scoring or parsing: {e}. Judge Output: '{generated_text}'. Input: ...{input_text[-300:]}")
             raise

    def get_dimension_score(self, prompt: str, response: str, dimension: str) -> float:
        """Gets the normalized score (0-1) for a single dimension."""
        if dimension not in config.DIMENSION_DESCRIPTIONS:
            logger.error(f"Dimension '{dimension}' not found in descriptions.")
            return 0.0

        dimension_description = config.DIMENSION_DESCRIPTIONS[dimension]
        input_text = config.REWARD_JUDGE_PROMPT_TEMPLATE.format(
            dimension=dimension,
            dimension_description=dimension_description,
            prompt=prompt,
            response=response
        )
        try:
            score = self._get_score_from_judge(input_text)
            normalized_score = (score - config.REWARD_SCORE_MIN) / (config.REWARD_SCORE_MAX - config.REWARD_SCORE_MIN)
            return normalized_score
        except RetryError as e:
            logger.error(f"Failed to get score for dimension '{dimension}' after multiple retries: {e}. Assigning neutral normalized score.")
            return 0.5
        except Exception as e:
             logger.error(f"Unexpected error getting score for dimension '{dimension}': {e}. Assigning neutral normalized score.")
             return 0.5

    def get_reward(self, prompts: list[str], responses: list[str], dimensions_to_use: list[str] = None) -> torch.Tensor:
        """
        Calculates the aggregated reward for a batch of prompts and responses.
        Processes samples sequentially within the batch.
        Returns results as a tensor on the reward model's device.
        """
        if not prompts or not responses or len(prompts) != len(responses):
            logger.error("Invalid input: prompts and responses lists must be non-empty and have the same length.")
            return torch.tensor([], dtype=torch.float32).to(self.device)

        if dimensions_to_use is None:
            dimensions_to_use = config.EXPLAINABLE_DIMENSIONS

        if not dimensions_to_use:
            logger.warning("No dimensions specified to calculate reward. Returning zero reward.")
            return torch.zeros(len(prompts), dtype=torch.float32).to(self.device)

        batch_rewards = []
        for i in range(len(prompts)):
            prompt = prompts[i]
            response = responses[i]

            # Handle potential None or empty strings
            if not isinstance(prompt, str) or not isinstance(response, str) or not prompt or not response:
                logger.warning(f"Skipping invalid prompt or response at index {i}. Prompt: '{prompt}', Response: '{response}'. Assigning zero reward.")
                batch_rewards.append(0.0)
                continue

            dimension_scores = []
            logger.debug(f"Scoring response {i+1}/{len(prompts)} for prompt: '{prompt[:100]}...'")
            for dim in dimensions_to_use:
                score = self.get_dimension_score(prompt, response, dim)
                dimension_scores.append(score)
                logger.debug(f"  Dimension '{dim}': {score:.3f}")

            if not dimension_scores:
                logger.warning(f"No valid dimension scores obtained for response {i+1}. Assigning zero reward.")
                final_reward = 0.0
            elif config.REWARD_AGGREGATION == "mean":
                final_reward = sum(dimension_scores) / len(dimension_scores)
            else:
                logger.error(f"Unknown reward aggregation method: {config.REWARD_AGGREGATION}. Assigning zero reward.")
                final_reward = 0.0

            batch_rewards.append(final_reward)
            logger.debug(f"  Aggregated Reward: {final_reward:.4f}")

        return torch.tensor(batch_rewards, dtype=torch.float32).to(self.device)


if __name__ == "__main__":
    reward_model = ExplainableRewardModel()

    # Example from dataset HelpSteer2
    test_prompt = "in a chat, can I feed chatGPT informations so it can answer questions better?"
    test_response_good = """Yes, you can provide ChatGPT with information and context to help it answer questions more accurately and relevantly. Here are a few tips:\n
1. Provide specific details: If you're asking a question about a particular topic, provide specific details and context to help ChatGPT understand the context and provide more accurate information.\n
2. Use clear and concise language: Use clear and concise language when providing information or asking questions to help ChatGPT understand your intent and provide relevant answers.\n
3. Provide examples: Providing examples can help ChatGPT better understand the context and provide more accurate and relevant answers.\n
4. Be specific: Instead of asking general questions, be specific and provide details about what you're looking for. This will help ChatGPT provide more accurate and relevant answers.\n
5. Provide background information: If you're asking a question about a particular topic, provide background information to help ChatGPT understand the context and provide more accurate answers.\n
Overall, providing clear and concise information and context can help ChatGPT provide more accurate and relevant answers to your questions."""
    test_response_bad = "Yes, you can feed ChatGPT information to help it answer questions better. This is called \"training\" the model, and it involves providing it with examples of good and bad responses to questions. You can also use tools like GPT-J's \"Teacher\" mode to provide more detailed feedback on its responses."
    test_response_empty = ""

    print("\n--- Testing Good Response ---")
    reward_good = reward_model.get_reward([test_prompt], [test_response_good])
    print(f"Final Reward (Good): {reward_good.item():.4f}")

    print("\n--- Testing Bad Response ---")
    reward_bad = reward_model.get_reward([test_prompt], [test_response_bad])
    print(f"Final Reward (Bad): {reward_bad.item():.4f}")

    print("\n--- Testing Batch & Empty Response ---")
    rewards_batch = reward_model.get_reward(
        [test_prompt, test_prompt, "What is 2+2?"],
        [test_response_good, test_response_bad, test_response_empty]
    )
    print(f"Final Rewards (Batch): {rewards_batch.tolist()}")

    print("\n--- Testing Specific Dimension ---")
    correctness_score = reward_model.get_dimension_score(test_prompt, test_response_bad, "correctness")
    print(f"Correctness Score (Bad Response): {correctness_score:.4f}")

    helpfulness_score = reward_model.get_dimension_score(test_prompt, test_response_good, "helpfulness")
    print(f"Helpfulness Score (Good Response): {helpfulness_score:.4f}")

