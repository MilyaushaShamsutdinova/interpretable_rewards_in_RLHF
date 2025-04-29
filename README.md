# XAI Course project report: Interpretable Rewards in LLM Alignment via PPO vs. REINFORCE

**Author:** Milyausha Shamsutdinova \
**Date:** April 29, 2025


## Abstract

This project aimed to implement and compare two Reinforcement Learning (RL) algorithms, Proximal Policy Optimization (PPO) and REINFORCE with baseline, for aligning Large Language Models (LLMs) using an *explainable* reward signal. Inspired by the "Explainable Rewards in RLHF Using LLM-as-a-Judge" paper, the core idea was to replace the traditional opaque reward model with an LLM acting as a judge, scoring responses along multiple predefined dimensions (e.g., helpfulness, correctness). The goal was to investigate how PPO and REINFORCE differ in performance, stability, and their ability to optimize for these explicit dimensions when trained with this explainable reward mechanism. Key components, including the explainable reward model, data preprocessing, and training setups for both PPO (using the TRL library) and REINFORCE (manual implementation), were developed. However, runtime errors were encountered during the execution of both training processes, preventing the successful generation of fully trained models and subsequent comparative analysis of results. This report details the project's methodology, the implementation work completed, the challenges faced, and discusses the expected outcomes had the training been successful.


## 1. Introduction

Aligning Large Language Models (LLMs) with human values and preferences is a critical challenge in AI development. Reinforcement Learning from Human Feedback (RLHF) has emerged as a standard technique, typically involving Supervised Fine-Tuning (SFT) followed by RL optimization using a reward model trained on human preference data. However, traditional RLHF often relies on a single, scalar reward signal derived from a "black-box" reward model. This lack of transparency makes it difficult to understand *why* a model prefers one response over another and *which specific values* are being optimized during alignment (Ouyang et al., 2022; Bai et al., 2022a).

This project addresses the opacity issue by adopting the methodology proposed in "Explainable Rewards in RLHF Using LLM-as-a-Judge". Instead of training a separate reward model on human preferences, this approach leverages an existing powerful LLM (the "Judge") to evaluate generated responses along predefined, human-interpretable dimensions (e.g., helpfulness, correctness, coherence). The scores across these dimensions are then aggregated into a single reward signal for RL training. This offers greater transparency into the alignment process.

The primary goal of this project was to implement this explainable reward mechanism and use it to compare two distinct policy gradient RL algorithms:

1.  **Proximal Policy Optimization (PPO):** A popular actor-critic algorithm widely used in RLHF, known for its relative stability achieved through clipped surrogate objectives (Schulman et al., 2017).
2.  **REINFORCE with Baseline:** A fundamental policy gradient algorithm that directly optimizes the expected return, often considered simpler but potentially less stable than PPO, especially in high-variance environments (Williams, 1992). The "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" paper suggests that the typical assumptions motivating PPO might be less relevant in the LLM fine-tuning context, potentially making simpler algorithms like REINFORCE viable and efficient.

By comparing these two algorithms using the same explainable reward signal, this project aimed to shed light on their respective performance, training dynamics, and effectiveness in optimizing for specific, interpretable aspects of LLM behavior.

## 2. Methodology

### 2.1. Overall Framework: Explainable Rewards via LLM-as-a-Judge

The core of the project follows the explainable RLHF framework:

1.  **Dimension Identification:** Define key dimensions relevant for evaluating LLM responses based on the target task (e.g., helpfulness, correctness, coherence, verbosity from HelpSteer2).
2.  **LLM-as-a-Judge Scoring:** Use a capable LLM (the "Judge") to score generated responses along each identified dimension based on specific definitions and scoring rubrics provided via prompting.
3.  **Reward Aggregation:** Combine the individual dimension scores into a single scalar reward signal (e.g., using simple averaging) to be used by the RL algorithm.
4.  **RL Fine-tuning:** Use the aggregated explainable reward to fine-tune the base LLM using either PPO or REINFORCE.

This process is illustrated conceptually in Figure 1 of the project proposal.

### 2.2. Base Model and Dataset

*   **Base LLM:** `Qwen/Qwen1.5-0.5B-Instruct` was selected as the base instruction-tuned model to be aligned via RL. This model provides a strong starting point, having already undergone initial instruction tuning.
*   **Dataset for RL Prompts:** The `nvidia/HelpSteer2` dataset was used as the source of prompts for the RL training phase. Only the `prompt` column was extracted for training the PPO and REINFORCE algorithms. A subset of the training split (1000 samples) and validation split (100 samples) was used for development runs.
*   **SFT Skipping:** Given that an instruction-tuned model was used as the base, the standard SFT step was skipped to simplify the pipeline and focus on the RL comparison, following the rationale discussed during the project refinement phase. The base instruction-tuned model served as both the initial policy and the reference model for KL penalty calculation.

### 2.3. Explainable Reward Model Implementation (`src/reward.py`)

A custom Python class `ExplainableRewardModel` (inheriting from `torch.nn.Module` for compatibility with TRL) was implemented to encapsulate the LLM-as-a-Judge logic.

*   **Judge LLM:** The same `Qwen/Qwen1.5-0.5B-Instruct` model was used as the Judge LLM for simplicity, loaded in 4-bit precision.
*   **Dimensions:** The dimensions specified in `src/config.py` (`helpfulness`, `correctness`, `coherence`, `verbosity`, `complexity`) were used, with concise definitions provided to the judge.
*   **Prompting:** A specific prompt template (`REWARD_JUDGE_PROMPT_TEMPLATE`) was designed to instruct the Judge LLM to evaluate a given prompt-response pair on a single dimension and output an integer score within a defined range (0-4).
*   **Scoring & Robustness:** The `_get_score_from_judge` method handles sending the formatted prompt to the judge, generating the score, and parsing the integer score from the output using regex. The `tenacity` library was used to implement automatic retries in case of generation or parsing failures. Scores were clamped to the valid range [0, 4].
*   **Normalization & Aggregation:** Individual dimension scores were normalized to a [0, 1] range. The `get_reward` (and `forward`) method iterates through the specified dimensions for each prompt-response pair, calls `get_dimension_score`, and aggregates the normalized scores using simple averaging (`REWARD_AGGREGATION = "mean"`) to produce the final scalar reward tensor for the RL algorithms. Error handling was included for invalid inputs or scoring failures.

### 2.4. RL Algorithm Implementation

Two RL algorithms were implemented in separate Jupyter notebooks:

**2.4.1. PPO (`notebooks/ppo_training.ipynb`)**

*   **Framework:** Leveraged the Hugging Face TRL library's `PPOTrainer` and `PPOConfig`.
*   **Model:** Used `trl.models.AutoModelForCausalLMWithValueHead` to load the base policy model (`Qwen/Qwen1.5-0.5B-Instruct`) with an attached value head, applying LoRA adapters for training.
*   **Reference Model:** A separate instance of the base `Qwen/Qwen1.5-0.5B-Instruct` model (without LoRA or value head) was loaded as the reference for KL penalty calculation.
*   **Reward:** The custom `ExplainableRewardModel` instance was passed directly to the `PPOTrainer`'s `reward_model` argument.
*   **Training:** Intended to use the built-in `ppo_trainer.train()` method, which handles the generation, reward computation, and PPO update steps internally.

**2.4.2. REINFORCE with Baseline (`notebooks/reinforce_training.ipynb`)**

*   **Framework:** Implemented manually using PyTorch, following the standard REINFORCE algorithm.
*   **Model:** Used the base `Qwen/Qwen1.5-0.5B-Instruct` model with newly initialized LoRA adapters applied via PEFT.
*   **Reference Model:** Same as PPO setup.
*   **Reward:** The custom `ExplainableRewardModel` instance was used to compute rewards after generation.
*   **Baseline:** A exponential moving average baseline (`MovingAverageBaseline` class) was implemented to reduce variance in the policy gradient estimate. The baseline subtracts the average reward observed over a recent window from the current reward to calculate the advantage.
*   **Training Loop:** A manual loop was implemented:
    1.  Generate responses for a batch of prompts using the policy model (`generate_responses` function).
    2.  Calculate the sequence log-probabilities of the generated responses using the policy model (requiring a forward pass that tracks gradients) (`calculate_log_probs` function).
    3.  Calculate the KL penalty against the reference model (`calculate_kl_penalty` function).
    4.  Compute the explainable reward for the generated responses.
    5.  Calculate the final reward (reward - KL penalty).
    6.  Update the baseline using the final rewards.
    7.  Compute the advantage (final reward - current baseline value).
    8.  Calculate the REINFORCE loss: `- (advantage.detach() * sequence_log_probs).mean()`.
    9.  Perform backpropagation using `torch.cuda.amp.GradScaler` for mixed precision.
    10. Accumulate gradients and perform optimizer/scheduler steps.
    11. Log metrics to WandB.
    12. Perform validation periodically.

### 2.5. Key Differences: PPO vs. REINFORCE

*   **Update Mechanism:** PPO uses a clipped surrogate objective and optimizes both policy and value functions, often involving multiple inner optimization epochs per data batch. REINFORCE directly uses the policy gradient theorem with baseline subtraction, typically performing one update per batch (or accumulation cycle).
*   **Variance Reduction:** PPO relies heavily on the learned value function (critic) and Generalized Advantage Estimation (GAE) for variance reduction (though our setup used `trl` defaults). REINFORCE relies primarily on the baseline subtraction method chosen (here, an exponential moving average).
*   **Implementation Complexity:** Using TRL significantly simplifies PPO implementation. REINFORCE required a manual implementation of the training loop, log probability calculation, baseline, and gradient updates.

## 3. Implementation Details

*   **Libraries:** `transformers`, `datasets`, `trl`, `peft`, `torch`, `numpy`, `wandb`, `huggingface_hub`, `python-dotenv`, `tenacity`, `tqdm`.
*   **Hardware:** Training was attempted on available GPU resources (details might be added if known, e.g., Kaggle P100, Colab T4).
*   **Efficiency:** 4-bit quantization (`bitsandbytes`) and LoRA (`peft`) were used for parameter-efficient fine-tuning to make training feasible on constrained hardware.
*   **Configuration:** Centralized configuration was managed in `src/config.py`.
*   **Code Structure:** Organized into `src/` for modules, `data/` for preprocessing, and `notebooks/` for exploration and training execution.

## 4. Experiments and Execution Attempts

The plan involved:
1.  Preparing the HelpSteer2 prompts using `preprocess_helpsteer.py`.
2.  Executing the PPO training notebook (`ppo_training.ipynb`).
3.  Executing the REINFORCE training notebook (`reinforce_training.ipynb`).
4.  Saving the trained LoRA adapters for both models to the Hugging Face Hub.
5.  Evaluating the resulting models using the evaluation notebook (`notebooks/evaluation.ipynb`) on a benchmark like Anthropic HH-RLHF, comparing performance based on the explainable reward metric.

**Actual Execution:**
*   Data preprocessing and reward model implementation were completed successfully.
*   The PPO training notebook (`ppo_training.ipynb`) was executed, but encountered an `AttributeError: 'tuple' object has no attribute 'logits'` during the internal `trl` training loop (`ppo_trainer.train()`). This error typically indicates an issue with how the reference model or its outputs are being handled within the TRL framework, potentially related to the model class used (`AutoModelForCausalLMWithValueHead`) or PEFT interactions.
*   The REINFORCE training notebook (`reinforce_training.ipynb`) was executed after refactoring into helper functions. It encountered an `IndexError: index -1 is out of bounds for dimension 0 with size 0` during the `policy_model.generate` call within the manual loop, followed by a `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` during `backward()`. The `IndexError` suggests an issue with generation internals (potentially related to caching or state in PEFT models), and the `RuntimeError` occurred because the generation failure broke the computational graph, resulting in a loss tensor with no gradient history.

## 5. Challenges encountered

*   **TRL PPO Debugging:** Pinpointing the exact cause of the `AttributeError` within the `trl` library's PPO implementation proved difficult. Interactions between `AutoModelForCausalLMWithValueHead`, PEFT adapters, 4-bit quantization, and the reference model handling within `trl` can be complex. The fix likely involves ensuring the reference model is loaded as a standard `AutoModelForCausalLM` and that arguments passed to `PPOTrainer` are correct.
*   **Manual REINFORCE Implementation:**
    *   **Generation Stability:** The `IndexError` during generation suggests potential instability when using `.generate()` repeatedly within a training loop with PEFT models, possibly related to internal caching mechanisms. Explicitly setting `eval()` mode and disabling `use_cache` during generation was attempted but didn't fully resolve the issue in the user's run.
    *   **Gradient Flow:** The subsequent `RuntimeError` highlighted the fragility of the manual loop; any error that prevents the loss from being connected back to the trainable parameters through the log-probability calculation breaks the backward pass. Robust error handling per batch is essential but can mask underlying problems.
    *   **Log Probability Calculation:** Correctly calculating sequence log probabilities, especially handling padding and shifting indices, requires careful implementation.
*   **Resource Constraints:** Fine-tuning even small models (0.5B) with RL requires significant VRAM, especially with gradient accumulation and multiple model copies (for reference). Debugging OOM errors and finding optimal batch sizes can be time-consuming.
*   **LLM-as-Judge Reliability:** While not the primary failure point here, ensuring the Judge LLM consistently provides parseable integer scores within the correct range required robust parsing logic and retries (`tenacity`). Variations in judge output could add noise to the reward signal.

## 6. Expected Outcomes and Future Work

Had the training runs completed successfully, the following comparisons and analyses were planned:

*   **Performance Comparison:** Plotting the average explainable reward (aggregated score) on the validation set over training steps/epochs for both PPO and REINFORCE. This would show which algorithm converged faster or reached a higher final reward score according to the explainable metric.
*   **Stability Comparison:** Analyzing the variance of rewards and the KL divergence from the reference model during training. It was hypothesized (based on the "Back to Basics" paper) that REINFORCE might be more competitive than expected in the LLM fine-tuning setting due to strong initialization, potentially challenging PPO's perceived stability advantage.
*   **Explainability Analysis:** Generating responses from the final PPO-tuned and REINFORCE-tuned models on test prompts. Each response would be evaluated by the LLM-as-a-Judge across all dimensions. This would allow analysis of whether the models specialized (e.g., PPO high on helpfulness, REINFORCE high on coherence) or if they improved similarly across dimensions. Visualizations showing the dimensional score breakdown for specific examples were planned (as per the proposal deliverables).
*   **Qualitative Evaluation:** Manually inspecting generated outputs to assess fluency, coherence, and alignment beyond the quantitative scores.

**Future Work:**

1.  **Debugging Training Loops:** The immediate next step is to resolve the runtime errors in both the PPO (`trl`) and REINFORCE (manual) training loops. This might involve:
    *   Further investigation into `trl`'s handling of PEFT models with value heads and reference models.
    *   Carefully debugging the `generate` call and `cache_position` handling in the REINFORCE loop, potentially testing without PEFT initially or trying different generation parameters.
    *   Ensuring the gradient calculation in the REINFORCE log probability step is correctly implemented and maintains the graph connection.
2.  **Run Full Experiments:** Once debugged, execute the training runs for a sufficient number of steps/epochs on the target dataset subset (or full dataset if resources permit).
3.  **Quantitative Evaluation:** Implement the evaluation notebook (`notebooks/evaluation.ipynb`) to compare the saved models on the Anthropic HH-RLHF dataset using the explainable reward model and potentially pairwise comparisons.

## 7. Conclusion

This project successfully established the framework for comparing PPO and REINFORCE using explainable rewards for LLM alignment. Key components, including data preprocessing for RL prompts, the LLM-as-a-Judge reward model based on configurable dimensions, and the setup for both PPO (via TRL) and manual REINFORCE training loops with LoRA, were implemented. However, technical challenges and runtime errors encountered during the RL training phase prevented the completion of the experiments and the generation of comparative results. The documented methodology and implementation provide a foundation for future work focused on debugging the training processes and ultimately evaluating the performance, stability, and interpretability trade-offs between PPO and REINFORCE when guided by transparent, multi-dimensional reward signals.

## 8. References

[1] **Explainable Rewards in RLHF Using LLM-as-a-Judge** (2024) Anonymous Authors. (Paper under double-blind review). \
[2] **Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs** (2024) arXiv:2402.14740. 


