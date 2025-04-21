from datasets import load_dataset, DatasetDict
from src import config
import logging

logger = logging.getLogger(__name__)


def load_and_prepare_rl_dataset(seed=config.SEED):
    """
    Loads HelpSteer2, uses existing splits, prepares prompts for RL.
    """
    logger.info(f"Loading dataset: {config.DATASET_NAME}")
    try:
        dataset = load_dataset(config.DATASET_NAME)
    except Exception as e:
        logger.error(f"Failed to load dataset {config.DATASET_NAME}: {e}")
        raise

    logger.info(f"Dataset loaded: {dataset}")

    # Filter out entries with missing prompt
    dataset = dataset.filter(lambda x: x[config.PROMPT_COLUMN] is not None and len(x[config.PROMPT_COLUMN]) > 0)

    # Prepare for RL: Extract prompts for train/test
    rl_dataset = DatasetDict({
        'train': dataset['train'].select_columns([config.PROMPT_COLUMN]),
        'test': dataset['validation'].select_columns([config.PROMPT_COLUMN])
    })

    # Rename column for TRL compatibility if needed later
    try:
        rl_dataset = rl_dataset.rename_column(config.PROMPT_COLUMN, "query")
        logger.info("Renamed prompt column to 'query'.")
    except ValueError:
        logger.warning("Column 'query' already exists or rename failed. Assuming 'query' is present.")

    logger.info(f"RL Dataset prepared with {len(rl_dataset['train'])} training prompts and {len(rl_dataset['test'])} test prompts.")
    logger.debug(f"RL Dataset sample: {rl_dataset['train'][0]}")

    return rl_dataset


if __name__ == "__main__":
    print("Running data preprocessing...")
    rl_data = load_and_prepare_rl_dataset()

    print("\nRL dataset structure:")
    print(rl_data)
    if rl_data:
        print("\nExample RL prompt (train):")
        print(rl_data['train'][5])
        print("\nExample RL prompt (test):")
        print(rl_data['test'][5])
    print("\nPreprocessing is done.")
