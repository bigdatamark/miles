# [MODULE DOC]
# HELPER: Tiny utility module giving end-to-end tests the directories for downloaded models and
# datasets. Each getter reads an environment variable and falls back to a default path under the
# filesystem root, so tests run unchanged on machines that keep models and data elsewhere.
import os


def get_test_model_dir() -> str:
    # [FUNCTION DOC] HELPER: Return the models directory: the value of the script model directory
    # environment variable if set, else "/root/models".
    return os.environ.get("MILES_SCRIPT_MODEL_DIR", "/root/models")


def get_test_data_dir() -> str:
    # [FUNCTION DOC] HELPER: Return the datasets directory: the value of the script data directory
    # environment variable if set, else "/root/datasets".
    return os.environ.get("MILES_SCRIPT_DATA_DIR", "/root/datasets")
