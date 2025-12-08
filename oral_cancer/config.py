import os

# Base Directories
# If this file is in oral_cancer/config.py, then:
# dirname(abspath(__file__)) -> .../oral_cancer
# dirname(...) -> .../ProjectRoot
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Input/Output Paths
RAW_INPUT_DIR = os.path.join(DATA_DIR, "raw", "TCMA", "tb09j6496")
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "raw", "TCMA", "clean_data")
FINAL_OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Data Processing Config
DROP_FEATURE = "1678.0"
N_FEATURES = 17 

# Feature Selection Config
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Logging Config
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
