MODEL_SIZE = "2.8b"

BASE_HF_MODEL = f"state-spaces/mamba-{MODEL_SIZE}-hf"

INSTRUCT_DATASET_NAME = "yahma/alpaca-cleaned"
INSTRUCT_MODEL_HF_NAME = f"scottsus/mamba-{MODEL_SIZE}-instruct-hf"
INSTRUCT_MODEL_WEIGHTS_FILE = f"scottsus_mamba-{MODEL_SIZE}-instruct.pt"

KI_DATASET_NAME = "mark-arts/opencurriculumv0"
KI_MODEL_HF_NAME = f"scottsus/mamba-{MODEL_SIZE}-instruct-custom"
KI_MODEL_WEIGHTS_FILE = f"scottsus_mamba-{MODEL_SIZE}-instruct-custom.pt"
