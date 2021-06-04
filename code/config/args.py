# ---------- Train -------------------
SEED = 2323
MULTI_GPUS = [2]
RESUME = False
BEST_MODEL_NAME = "{arch}_best.pth"

HIDDEN_SIZE = 768
MAX_LENGTH = 512
BATCH_SIZE = 32
NUM_EPOCHS = 10
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 4e-5
WARMUP_PROPORTION = 0.05
DROPOUT = 0.1
FP16 = True
AUG = False

do_ema = True
do_adv = True  # whether to do adversarial training
adv_epsilon = 1.0             # Epsilon for adversarial
adv_name = 'word_embeddings'  # name for adversarial layer

# ------------ Predict -----------------
OUTPUT_PATH = "output/submission.csv"