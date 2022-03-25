# coding=utf-8
"""Configuration of an experiment."""
from easydict import EasyDict as edict

# Get the configuration dictionary whose keys can be accessed with dot.
CONFIG = edict()

# ******************************************************************************
# Experiment params
# ******************************************************************************

# Directory for the experiment logs.
CONFIG.LOGDIR = '/tmp/scl_transformer_logs'
# Dataset for training TCC.
# Check dataset_splits.py for full list.
CONFIG.DATASETS = [
    'pouring',
    # 'baseball_pitch',
    # 'baseball_swing',
    # 'bench_press',
    # 'bowl',
    # 'clean_and_jerk',
    # 'golf_swing',
    # 'jumping_jacks',
    # 'pushup',
    # 'pullup',
    # 'situp',
    # 'squat',
    # 'tennis_forehand',
    # 'tennis_serve',
]

# self-supervised mode (SimClR-like methods compare two augmented views)
CONFIG.SSL = True 
# the name of dataset dir
CONFIG.PATH_TO_DATASET = 'pouring'
# Algorithm used for training: tcc, tcn, scl, classification.
CONFIG.TRAINING_ALGO = 'scl'
# Size of images/frames.
CONFIG.IMAGE_SIZE = 224  # For ResNet50

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Number of GPUs to use (applies to both training and testing).
CONFIG.NUM_GPUS = 1
# The index of the current machine.
CONFIG.SHARD_ID = 0
# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
CONFIG.RNG_SEED = 1

# ******************************************************************************
# Training params
# ******************************************************************************

CONFIG.TRAIN = edict()
# Number of training epoch.
CONFIG.TRAIN.MAX_EPOCHS = 500
# Number of samples in each batch.
CONFIG.TRAIN.BATCH_SIZE = 1
# Number of frames to use while training.
CONFIG.TRAIN.NUM_FRAMES = 240

# ******************************************************************************
# Eval params
# ******************************************************************************
CONFIG.EVAL = edict()
# Number of samples in each batch.
CONFIG.EVAL.BATCH_SIZE = 1
# Number of frames to use while evaluating. Only used to see loss in eval mode.
CONFIG.EVAL.NUM_FRAMES = 240
# Evaluate for each N epoches
CONFIG.EVAL.VAL_INTERVAL = 50

# A task evaluates the embeddings or the trained model.
# Currently available tasks are: 'algo_loss', 'classification',
# 'kendalls_tau', 'event_completion' (called progression in paper),
# Provide a list of tasks using which the embeddings will be evaluated.
CONFIG.EVAL.TASKS = [
    # 'algo_loss',
    'kendalls_tau',
    'retrieval',
    'classification',
    'event_completion',
]

# The video will be cut into clips for evaluation, 
# the max number of frame in each clip.
CONFIG.EVAL.FRAMES_PER_BATCH = 1000
CONFIG.EVAL.KENDALLS_TAU_STRIDE = 5  # 5 for Pouring, 2 for PennAction
CONFIG.EVAL.KENDALLS_TAU_DISTANCE = 'sqeuclidean'  # cosine, sqeuclidean
CONFIG.EVAL.CLASSIFICATION_FRACTIONS = [0.1, 0.5, 1.0]
CONFIG.EVAL.RETRIEVAL_KS = [5, 10, 15]

# ******************************************************************************
# Model params
# ******************************************************************************
CONFIG.MODEL = edict()

# The model used to construct temporal context
# transformer, conv, vanilla
CONFIG.MODEL.EMBEDDER_TYPE = 'transformer'

CONFIG.MODEL.BASE_MODEL = edict()
CONFIG.MODEL.BASE_MODEL.NETWORK = 'Resnet50_byol'
# 3: conv1-conv4 of resnet50 will be frozen, and conv5 will be finetuned
CONFIG.MODEL.BASE_MODEL.LAYER = 3
# The video will be sent to 2D resnet50 as batched frames, 
# the max number of frame in each batch.
CONFIG.MODEL.BASE_MODEL.FRAMES_PER_BATCH = 40

# Select which layers to train.
# train_base defines how we want proceed with the base model.
# 'frozen' : Weights are fixed and batch_norm stats are also fixed.
# 'train_all': Everything is trained and batch norm stats are updated.
# 'only_bn': Only tune batch_norm variables and update batch norm stats.
CONFIG.MODEL.TRAIN_BASE = 'frozen'

CONFIG.MODEL.EMBEDDER_MODEL = edict()
# Paramters for transformers
CONFIG.MODEL.EMBEDDER_MODEL.HIDDEN_SIZE = 256
CONFIG.MODEL.EMBEDDER_MODEL.D_FF = 1024
CONFIG.MODEL.EMBEDDER_MODEL.NUM_HEADS = 8
CONFIG.MODEL.EMBEDDER_MODEL.NUM_LAYERS = 3
# List of 3D Conv layers defined as (channels, kernel_size, activate).
CONFIG.MODEL.EMBEDDER_MODEL.CONV_LAYERS = [
    (256, 3, 1),
    (256, 3, 1),
]
CONFIG.MODEL.EMBEDDER_MODEL.FLATTEN_METHOD = 'max_pool'
# List of fc layers defined as (channels, activate).
CONFIG.MODEL.EMBEDDER_MODEL.FC_LAYERS = [
    (256, True),
    (256, True),
]
CONFIG.MODEL.EMBEDDER_MODEL.CAPACITY_SCALAR = 2
CONFIG.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE = 128
CONFIG.MODEL.EMBEDDER_MODEL.FC_DROPOUT_RATE = 0.1
CONFIG.MODEL.EMBEDDER_MODEL.USE_BN = True

# Whether L2 normalize the final embeddings.
# Only effect the output during evaluation
CONFIG.MODEL.L2_NORMALIZE = True

# The projection head introduced by SimCLR
CONFIG.MODEL.PROJECTION = True
CONFIG.MODEL.PROJECTION_HIDDEN_SIZE = 512
CONFIG.MODEL.PROJECTION_SIZE = 128

# ******************************************************************************
# our Sequential Contrastive Loss params
# ******************************************************************************
# Read our CARL paper for better understanding
CONFIG.SCL = edict()
CONFIG.SCL.LABEL_VARIENCE = 10.0
CONFIG.SCL.SOFTMAX_TEMPERATURE = 0.1
CONFIG.SCL.POSITIVE_TYPE = 'gauss'
CONFIG.SCL.NEGATIVE_TYPE = 'single_noself'
CONFIG.SCL.POSITIVE_WINDOW = 5

# ******************************************************************************
# TCC params
# ******************************************************************************
CONFIG.TCC = edict()
CONFIG.TCC.CYCLE_LENGTH = 2
CONFIG.TCC.LABEL_SMOOTHING = 0.1
CONFIG.TCC.SOFTMAX_TEMPERATURE = 0.1
CONFIG.TCC.LOSS_TYPE = 'regression_mse_var'
CONFIG.TCC.NORMALIZE_INDICES = True
CONFIG.TCC.VARIANCE_LAMBDA = 0.001
CONFIG.TCC.FRACTION = 1.0
CONFIG.TCC.HUBER_DELTA = 0.1
CONFIG.TCC.SIMILARITY_TYPE = 'l2'  # l2, cosine

# ******************************************************************************
# Time Contrastive Network params
# ******************************************************************************
CONFIG.TCN = edict()
CONFIG.TCN.POSITIVE_WINDOW = 5
CONFIG.TCN.REG_LAMBDA = 0.002

# ******************************************************************************
# Optimizer params
# ******************************************************************************
CONFIG.OPTIMIZER = edict()
# Supported optimizers are: AdamOptimizer, MomentumOptimizer
CONFIG.OPTIMIZER.TYPE = 'AdamOptimizer'
CONFIG.OPTIMIZER.WEIGHT_DECAY = 0.00001
CONFIG.OPTIMIZER.GRAD_CLIP = 10

CONFIG.OPTIMIZER.LR = edict()
# Initial learning rate for optimizer.
CONFIG.OPTIMIZER.LR.INITIAL_LR = 0.0001
# Learning rate decay strategy.
# Currently Supported strategies: fixed, cosine, cosinewarmup
CONFIG.OPTIMIZER.LR.DECAY_TYPE = 'cosine'
CONFIG.OPTIMIZER.LR.WARMUP_LR = 0.0001
CONFIG.OPTIMIZER.LR.FINAL_LR = 0.0
CONFIG.OPTIMIZER.LR.NUM_WARMUP_STEPS = 1

# ******************************************************************************
# Data params
# ******************************************************************************
CONFIG.DATA = edict()
CONFIG.DATA.FRACTION = 1.0 # The labeled fraction of samples for supervised learning
CONFIG.DATA.ADDITION_TRAINSET = False # additional traning set for finegym
CONFIG.DATA.SAMPLING_STRATEGY = 'time_augment' # offset_uniform (for TCC), time_augment (for SCL)
CONFIG.DATA.NUM_CONTEXTS = 1  # number of frames that will be embedded jointly,
CONFIG.DATA.CONTEXT_STRIDE = 1  # stride between context frames
CONFIG.DATA.SAMPLING_REGION = 1.5
CONFIG.DATA.CONSISTENT_OFFSET = 0.2
# Set this to False if your TFRecords don't have per-frame labels.
CONFIG.DATA.FRAME_LABELS = True
# stride of frames while embedding a video during evaluation.
CONFIG.DATA.SAMPLE_ALL_STRIDE = 1
CONFIG.DATA.NUM_WORKERS = 4

# ******************************************************************************
# Spatial Data Augmentation params
# ******************************************************************************
CONFIG.AUGMENTATION = edict()
CONFIG.AUGMENTATION.STRENGTH = 1.0
CONFIG.AUGMENTATION.RANDOM_FLIP = True
CONFIG.AUGMENTATION.RANDOM_CROP = True
CONFIG.AUGMENTATION.BRIGHTNESS = True
CONFIG.AUGMENTATION.BRIGHTNESS_MAX_DELTA = 0.8
CONFIG.AUGMENTATION.CONTRAST = True
CONFIG.AUGMENTATION.CONTRAST_MAX_DELTA = 0.8
CONFIG.AUGMENTATION.HUE = True
CONFIG.AUGMENTATION.HUE_MAX_DELTA = 0.2
CONFIG.AUGMENTATION.SATURATION = True
CONFIG.AUGMENTATION.SATURATION_MAX_DELTA = 0.8

# ******************************************************************************
# Logging params
# ******************************************************************************
CONFIG.LOGGING = edict()
# Number of epoches between summary logging.
CONFIG.LOGGING.REPORT_INTERVAL = 20

# ******************************************************************************
# Checkpointing params
# ******************************************************************************
CONFIG.CHECKPOINT = edict()
# Number of epoches between consecutive checkpoints.
CONFIG.CHECKPOINT.SAVE_INTERVAL = 50


def get_cfg():
    """
    Get a copy of the default config.
    """
    return CONFIG