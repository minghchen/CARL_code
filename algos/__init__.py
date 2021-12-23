# coding=utf-8
from algos.classification import Classification
from algos.tcc import TCC
from algos.tcn import TCN
from algos.simclr import SimCLR

ALGO_NAME_TO_ALGO_CLASS = {
    'classification': Classification,
    'tcc': TCC,
    'tcn': TCN,
    'simclr': SimCLR,
}

def get_algo(cfg):
    """Returns training algo."""
    algo_name = cfg.TRAINING_ALGO
    if algo_name not in ALGO_NAME_TO_ALGO_CLASS.keys():
        raise ValueError('%s not supported yet.' % algo_name)
    algo = ALGO_NAME_TO_ALGO_CLASS[algo_name]
    return algo(cfg)
