# copy from slowfast.utils.parser
import os
import yaml
import utils.logging as logging
import argparse
from utils.config import get_cfg
from easydict import EasyDict

logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="TCC training pipeline.")
    parser.add_argument('--local_rank', default=0, type=int, help='rank in local processes')

    parser.add_argument('--workdir', type=str, default='/home/username/datasets', help='Path to datasets and pretrained models.')
    parser.add_argument('--logdir', type=str, default=None, help='Path to logs.')
    parser.add_argument('--continue_train', action='store_true',
                        default=False, help='Continue with training even when \
                        train_logs exist. Useful if one has to resume training. \
                        By default switched off to prevent overwriting existing \
                        experiments.')
    parser.add_argument('--visualize', action='store_true',
                        default=False, help='Visualize images, gradients etc. \
                        Switched off by for default to speed training up and \
                        takes less memory.')
    parser.add_argument(
        "--cfg_file",
        help="Path to the config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--opts",
        help="See ./utils/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()

def convert_value(value, v):
    if isinstance(value, bool):
        if v.strip() == "False" or v.strip() == "false":
            return False
        elif v.strip() == "True" or v.strip() == "true":
            return True
    elif isinstance(value, str):
        return str(v)
    elif isinstance(value, int):
        return int(v)
    elif isinstance(value, float):
        return float(v)
    elif isinstance(value, list):
        return [convert_value(value[0], _v) for _v in v.strip("[").strip("]").split(" ")]
    else:
        raise ValueError("Don't support for config type:", type(value))


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None and os.path.exists(args.cfg_file):
        logger.info('Using config from %s.', args.cfg_file)
        with open(args.cfg_file, 'r') as config_file:
            config_dict = yaml.safe_load(config_file)
        cfg.update(config_dict)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        for full_key, v in zip(args.opts[0::2], args.opts[1::2]):
            key_list = full_key.split(".")
            d = cfg
            for subkey in key_list[:-1]:
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = convert_value(d[subkey], v)

    if args.logdir is not None:
        cfg.LOGDIR = args.logdir
    else:
        cfg.LOGDIR = os.path.join('/tmp', cfg.LOGDIR)

    cfg.EVAL.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
    cfg.EVAL.NUM_FRAMES = cfg.TRAIN.NUM_FRAMES
    return cfg

def to_dict(config):
    if isinstance(config, list):
        return [to_dict(c) for c in config]
    elif isinstance(config, EasyDict):
        return dict([(k, to_dict(v)) for k, v in config.items()])
    else:
        return config

def setup_train_dir(cfg, logdir, continue_train=False):
    """Setups directory for training."""
    os.makedirs(logdir, exist_ok=True)
    config_path = os.path.join(logdir, 'config.yml')
    '''
    if not continue_train:
        import shutil
        shutil.rmtree(logdir)
    '''
    if not os.path.exists(config_path):
        logger.info(
            'Using config from config.py as no config.yml file exists in '
            '%s', logdir)
        with open(config_path, 'w') as config_file:
            config = dict([(k, to_dict(v)) for k, v in cfg.items()])
            yaml.safe_dump(config, config_file, default_flow_style=False)
    else:
        logger.info('Using config from config.yml that exists in %s.', logdir)
        with open(config_path, 'r') as config_file:
            config_dict = yaml.safe_load(config_file)
        cfg.update(config_dict)

    train_logs_dir = os.path.join(logdir, 'train_logs')
    os.makedirs(train_logs_dir, exist_ok=True)