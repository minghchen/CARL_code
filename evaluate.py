# coding=utf-8
"""Evaluate embeddings on downstream tasks."""

import os
import math
import torch
import pprint
import numpy as np
from tqdm import tqdm
import utils.logging as logging
from torch.utils.tensorboard import SummaryWriter

from utils.parser import parse_args, load_config, setup_train_dir
from models import build_model, save_checkpoint, load_checkpoint
from utils.optimizer import construct_optimizer
from datasets import construct_dataloader
from evaluation import get_tasks
from visualize_alignment import create_video, create_single_video, create_multiple_video
from visualize_retrieval import create_retrieval_video

logger = logging.get_logger(__name__)

def get_embeddings_dataset(cfg, model, data_loader):
    """Get embeddings from a one epoch iterator."""
    max_frames_per_batch = cfg.EVAL.FRAMES_PER_BATCH
    num_contexts = cfg.DATA.NUM_CONTEXTS
    embs_list = []
    steps_list = []
    seq_lens_list = []
    frame_labels_list = []
    names_list = []
    input_lens_list = []

    model.eval()
    with torch.no_grad():
        for video, frame_label, seq_len, chosen_steps, video_masks, names in data_loader:
            assert video.size(0) == 1 # batch_size==1
            assert video.size(1) == frame_label.size(1) == int(seq_len.item())
            embs = []
            seq_len = seq_len.item()
            num_batches = int(math.ceil(float(seq_len)/max_frames_per_batch))
            frames_per_batch = int(math.ceil(float(seq_len)/num_batches))
            for i in range(num_batches):
                curr_idx = i * frames_per_batch
                num_steps = min(seq_len - curr_idx, frames_per_batch)
                steps = torch.arange(curr_idx, curr_idx+num_steps)
                if num_contexts != 1:
                    # Get multiple context steps depending on config at selected steps.
                    context_stride = cfg.DATA.CONTEXT_STRIDE
                    steps = steps.view(-1,1) + context_stride*torch.arange(-(num_contexts-1), 1).view(1,-1)
                steps = torch.clamp(steps.view(-1), 0, seq_len - 1)
                curr_data = video[:, steps]
                # print(i, num_steps, seq_len, curr_data.shape)
                if cfg.USE_AMP:
                    with torch.cuda.amp.autocast():
                        emb_feats = model(curr_data, num_steps)
                else:
                    emb_feats = model(curr_data, num_steps)
                embs.append(emb_feats[0].cpu())
            valid = (frame_label[0]>=0)
            embs = torch.cat(embs, dim=0)
            embs_list.append(embs[valid].numpy())
            frame_labels_list.append(frame_label[0][valid].cpu().numpy())
            seq_lens_list.append(seq_len)
            input_lens_list.append(len(video[0]))
            steps_list.append(chosen_steps[0].cpu().numpy())
            names_list.append(names[0])

        dataset = {'embs': embs_list,
                    'labels': frame_labels_list,
                    'seq_lens': seq_lens_list,
                    'input_lens': input_lens_list,
                    'steps': steps_list,
                    'names': names_list}

        logger.info(f"embeddings_dataset size: {len(dataset['embs'])}")
    return dataset

def evaluate_once(cfg, model, train_loader, val_loader, train_emb_loader, val_emb_loader, 
                    iterator_tasks, embedding_tasks, cur_epoch, summary_writer):
    """Evaluate learnt embeddings on downstream tasks."""

    metrics = {}
    if iterator_tasks:
        for task_name, task in iterator_tasks.items():
            metrics[task_name] = task.evaluate(model, train_loader, val_loader, cur_epoch, summary_writer)

    if embedding_tasks:
        for i, dataset_name in enumerate(cfg.DATASETS):
            dataset = {'name': dataset_name}
            logger.info(f"generating train embeddings for {dataset_name} dataset at {cur_epoch}.")
            dataset['train_dataset'] = get_embeddings_dataset(cfg, model, train_emb_loader[i])
            logger.info(f"generating val embeddings for {dataset_name} dataset at {cur_epoch}.")
            dataset['val_dataset'] = get_embeddings_dataset(cfg, model, val_emb_loader[i])

            for task_name, task in embedding_tasks.items():
                if task_name not in metrics:
                    metrics[task_name] = {}
                metrics[task_name][dataset_name] = task.evaluate(dataset, cur_epoch, summary_writer)

            if dataset_name == "pouring" or dataset_name == "baseball_pitch":
                print("generating visualization for video alignment")
                time_stride=10
                K = 5
                q_id = 0
                k_ids = [1, 2, 3, 4, 5]
                query_data = dataset['val_dataset']['embs'][q_id]
                key_data_list = [dataset['val_dataset']['embs'][k_id] for k_id in k_ids]

                key_frames_list = [0 for _ in range(K)]
                for data_id, data in enumerate(val_emb_loader[i].dataset.dataset):
                    if data['name'] == dataset['val_dataset']['names'][q_id]:
                        query_video = val_emb_loader[i].dataset[data_id][0].permute(0,2,3,1)
                    else:
                        for k, k_id in enumerate(k_ids):
                            if data['name'] == dataset['val_dataset']['names'][k_id]:
                                key_frames_list[k] = val_emb_loader[i].dataset[data_id][0].permute(0,2,3,1)
                '''
                create_multiple_video(np.arange(len(query_video)).reshape(-1,1), query_video, 
                        [np.arange(len(key_video)).reshape(-1,1) for key_video in key_frames_list], key_frames_list, 
                        os.path.join(cfg.LOGDIR, f'origin_multi_{cur_epoch}.mp4'), use_dtw=True, interval=50)
                create_multiple_video(query_data, query_video, key_data_list, key_frames_list, 
                        os.path.join(cfg.LOGDIR, f'alignment_multi_{cur_epoch}.mp4'), use_dtw=True, interval=50)
                '''
                key_video, key_data = key_frames_list[0], key_data_list[0]
                create_video(np.arange(len(query_video)).reshape(-1,1), query_video, np.arange(len(key_video)).reshape(-1,1), key_video, 
                        os.path.join(cfg.LOGDIR, f'origin_{cur_epoch}.mp4'), use_dtw=False, interval=50, time_stride=time_stride, image_out=True)
                create_video(query_data, query_video, key_data, key_video, 
                        os.path.join(cfg.LOGDIR, f'alignment_{cur_epoch}.mp4'), use_dtw=True, interval=50, time_stride=time_stride, image_out=True)
                
            del dataset

    # Add all metrics in a separate tag so that analysis is easier.
    for task_name in embedding_tasks.keys():
        for dataset in cfg.DATASETS:
            # logger.info(f"metrics/{dataset}_{task_name}: {metrics[task_name][dataset]:.3f}")
            summary_writer.add_scalar('metrics/%s_%s' % (dataset, task_name),
                                metrics[task_name][dataset], cur_epoch)
        avg_metric = sum(metrics[task_name].values())
        avg_metric /= len(cfg.DATASETS)
        logger.info(f"metrics/all_{task_name}: {avg_metric:.3f}")
        summary_writer.add_scalar('metrics/all_%s' % task_name,
                        avg_metric, cur_epoch)
    


def evaluate():
    """Evaluate embeddings."""
    args = parse_args()
    cfg = load_config(args)
    setup_train_dir(cfg, cfg.LOGDIR, args.continue_train)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
    cfg.NUM_GPUS = torch.cuda.device_count()
    cfg.args = args

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # distributed logging and ignore warning message
    logging.setup_logging(cfg.LOGDIR)
    # Setup summary writer.
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'eval_logs'))

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model
    model = build_model(cfg)
    torch.cuda.set_device(args.local_rank)
    model = model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [args.local_rank], 
            output_device = args.local_rank, find_unused_parameters=False)
    optimizer = construct_optimizer(model, cfg)
    start_epoch = load_checkpoint(cfg, model, optimizer)

    # Setup Dataset Iterators from train and val datasets.
    train_loader, train_emb_loader = construct_dataloader(cfg, "train")
    val_loader, val_emb_loader = construct_dataloader(cfg, "val")
    iterator_tasks, embedding_tasks = get_tasks(cfg)

    evaluate_once(cfg, model, train_loader, val_loader, train_emb_loader, val_emb_loader, 
                        iterator_tasks, embedding_tasks, start_epoch, summary_writer)

if __name__ == '__main__':
    evaluate()
