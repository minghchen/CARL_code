# coding=utf-8
"""Evaluate embeddings on downstream tasks."""

import os
import shutil
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pprint
import torch
import numpy as np
import pickle
from itertools import chain
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.distance import cdist

import utils.logging as logging
from utils.dtw import dtw
from utils.distributed import all_gather_unaligned, synchronize, is_root_proc
from utils.parser import parse_args, load_config, setup_train_dir
from models import build_model, save_checkpoint, load_checkpoint
from utils.optimizer import construct_optimizer
from datasets import construct_dataloader, unnorm
from evaluation import get_tasks
from visualize_alignment import create_video
from visualize_retrieval import create_retrieval_video

logger = logging.get_logger(__name__)

class FinegymEval(torch.utils.data.Dataset):
    def __init__(self, cfg, split, data_files):
        self.cfg = cfg
        self.split = split
        self.data_files = data_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        with open(self.data_files[index], 'rb') as f:
            data = pickle.load(f)
        valid = (data['labels']>=0)
        return data['embs'][valid].float(), data['labels'][valid].long()

def colleta_fn(batch):
    xs = []
    ys = []
    for x, y in batch:
        xs.append(x)
        ys.append(y)
    return xs, ys

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

def get_embeddings_dataset(cfg, model, data_loader, output_dir):
    """Get embeddings from a one epoch iterator."""
    max_frames_per_batch = cfg.EVAL.FRAMES_PER_BATCH
    num_contexts = cfg.DATA.NUM_CONTEXTS
    output_files = []
    oneset_dataset = []
    if is_root_proc():
        data_loader = tqdm(data_loader, total=len(data_loader))
    model.eval()
    with torch.no_grad():
        for video, frame_label, seq_len, _, _, names in data_loader:
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
            embs = torch.cat(embs, dim=0)

            data = {'embs': embs,
                    'labels': frame_label[0],
                    'seq_len': seq_len,
                    'name': names[0]}
            output_file = os.path.join(output_dir, names[0]) + '.pkl'
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
            output_files.append(output_file)

            mask_for_UB_S1 = torch.bitwise_and((frame_label[0]>=74), (frame_label[0]<=88))
            if cfg.EVAL.CLASS_NUM == 99 and mask_for_UB_S1.long().sum() > 0:
                oneset_dataset.append({
                    'data': embs[mask_for_UB_S1].numpy(),
                    'label': frame_label[0][mask_for_UB_S1].numpy(),
                    'name': names[0],
                    'mask': mask_for_UB_S1
                })
    return output_files, oneset_dataset

def evaluate_once(cfg, model, train_loader, val_loader, train_emb_loader, val_emb_loader, 
                    iterator_tasks, embedding_tasks, cur_epoch, summary_writer):
    """Evaluate learnt embeddings on downstream tasks."""
    
    train_output_dir=os.path.join(cfg.LOGDIR, "finegym_eval_trainset")
    if is_root_proc():
        if os.path.exists(train_output_dir):
            shutil.rmtree(train_output_dir)
        os.makedirs(train_output_dir)
    synchronize()
    logger.info(f"generating train embeddings for finegym dataset at {train_output_dir} of epoch {cur_epoch}.")
    train_files, train_oneset_dataset = get_embeddings_dataset(cfg, model, train_emb_loader[0], train_output_dir)
    if cfg.NUM_GPUS > 1:
        train_files = list(chain(*all_gather_unaligned(train_files)))
        train_oneset_dataset = list(chain(*all_gather_unaligned(train_oneset_dataset)))

    val_output_dir=os.path.join(cfg.LOGDIR, "finegym_eval_valset")
    if is_root_proc():
        if os.path.exists(val_output_dir):
            shutil.rmtree(val_output_dir)
        os.makedirs(val_output_dir)
    synchronize()
    logger.info(f"generating val embeddings for finegym dataset at {val_output_dir} of epoch {cur_epoch}.")
    val_files, val_oneset_dataset = get_embeddings_dataset(cfg, model, val_emb_loader[0], val_output_dir)
    if cfg.NUM_GPUS > 1:
        val_files = list(chain(*all_gather_unaligned(val_files)))
        val_oneset_dataset = list(chain(*all_gather_unaligned(val_oneset_dataset)))

    fractions = cfg.EVAL.CLASSIFICATION_FRACTIONS
    if cfg.TRAINING_ALGO == 'classification':
        fractions = [1]
    learning_rate = cfg.EVAL.CLASSIFICATION_LR
    for fraction in fractions:
        batch_size = 10
        num_train = max(cfg.NUM_GPUS*batch_size, int(fraction * len(train_files)))
        train_embs = FinegymEval(cfg, 'train', train_files[:num_train])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_embs) if cfg.NUM_GPUS > 1 else None
        train_embs_loader = torch.utils.data.DataLoader(train_embs, batch_size=batch_size, 
                                    shuffle=False if cfg.NUM_GPUS > 1 else True, collate_fn=colleta_fn,
                                    num_workers=cfg.DATA.NUM_WORKERS, pin_memory=True, sampler=train_sampler,
                                    drop_last=True)
        val_embs = FinegymEval(cfg, 'val', val_files)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_embs) if cfg.NUM_GPUS > 1 else None
        val_embs_loader = torch.utils.data.DataLoader(val_embs, batch_size=batch_size,
                                    shuffle=False, collate_fn=colleta_fn,
                                    num_workers=cfg.DATA.NUM_WORKERS, pin_memory=True, sampler=val_sampler,
                                    drop_last=False)

        model = LogisticRegression(cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE, cfg.EVAL.CLASS_NUM)
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [cfg.args.local_rank], 
                    output_device = cfg.args.local_rank, find_unused_parameters=True)
        criterion = torch.nn.CrossEntropyLoss()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-6)

        total_e = cfg.EVAL.CLASSIFICATION_EPOCHS
        train_accs = []
        val_accs = []
        for e in range(total_e):
            if cfg.NUM_GPUS > 1 and hasattr(train_embs_loader.sampler, 'set_epoch'):
                train_embs_loader.sampler.set_epoch(e)
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate * (1 + math.cos(math.pi * e/(1.0*total_e)))/2
            correct = 0.0
            total = 0.0
            for embs, labels in train_embs_loader:
                embs = torch.cat(embs, dim=0).cuda()
                labels = torch.cat(labels, dim=0).cuda()
                optimizer.zero_grad()
                outputs = model(embs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.cpu(), 1)
                total += labels.size(0)
                correct += (predicted == labels.cpu()).int().sum()
            correct = sum(all_gather_unaligned(correct))
            total = sum(all_gather_unaligned(total))
            train_accuracy = 100 * correct/total
            train_accs.append(correct/total)
            if e % 10 == 0:
                logger.info(f'[{e}/{total_e}] classification_{fraction} train set: {train_accuracy:.3f}% ({correct}/{total})')

            correct = 0.0
            total = 0.0
            for embs, labels in val_embs_loader:
                embs = torch.cat(embs, dim=0).cuda()
                labels = torch.cat(labels, dim=0)
                outputs = model(embs)
                _, predicted = torch.max(outputs.cpu(), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            correct = sum(all_gather_unaligned(correct))
            total = sum(all_gather_unaligned(total))
            accuracy = 100 * correct/total
            val_accs.append(correct/total)
            if e % 10 == 0:
                logger.info(f'[{e}/{total_e}] classification_{fraction} val set: {accuracy:.3f}% ({correct}/{total})')
        
        summary_writer.add_scalar(f'classification_{fraction}/train', train_accuracy, cur_epoch)
        summary_writer.add_scalar(f'classification_{fraction}/val', accuracy, cur_epoch)
        torch.cuda.empty_cache()

    if cfg.EVAL.CLASS_NUM == 99 and is_root_proc():
        evaluate_oneset_data(cfg, val_emb_loader, cur_epoch, summary_writer, val_oneset_dataset)
        
    synchronize()

def evaluate_oneset_data(cfg, val_emb_loader, cur_epoch, summary_writer, val_oneset_dataset):

    print("generating visualization for video alignment")
    time_stride=10
    query_data = val_oneset_dataset[0]
    key_data = val_oneset_dataset[1]

    for data in val_emb_loader[0].dataset.dataset:
        if data['name'] == query_data['name']:
            query_video = val_emb_loader[0].dataset[data["id"]][0][query_data['mask']].permute(0,2,3,1)
        elif data['name'] == key_data['name']:
            key_video = val_emb_loader[0].dataset[data["id"]][0][key_data['mask']].permute(0,2,3,1)

    create_video(np.arange(len(query_video)).reshape(-1,1), query_video, np.arange(len(key_video)).reshape(-1,1), key_video, 
            os.path.join(cfg.LOGDIR, f'origin_{cur_epoch}.mp4'), use_dtw=False, interval=50, time_stride=time_stride, image_out=True)
    create_video(query_data['data'], query_video, key_data['data'], key_video, 
            os.path.join(cfg.LOGDIR, f'alignment_{cur_epoch}.mp4'), use_dtw=True, interval=50, time_stride=time_stride, image_out=True)

    print("generating visualization for frame retrieval")
    time_stride = 10
    K = 5
    query_feats = query_data['data'][::time_stride]
    query_video = query_video[::time_stride]
    retrieval_frames = [[] for _ in range(K)]
    candidate_feats = []
    for j in range(1, len(val_oneset_dataset)):
        candidate_feats.append(val_oneset_dataset[j]['data'][::time_stride])
    number = np.cumsum([len(c) for c in candidate_feats])
    candidate_feats = np.concatenate(candidate_feats, axis=0)
    dists = cdist(query_feats, candidate_feats, 'sqeuclidean')
    topk = np.argsort(dists, axis=1)[:, :K]
    for t in range(len(topk)):
        print(f"step {t}/{len(topk)}")
        for k in range(K):
            video_i = np.searchsorted(number, topk[t, k])
            ret_data = val_oneset_dataset[video_i+1]
            for data in val_emb_loader[0].dataset.dataset:
                if data['name'] == ret_data['name']:
                    video = val_emb_loader[0].dataset[data["id"]][0][ret_data['mask']].permute(0,2,3,1)[::time_stride]
                    retrieval_frames[k].append(video[topk[t, k]-number[video_i]])
                    break

    create_retrieval_video(query_video, retrieval_frames,
        os.path.join(cfg.LOGDIR, f'retrieval_{cur_epoch}.mp4'), K, interval=1000, image_out=True)

def evaluate():
    """Evaluate embeddings."""
    args = parse_args()
    cfg = load_config(args)
    setup_train_dir(cfg, cfg.LOGDIR, args.continue_train)
    cfg.PATH_TO_DATASET = os.path.join(args.workdir, cfg.PATH_TO_DATASET)
    cfg.NUM_GPUS = torch.cuda.device_count()
    cfg.args = args
    if args.logdir is not None:
        cfg.LOGDIR = args.logdir

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
