# coding=utf-8
"""Visualize alignment based on nearest neighbor in embedding space."""
import os
import torch
import math
import numpy as np
from scipy.spatial.distance import cdist
import argparse
import utils.logging as logging
from utils.dtw import dtw
from utils.config import get_cfg

import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FuncAnimation  # pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt

logger = logging.get_logger(__name__)

EPSILON = 1e-7

def get_nn(embs, query_emb):
    dist = cdist(embs, query_emb, axis=1)
    assert len(dist) == len(embs)
    return np.argmin(dist), np.min(dist)


def unnorm(query_frame):
    min_v = query_frame.min()
    max_v = query_frame.max()
    query_frame = (query_frame - min_v) / (max_v - min_v)
    return query_frame


def align(query_feats, candidate_feats, use_dtw):
    """Align videos based on nearest neighbor or dynamic time warping."""
    if use_dtw:
        _, _, _, path = dtw(query_feats, candidate_feats, dist='sqeuclidean')
        _, uix = np.unique(path[0], return_index=True)
        nns = path[1][uix]
    else:
        dists = cdist(query_feats, candidate_feats, 'sqeuclidean')
        nns = np.argmin(dists, axis=1)
    return nns


def create_video(query_embs, query_frames, key_embs, key_frames, video_path, use_dtw, interval=50, time_stride=1, image_out=False):
    """Create aligned videos."""
    nns = align(query_embs, key_embs, use_dtw)
    if time_stride>1:
        query_frames = query_frames[::time_stride]
        nns = nns[::time_stride]
        interval = interval*time_stride

    plt.figure(figsize=(5,1))
    nns_stride = np.floor(nns/time_stride)
    print(nns_stride)
    for t, t_nns in enumerate(nns_stride):
        plt.plot([t, t_nns], [1, 0], 'k--')
        plt.show()
    plt.grid(False)
    plt.savefig(video_path.split('.mp4')[0]+".png")

    fig, ax = plt.subplots(ncols=2, figsize=(10, 10), tight_layout=True)

    def update(i):
        """Update plot with next frame."""
        if i % 10 == 0:
            logger.info(f'{i}/{len(query_frames)}')
        ax[0].imshow(unnorm(query_frames[i]))
        ax[1].imshow(unnorm(key_frames[nns[i]]))
        # Hide grid lines
        ax[0].grid(False)
        ax[1].grid(False)

        # Hide axes ticks
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_yticks([])
        plt.tight_layout()
    
    if image_out:
        image_folder = video_path.split('.mp4')[0]
        os.makedirs(image_folder, exist_ok=True)
        for i in np.arange(len(query_frames)):
            update(i)
            plt.savefig(os.path.join(image_folder, f"frame_{i}.png"))
    else:
        anim = FuncAnimation(
            fig,
            update,
            frames=np.arange(len(query_frames)),
            interval=interval,
            blit=False)
        anim.save(video_path, dpi=80)


def create_multiple_video(query_embs, query_frames, key_embs_list, key_frames_list, video_path, use_dtw, 
                        interval=50):
    """Create aligned videos."""
    K = len(key_embs_list)
    nns_list = []
    for key_embs in key_embs_list:
        nns = align(query_embs, key_embs, use_dtw)
        nns_list.append(nns)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 10), tight_layout=True)

    def update(i):
        """Update plot with next frame."""
        if i % 10 == 0:
            logger.info(f'{i}/{len(query_frames)}')
        ax[0, 0].imshow(unnorm(query_frames[i]))
        ax[0, 0].grid(False)
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
        for k in range(K):
            ax[(k+1)//3,(k+1)%3].imshow(unnorm(key_frames_list[k][nns_list[k][i]]))
            ax[(k+1)//3,(k+1)%3].grid(False)
            ax[(k+1)//3,(k+1)%3].set_xticks([])
            ax[(k+1)//3,(k+1)%3].set_yticks([])
        plt.tight_layout()
    
    anim = FuncAnimation(
        fig,
        update,
        frames=np.arange(len(query_frames)),
        interval=interval,
        blit=False)
    anim.save(video_path, dpi=80)


def create_single_video(frames, labels, video_path, interval=50, time_stride=1, image_out=False):
    """Create aligned videos."""
    fig, ax = plt.subplots(ncols=1, figsize=(10, 10), tight_layout=True)
    if time_stride>1:
        frames = frames[::time_stride]
        interval = interval*time_stride

    print(labels[::time_stride])

    def update(i):
        """Update plot with next frame."""
        if i % 10 == 0:
            print(f'{i}/{len(frames)}')
        ax.imshow(unnorm(frames[i]))
        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
    
    if image_out:
        image_folder = video_path.split('.mp4')[0]
        os.makedirs(image_folder, exist_ok=True)
        for i in np.arange(len(frames)):
            update(i)
            plt.savefig(os.path.join(image_folder, f"frame_{i}.png"))
    else:
        anim = FuncAnimation(
            fig,
            update,
            frames=np.arange(len(frames)),
            interval=interval,
            blit=False)
        anim.save(video_path, dpi=80)


def visualize(args, cfg):
    """Visualize alignment."""
    import pickle
    import torch
    from torchvision.io import read_video

    with open(os.path.join(args.data_path, "pouring", 'train.pkl'), 'rb') as f:
        dataset = pickle.load(f)

    for data in dataset:
        name = data["name"]
        video_file = os.path.join(args.data_path, "pouring", data["video_file"])
        if name == args.reference_video:
            print(name)
            video, _, info = read_video(video_file, pts_unit='sec')
            video = video.permute(0,3,1,2).float() / 255.0
            query_frames = video.numpy()
            query_embs = np.arange(len(query_frames)).reshape(-1,1)
        elif name == args.candidate_video:
            print(name)
            video, _, info = read_video(video_file, pts_unit='sec')
            video = video.permute(0,3,1,2).float() / 255.0
            key_frames = video.numpy()
            key_embs = np.arange(len(key_frames)).reshape(-1,1)

    create_video(
        query_embs, query_frames, key_embs, key_frames,
        args.video_path,
        args.use_dtw,
        interval=args.interval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize alignment.")
    parser.add_argument('--data_path', type=str, default='/home/username/datasets', help='Path to video data.')
    parser.add_argument('--video_path', type=str, default=None, help='Path to output aligned video.')
    parser.add_argument('--use_dtw', action='store_true', default=False, help='Use dynamic time warping.')
    parser.add_argument('--reference_video', type=str, default=None, help='Reference video.')
    parser.add_argument('--candidate_video', type=str, default=None, help='Target video.')
    parser.add_argument('--interval', type=int, default=50, help='Time in ms b/w consecutive frames.')

    args = parser.parse_args()
    cfg = get_cfg()
    if args.video_path is None:
        args.video_path = "./test.mp4"
    args.reference_video = 'milk_to_clear99_real_view_1'
    args.candidate_video = 'clearsoda_to_white_real_view_1'
    visualize(args, cfg)
