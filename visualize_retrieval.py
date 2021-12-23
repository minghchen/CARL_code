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

def unnorm(query_frame):
    min_v = query_frame.min()
    max_v = query_frame.max()
    query_frame = (query_frame - min_v) / (max_v - min_v)
    return query_frame


def create_retrieval_video(query_frames, key_frames_list, video_path, K=5, interval=50, image_out=False):
    """Create aligned videos."""
    fig, ax = plt.subplots(ncols=K+1, figsize=(10, 10), tight_layout=True)

    def update(i):
        """Update plot with next frame."""
        if i % 10 == 0:
            logger.info(f'{i}/{len(query_frames)}')
        ax[0].imshow(unnorm(query_frames[i]))
        ax[0].grid(False)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        for k in range(K):
            ax[k+1].imshow(unnorm(key_frames_list[k][i]))
            ax[k+1].grid(False)
            ax[k+1].set_xticks([])
            ax[k+1].set_yticks([])
        plt.tight_layout()

    if image_out:
        image_folder = video_path.split('.mp4')[0]
        os.mkdir(image_folder)
        for i in np.arange(len(query_frames)):
            update(i)
            plt.savefig(os.path.join(image_folder, f"frame_{i}.png"), bbox_inches='tight')
    else:
        anim = FuncAnimation(
            fig,
            update,
            frames=np.arange(len(query_frames)),
            interval=interval,
            blit=False)
        anim.save(video_path, dpi=80)
