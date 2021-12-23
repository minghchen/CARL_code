# coding=utf-8
import torch
from models import Classifier

import utils.logging as logging

logger = logging.get_logger(__name__)

class Classification(object):
    """Performs classification using labels."""
    # Classification does not support multiple datasets yet
    def __init__(self, cfg):
        self.cfg = cfg
        
    def compute_loss(self, model, videos, labels, seq_lens, chosen_steps, video_masks, training=True):

        num_frames = self.cfg.TRAIN.NUM_FRAMES

        batch_size, num_steps, c, h, w = videos.shape
        video_masks = video_masks.view(-1, 1, num_steps)
        logits = model(videos, num_frames, video_masks=video_masks, classification=True)

        labels = labels.long().to(logits.device).view(-1)
        valid = (labels>=0)
        logits = logits.view(-1, logits.size(-1))
        video_masks = video_masks.to(logits.device).view(-1)
        if training:
            loss = torch.nn.CrossEntropyLoss(reduction="none")(logits[valid], labels[valid])
        else:
            loss = (torch.argmax(logits, dim=1)[valid] == labels[valid]).float()
        loss = torch.sum(loss*video_masks[valid]) / torch.sum(video_masks[valid])
        return {"loss": loss}
