'''
Date: 2021-12-14 19:30:02
LastEditors: Minghao Chen
LastEditTime: 2022-03-24 19:57:16
FilePath: /CARL_code/algos/scl.py
'''
# coding=utf-8
import torch
import torch.nn.functional as F
import numpy as np
import math

def safe_div(a, b):
    out = a / b
    out[torch.isnan(out)] = 0
    return out

class SCL(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.positive_type = cfg.SCL.POSITIVE_TYPE
        self.negative_type = cfg.SCL.NEGATIVE_TYPE
        self.temperature = cfg.SCL.SOFTMAX_TEMPERATURE
        self.label_varience = cfg.SCL.LABEL_VARIENCE
        self.embedding_size = cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
        self.positive_window = cfg.SCL.POSITIVE_WINDOW

    def compute_loss(self, model, videos, seq_lens, chosen_steps, video_masks=None, training=True):
        """One pass through the model.

        Args:
        videos: Tensor, batches of tensors from many videos.
        training: Boolean, if True model is run in training mode.

        Returns:
        loss: Tensor, Float tensor containing loss
        """
        num_frames = self.cfg.TRAIN.NUM_FRAMES

        batch_size, num_views, num_steps, c, h, w = videos.shape
        videos = videos.view(-1, num_steps, c, h, w)
        if video_masks is not None:
            video_masks = video_masks.view(-1, 1, num_steps)
        # add mlp projection head
        embs = model(videos, num_frames, video_masks=video_masks, project=self.cfg.MODEL.PROJECTION)
        embs = embs.view(batch_size, num_views, num_frames, embs.size(-1))
        seq_lens = seq_lens.view(batch_size, num_views)

        loss = self.compute_sequence_loss(embs, seq_lens.to(embs.device), chosen_steps.to(embs.device), video_masks.to(embs.device))
        return loss

    def compute_sequence_loss(self, embs, seq_lens, steps, masks=None):

        batch_size, num_views, num_frames, channels = embs.shape

        embs = embs.view(-1, channels) # (batch_size*num_views*num_frames, channels)
        steps = steps.view(-1)
        seq_lens = seq_lens.unsqueeze(-1).expand(batch_size, num_views, num_frames).contiguous().view(-1).float()
        input_masks = masks.view(-1, 1)*masks.view(1, -1)

        logits = torch.matmul(embs, embs.transpose(0,1)) / self.temperature
        distence = torch.abs(steps.view(-1,1)/seq_lens.view(-1,1)*seq_lens.view(1,-1)-steps.view(1,-1))
        distence.masked_fill_((input_masks==0), 1e6)
        weight = torch.ones_like(logits)
        nn = torch.zeros_like(steps).long()

        # negative weight
        for b in range(batch_size):
            start = b*num_views*num_frames
            mid = start+num_frames
            end = (b+1)*num_views*num_frames
            nn[start:mid] = mid+torch.argmin(distence[start:mid,mid:end], dim=1)
            nn[mid:end] = start+torch.argmin(distence[mid:end,start:mid], dim=1)
            if "single" in self.negative_type:
                weight[start:end,:start].fill_(0)
                weight[start:end,end:].fill_(0)
            if "noself" in self.negative_type:
                weight[start:mid,start:mid] = 0
                weight[mid:end,mid:end] = 0
        weight.masked_fill_((input_masks==0), 1e-6)

        # positive weight
        label = torch.zeros_like(logits)
        if self.positive_type == "gauss":
            pos_weight = torch.exp(-torch.square(distence)/(2*self.label_varience)).type_as(logits)
            # according to three sigma law, we can ignore the distence further than three sigma.
            # it may avoid the numerical unstablity and keep the performance.
            # pos_weight[(distence>3*np.sqrt(self.label_varience))] = 0
            for b in range(batch_size):
                start = b*num_views*num_frames
                mid = start+num_frames
                end = (b+1)*num_views*num_frames
                cur_pos_weight = pos_weight[start:mid,mid:end]
                label[start:mid,mid:end] = safe_div(cur_pos_weight, cur_pos_weight.sum(dim=1, keepdim=True))
                cur_pos_weight = pos_weight[mid:end,start:mid]
                label[mid:end,start:mid] = safe_div(cur_pos_weight, cur_pos_weight.sum(dim=1, keepdim=True))

        exp_logits = torch.exp(logits)
        sum_negative = torch.sum(weight*exp_logits, dim=1, keepdim=True)

        loss = F.kl_div(torch.log(safe_div(exp_logits, sum_negative) + 1e-6), label, reduction="none")
        loss = torch.sum(loss*input_masks)
        loss = loss / torch.sum(masks)
        
        return {"loss": loss}