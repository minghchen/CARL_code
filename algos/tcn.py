# coding=utf-8
import torch


class TCN(object):
    """Time-contrastive Network."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reg_lambda = self.cfg.TCN.REG_LAMBDA

    def compute_loss(self, model, videos, seq_lens, chosen_steps, video_masks=None, training=True):
        num_frames = self.cfg.TRAIN.NUM_FRAMES
        batch_size, num_steps, c, h, w = videos.shape
        if video_masks is not None:
            video_masks = video_masks.view(-1, 1, num_steps)
        embs = model(videos, num_frames, video_masks=video_masks)
        losses = []
        for i in range(batch_size):
            losses.append(self.single_sequence_loss(embs[i], num_frames))
        loss = torch.mean(torch.stack(losses, dim=0))
        return {"loss": loss}

    def single_sequence_loss(self, embs, num_frames):
        """Returns n-pairs loss for a single sequence."""

        labels = torch.arange(num_frames//2).to(embs.device)
        embeddings_anchor = embs[0::2]
        embeddings_positive = embs[1::2]
        loss = self.npairs_loss(labels, embeddings_anchor, embeddings_positive)
        return loss

    def npairs_loss(self, labels, embeddings_anchor, embeddings_positive):
        """Returns n-pairs metric loss."""
        reg_anchor = torch.mean(torch.sum(torch.square(embeddings_anchor), 1))
        reg_positive = torch.mean(torch.sum(torch.square(embeddings_positive), 1))
        l2loss = 0.25 * self.reg_lambda * (reg_anchor + reg_positive)

        # Get per pair similarities.
        similarity_matrix = torch.matmul(
            embeddings_anchor, embeddings_positive.transpose(0,1))

        # Add the softmax loss.
        xent_loss = torch.nn.CrossEntropyLoss(reduction='mean')(similarity_matrix, labels)
        xent_loss = torch.mean(xent_loss)

        return l2loss + xent_loss
