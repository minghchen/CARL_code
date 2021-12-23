# coding=utf-8
import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
import utils.logging as logging

logger = logging.get_logger(__name__)

def softmax(w, t=1.0):
    e = np.exp(np.array(w) / t)
    return e / np.sum(e)

class KendallsTau(object):
    """Calculate Kendall's Tau."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.downstream_task = True
        self.stride = cfg.EVAL.KENDALLS_TAU_STRIDE
        self.dist_type = cfg.EVAL.KENDALLS_TAU_DISTANCE
        if cfg.MODEL.L2_NORMALIZE:
            self.temperature = 0.1
        else:
            self.temperature = 1

    def evaluate(self, dataset, cur_epoch, summary_writer):
        """Labeled evaluation."""
        train_embs = dataset['train_dataset']['embs']

        self.get_kendalls_tau(
                train_embs,
                cur_epoch, summary_writer,
                '%s_train' % dataset['name'], visualize=True)

        val_embs = dataset['val_dataset']['embs']

        tau = self.get_kendalls_tau(val_embs, cur_epoch, summary_writer, '%s_val' % dataset['name'], visualize=True)
        return tau

    def get_kendalls_tau(self, embs_list, cur_epoch, summary_writer, split, visualize=False):
        """Get nearest neighbours in embedding space and calculate Kendall's Tau."""
        num_seqs = len(embs_list)
        taus = np.zeros((num_seqs * (num_seqs - 1)))
        idx = 0
        for i in range(num_seqs):
            query_feats = embs_list[i][::self.stride]
            for j in range(num_seqs):
                if i == j: continue
                candidate_feats = embs_list[j][::self.stride]
                dists = cdist(query_feats, candidate_feats, self.dist_type)
                nns = np.argmin(dists, axis=1)
                if visualize:
                    if (i==0 and j == 1) or (i < j and num_seqs == 14):
                        sim_matrix = []
                        for k in range(len(query_feats)):
                            sim_matrix.append(softmax(-dists[k], t=self.temperature))
                        sim_matrix = np.array(sim_matrix, dtype=np.float32)
                        summary_writer.add_image(f'{split}/sim_matrix_{i}_{j}', sim_matrix.T, cur_epoch, dataformats='HW')
                taus[idx] = kendalltau(np.arange(len(nns)), nns).correlation
                idx += 1
        # Remove NaNs.
        taus = taus[~np.isnan(taus)]
        tau = np.mean(taus)

        logger.info('epoch[{}/{}] {} set alignment tau: {:.4f}'.format(
            cur_epoch, self.cfg.TRAIN.MAX_EPOCHS, split, tau))

        summary_writer.add_scalar('kendalls_tau/%s_align_tau' % split, tau, cur_epoch)
        return tau
