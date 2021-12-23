# coding=utf-8
import torch

class TCC(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss_type=cfg.TCC.LOSS_TYPE
        self.similarity_type=cfg.TCC.SIMILARITY_TYPE
        self.cycle_length=cfg.TCC.CYCLE_LENGTH
        self.temperature=cfg.TCC.SOFTMAX_TEMPERATURE
        self.label_smoothing=cfg.TCC.LABEL_SMOOTHING
        self.variance_lambda=cfg.TCC.VARIANCE_LAMBDA
        self.huber_delta=cfg.TCC.HUBER_DELTA
        self.normalize_indices=cfg.TCC.NORMALIZE_INDICES

    def compute_loss(self, model, videos, seq_lens, chosen_steps, video_masks=None, training=True):
        """One pass through the model.

        Args:
        videos: Tensor, batches of tensors from many videos.
        training: Boolean, if True model is run in training mode.

        Returns:
        loss: Tensor, Float tensor containing loss
        """
        num_frames = self.cfg.TRAIN.NUM_FRAMES

        if self.cfg.SSL:
            batch_size, num_views, num_steps, c, h, w = videos.shape
            videos = videos.view(-1, num_steps, c, h, w)
            chosen_steps = chosen_steps.view(-1, num_frames)
            seq_lens = seq_lens.view(batch_size, num_views).view(-1)
        else:
            batch_size, num_steps, c, h, w = videos.shape
        if video_masks is not None:
            video_masks = video_masks.view(-1, 1, num_steps)
        embs = model(videos, num_frames, video_masks=video_masks)
        loss = self.compute_deterministic_alignment_loss(embs, seq_lens, chosen_steps)
        return loss

    def compute_deterministic_alignment_loss(self, embs, seq_lens, steps):

        labels_list = []
        logits_list = []
        steps_list = []
        seq_lens_list = []

        batch_size, num_frames, channels = embs.shape

        for i in range(batch_size):
            for j in range(batch_size):
                # We do not align the sequence with itself.
                if i == j:
                    continue
                logits, labels = self.align_pair_of_sequences(embs[i], embs[j])
                logits_list.append(logits)
                labels_list.append(labels)
                steps_list.append(steps[i].unsqueeze(0).expand(num_frames, num_frames))
                seq_lens_list.append(seq_lens[i].view(1,).expand(num_frames))
                
        logits = torch.cat(logits_list, 0)
        labels = torch.cat(labels_list, 0)
        steps = torch.cat(steps_list, 0)
        seq_lens = torch.cat(seq_lens_list, 0)

        if self.loss_type == 'classification':
            loss = {"loss": torch.nn.KLDivLoss(reduction='mean')(logits, labels)}
        elif 'regression' in self.loss_type:
            loss = self.regression_loss(logits, labels, steps, seq_lens)

        return loss

    def align_pair_of_sequences(self, embs1, embs2):
        """Align a given pair embedding sequences.

        Args:
            embs1: Tensor, Embeddings of the shape [M, D] where M is the number of
            embeddings and D is the embedding size.
            embs2: Tensor, Embeddings of the shape [N, D] where N is the number of
            embeddings and D is the embedding size.
        Returns:
            logits: Tensor, Pre-softmax similarity scores after cycling back to the
            starting sequence.
            labels: Tensor, One hot labels containing the ground truth. The index where
            the cycle started is 1.
        """
        num_steps, channels = embs1.shape

        sim_12 = self.get_scaled_similarity(embs1, embs2)
        # Softmax the distance.
        softmaxed_sim_12 = torch.softmax(sim_12, dim=-1)

        # Calculate soft-nearest neighbors.
        nn_embs = torch.matmul(softmaxed_sim_12, embs2)

        # Find distances between nn_embs and embs1.
        sim_21 = self.get_scaled_similarity(nn_embs, embs1)

        logits = sim_21
        labels = torch.diag(torch.ones(num_steps)).type_as(logits)
        if self.label_smoothing:
            labels = (1-num_steps*self.label_smoothing/(num_steps-1))*labels + \
                        self.label_smoothing/(num_steps-1)*torch.ones_like(labels)

        return logits, labels

    def get_scaled_similarity(self, embs1, embs2):
        num_steps, channels = embs1.shape
        # Find distances between embs1 and embs2.
        if self.similarity_type == 'cosine':
            sim_12 = torch.matmul(embs1, embs2.transpose(0,1))
        elif self.similarity_type == 'l2':
            norm1 = torch.square(embs1).sum(1).view(-1,1)
            norm2 = torch.square(embs2).sum(1).view(1,-1)
            sim_12 = - (norm1 + norm2 - 2*torch.matmul(embs1, embs2.transpose(0,1)))
        else:
            raise ValueError('Unsupported similarity type %s.' % self.similarity_type)
        return sim_12 / channels / self.temperature

    def regression_loss(self, logits, labels, steps, seq_lens):
        """Loss function based on regressing to the correct indices.

        In the paper, this is called Cycle-back Regression. There are 3 variants
        of this loss:
        i) regression_mse: MSE of the predicted indices and ground truth indices.
        ii) regression_mse_var: MSE of the predicted indices that takes into account
        the variance of the similarities. This is important when the rate at which
        sequences go through different phases changes a lot. The variance scaling
        allows dynamic weighting of the MSE loss based on the similarities.
        iii) regression_huber: Huber loss between the predicted indices and ground
        truth indices.


        Args:
            logits: Tensor, Pre-softmax similarity scores after cycling back to the
            starting sequence.
            labels: Tensor, One hot labels containing the ground truth. The index where
            the cycle started is 1.
            num_steps: Integer, Number of steps in the sequence embeddings.
            steps: Tensor, step indices/frame indices of the embeddings of the shape
            [N, T] where N is the batch size, T is the number of the timesteps.
            seq_lens: Tensor, Lengths of the sequences from which the sampling was done.
            This can provide additional temporal information to the alignment loss.

            loss_type: String, This specifies the kind of regression loss function.
            Currently supported loss functions: regression_mse, regression_mse_var,
            regression_huber.
            normalize_indices: Boolean, If True, normalizes indices by sequence lengths.
            Useful for ensuring numerical instabilities don't arise as sequence
            indices can be large numbers.
            variance_lambda: Float, Weight of the variance of the similarity
            predictions while cycling back. If this is high then the low variance
            similarities are preferred by the loss while making this term low results
            in high variance of the similarities (more uniform/random matching).
            huber_delta: float, Huber delta described in tf.keras.losses.huber_loss.

        Returns:
            loss: Tensor, A scalar loss calculated using a variant of regression.
        """
        steps = steps.type_as(logits)
        if self.normalize_indices:
            seq_lens = seq_lens.type_as(logits)
            steps = steps / seq_lens.unsqueeze(1)

        beta = torch.softmax(logits, dim=-1)
        true_time = torch.sum(steps * labels, dim=-1)
        pred_time = torch.sum(steps * beta, dim=-1)

        if self.loss_type in ['regression_mse', 'regression_mse_var']:
            if 'var' in self.loss_type:
                # Variance aware regression.
                pred_time_variance = torch.sum(torch.square(steps - pred_time.unsqueeze(-1)) * beta, dim=-1)
                assert torch.min(pred_time_variance) > 0
                # Using log of variance as it is numerically stabler.
                pred_time_log_var = torch.log(pred_time_variance)
                squared_error = torch.square(true_time - pred_time)
                loss = torch.mean(torch.exp(-pred_time_log_var) * squared_error
                                        + self.variance_lambda * pred_time_log_var)
                return {"loss": loss, "squared_error": torch.mean(squared_error), 
                                    "pred_time_log_var": torch.mean(pred_time_log_var)}
            else:
                return {"loss": torch.nn.MSELoss()(pred_time, true_time)}
        elif self.loss_type == 'regression_huber':
            return {"loss": torch.nn.SmoothL1Loss()(pred_time, true_time)}
        else:
            raise ValueError('Unsupported regression loss %s. Supported losses are: '
                            'regression_mse, regresstion_mse_var and regression_huber.'
                            % self.loss_type)
                