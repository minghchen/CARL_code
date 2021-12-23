# coding=utf-8
r"""Evaluation on detecting key events using a RNN.
"""
import math
import torch
import numpy as np
import sklearn

from datasets.dataset_splits import DATASET_TO_NUM_CLASSES
import utils.logging as logging

logger = logging.get_logger(__name__)


class VectorRegression(sklearn.base.BaseEstimator):
    """Class to perform regression on multiple outputs."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y):
        _, m = y.shape
        # Fit a separate regressor for each column of y
        self.estimators_ = [sklearn.base.clone(self.estimator).fit(x, y[:, i])
                            for i in range(m)]
        return self

    def predict(self, x):
        # Join regressors' predictions
        res = [est.predict(x)[:, np.newaxis] for est in self.estimators_]
        return np.hstack(res)

    def score(self, x, y):
        # Join regressors' scores
        res = [est.score(x, y[:, i]) for i, est in enumerate(self.estimators_)]
        return np.mean(res)


def fit_model(train_embs, train_labels, val_embs, val_labels,
              global_step, num_classes, prefix, report_error=False):
    """Linear Regression to regress to fraction completed."""

    train_embs = np.concatenate(train_embs, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    val_embs = np.concatenate(val_embs, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    lin_model = VectorRegression(sklearn.linear_model.LinearRegression())
    lin_model.fit(train_embs, train_labels)

    train_score = lin_model.score(train_embs, train_labels)
    val_score = lin_model.score(val_embs, val_labels)

    return lin_model, train_score, val_score

def regression_labels_for_class(labels, class_idx):
    # Assumes labels are ordered. Find the last occurrence of particular class.
    transition_frame = np.argwhere(labels == class_idx)[-1, 0]
    return (np.arange(float(len(labels))) - transition_frame) / len(labels)


def get_regression_labels(class_labels, num_classes):
    regression_labels = []
    for i in range(num_classes - 1):
        regression_labels.append(regression_labels_for_class(class_labels, i))
    return np.stack(regression_labels, axis=1)


def get_targets_from_labels(all_class_labels, num_classes):
    all_regression_labels = []
    for class_labels in all_class_labels:
        all_regression_labels.append(get_regression_labels(class_labels,
                                                        num_classes))
    return all_regression_labels


class EventCompletion(object):
    """Predict event completion using linear regression."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.downstream_task = True

    def evaluate(self, dataset, cur_epoch, summary_writer, visualize=True):
        """Labeled evaluation."""
        fractions = self.cfg.EVAL.CLASSIFICATION_FRACTIONS

        train_embs = dataset['train_dataset']['embs']
        val_embs = dataset['val_dataset']['embs']
        num_classes = DATASET_TO_NUM_CLASSES[dataset['name']]

        if len(train_embs) == 0 or len(val_embs) == 0:
            raise ValueError('All embeddings are NAN. Something is wrong with model.')

        val_labels = get_targets_from_labels(dataset['val_dataset']['labels'],
                                            num_classes)

        num_samples = len(dataset['train_dataset']['embs'])
        val_scores = []
        for fraction in fractions:
            num_samples_used = max(1, int(fraction * num_samples))
            train_embs = dataset['train_dataset']['embs'][:num_samples_used]
            train_labels = get_targets_from_labels(
                dataset['train_dataset']['labels'][:num_samples_used], num_classes)
            model, train_score, val_score = fit_model(train_embs, train_labels, val_embs, val_labels,
                        cur_epoch, num_classes, '%s_%s' % (dataset['name'], str(fraction)))
            prefix = '%s_%s' % (dataset['name'], str(fraction))
            logger.info('[Global step: {}] Event Completion {} Fraction Train '
                        'Score: {:.3f},'.format(cur_epoch, prefix, train_score))
            logger.info('[Global step: {}] Event Completion {} Fraction Val '
                        'Score: {:.3f},'.format(cur_epoch, prefix, val_score))
            summary_writer.add_scalar('event_completion/train_%s_score' % prefix,
                            train_score, cur_epoch)
            summary_writer.add_scalar('event_completion/val_%s_score' % prefix,
                            val_score, cur_epoch)
            val_scores.append(val_score)
            
        return val_scores[-1]
