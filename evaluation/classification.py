# coding=utf-8
import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import utils.logging as logging

logger = logging.get_logger(__name__)


def fit_linear_model(train_embs, train_labels,
                    val_embs, val_labels):
    """Fit a linear classifier."""
    lin_model = LogisticRegression(max_iter=100000, solver='lbfgs',
                                    multi_class='multinomial', verbose=0)
    lin_model.fit(train_embs, train_labels)
    train_acc = lin_model.score(train_embs, train_labels)
    val_acc = lin_model.score(val_embs, val_labels)
    return lin_model, train_acc, val_acc


def fit_svm_model(train_embs, train_labels,
                  val_embs, val_labels):
    """Fit a SVM classifier."""
    svm_model = SVC(decision_function_shape='ovo', verbose=0)
    svm_model.fit(train_embs, train_labels)
    train_acc = svm_model.score(train_embs, train_labels)
    val_acc = svm_model.score(val_embs, val_labels)
    return svm_model, train_acc, val_acc


def fit_linear_models(train_embs, train_labels, val_embs, val_labels,
                      model_type='svm'):
    """Fit Log Regression and SVM Models."""
    if model_type == 'linear':
        model, train_acc, val_acc = fit_linear_model(train_embs, train_labels,
                                                val_embs, val_labels)
    elif model_type == 'svm':
        model, train_acc, val_acc = fit_svm_model(train_embs, train_labels,
                                            val_embs, val_labels)
    else:
        raise ValueError('%s model type not supported' % model_type)
    return model, train_acc, val_acc


class Classification(object):
    """Classification using small linear models."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.downstream_task = True

    def evaluate(self, dataset, cur_epoch, summary_writer, visualize=True):
        """Labeled evaluation."""
        fractions = self.cfg.EVAL.CLASSIFICATION_FRACTIONS

        train_embs = np.concatenate(dataset['train_dataset']['embs'])
        val_embs = np.concatenate(dataset['val_dataset']['embs'])

        if len(train_embs) == 0 or len(val_embs) == 0:
            raise ValueError('All embeddings are NAN. Something is wrong with model.')

        val_labels = np.concatenate(dataset['val_dataset']['labels'])

        val_accs = []
        train_dataset = dataset['train_dataset']
        num_samples = len(train_dataset['embs'])

        def worker(fraction_used):
            num_samples_used = max(1, int(fraction_used * num_samples))
            train_embs = np.concatenate(train_dataset['embs'][:num_samples_used])
            train_labels = np.concatenate(train_dataset['labels'][:num_samples_used])
            return fit_linear_models(train_embs, train_labels, val_embs, val_labels, model_type='linear')

        for fraction in fractions:
            model, train_acc, val_acc = worker(fraction)

            prefix = '%s_%s' % (dataset['name'], str(fraction))
            logger.info('[Epoch: {}] Classification {} Fraction'
                        'Train Accuracy: {:.3f},'.format(cur_epoch,
                                                        prefix, train_acc))
            logger.info('[Epoch: {}] Classification {} Fraction'
                        'Val Accuracy: {:.3f},'.format(cur_epoch,
                                                        prefix, val_acc))
            summary_writer.add_scalar('classification/train_%s_accuracy' % prefix,
                            train_acc, cur_epoch)
            summary_writer.add_scalar('classification/val_%s_accuracy' % prefix,
                            val_acc, cur_epoch)
            val_accs.append(val_acc)
            
        return val_accs[-1]