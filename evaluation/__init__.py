# coding=utf-8
"""List all available tasks."""

#from evaluation.algo_loss import AlgoLoss
from evaluation.classification import Classification
from evaluation.event_completion import EventCompletion
from evaluation.kendalls_tau import KendallsTau
from evaluation.retrieval import Retrieval

TASK_NAME_TO_TASK_CLASS = {
    #'algo_loss': AlgoLoss,
    'kendalls_tau': KendallsTau,
    'retrieval': Retrieval,
    'classification': Classification,
    'event_completion': EventCompletion,
}

def get_tasks(cfg):
    """Returns evaluation tasks."""
    iterator_tasks = {}
    embedding_tasks = {}

    for task_name in list(set(cfg.EVAL.TASKS)):
        if task_name not in TASK_NAME_TO_TASK_CLASS.keys():
            raise ValueError('%s not supported yet.' % task_name)
        task = TASK_NAME_TO_TASK_CLASS[task_name](cfg)
        if task.downstream_task:
            embedding_tasks[task_name] = task
        else:
            iterator_tasks[task_name] = task
    return iterator_tasks, embedding_tasks
