import importlib
from collections.abc import Iterator

import hydra
from pydantic import BaseModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def get_class_from_string(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)  # Split at the last dot
    module = importlib.import_module(module_name)  # Import the module dynamically
    return getattr(module, class_name)  # Get the class from the module


class OptimizerInfo(BaseModel):
    class_name: str
    kwargs: dict


class LRSchedulerInfo(BaseModel):
    class_name: str
    kwargs: dict


def optim_factory(parameters: Iterator, optim_info: OptimizerInfo) -> Optimizer:
    optim_class: type[Optimizer] = get_class_from_string(optim_info.class_name)

    return optim_class(parameters, **optim_info.kwargs)


def lr_sched_factory(optimizer: Optimizer, lr_sched_info: LRSchedulerInfo) -> LRScheduler:
    lr_sched_class: type[LRScheduler] = hydra.utils.get_class(lr_sched_info.class_name)

    return lr_sched_class(optimizer, **lr_sched_info.kwargs)
