from __future__ import absolute_import
from enum import Enum

__version__ = "0.4.2" # 2024.11.15

class TaskType(Enum):
    CLASSIFICATION_ONE_HOT = 1
    CLASSIFICATION_COEXIST = 2
    REGRESSION     = 3
    SEGMENTATION   = 4
    RECONSTRUCTION = 5

TaskDict = {
    'cls': TaskType.CLASSIFICATION_ONE_HOT,
    'cls_coexist': TaskType.CLASSIFICATION_COEXIST,
    'regress': TaskType.REGRESSION,
    'seg': TaskType.SEGMENTATION,
    'rec': TaskType.RECONSTRUCTION
}