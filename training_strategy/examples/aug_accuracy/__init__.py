"""Top-level package for NN utilities for the augmentation comparison"""

__author__ = """Vladyslav Andriiashen"""
__email__ = "vladyslav.andriiashen@cwi.nl"

# from . import data_loader
from .data_loader import ImageDatasetTransformable, InputError
from .model import NNmodel
from .logger import Logger
from .activation import ActivationMap
from . import utils
