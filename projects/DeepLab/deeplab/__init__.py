# Copyright (c) Facebook, Inc. and its affiliates.
from .build_solver import build_lr_scheduler
from .config import add_deeplab_config
from .resnet import build_resnet_deeplab_backbone
from .semantic_seg import DeepLabV3Head, DeepLabV3PlusHead
from .bdd100k import load_bdd100k_semantic, register_bdd100k
from .bdd100k import _get_bdd100k_files, _get_bdd100k_metadata
