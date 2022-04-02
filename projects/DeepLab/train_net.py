#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
DeepLab Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import torch

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from detectron2.data import build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import DatasetEvaluators

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.projects.deeplab import FixedSizeCrop
from detectron2.projects.deeplab import CityscapesSemSegEvaluator, SemSegEvaluator


def build_sem_seg_train_aug(cfg, ignore_label=None):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        if cfg.INPUT.CROP.FIXED:
            augs.append(
                FixedSizeCrop(
                    cfg.INPUT.CROP.SIZE, True, cfg.MODEL.PIXEL_MEAN[0], ignore_label
                )
            )
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    if cfg.INPUT.FLIP.ENABLED:
        augs.append(T.RandomFlip())
    return augs


def build_sem_seg_test_bdd_aug(cfg, ignore_label):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TEST,
            cfg.INPUT.MAX_SIZE_TEST,
            cfg.INPUT.MIN_SIZE_TEST_SAMPLING,
        ),
        # FixedSizeCrop(cfg.INPUT.TEST_SIZE, True, 0.0, ignore_label),
        T.Resize(cfg.INPUT.TEST_SIZE),
    ]
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "sem_seg":
            return SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        ignore_label = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).ignore_label
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(
                cfg,
                is_train=True,
                augmentations=build_sem_seg_train_aug(cfg, ignore_label),
            )
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.INPUT.CROP.FIXED:
            ignore_label = MetadataCatalog.get(dataset_name).ignore_label
            mapper = DatasetMapper(
                cfg,
                is_train=False,
                augmentations=build_sem_seg_test_bdd_aug(cfg, ignore_label),
            )
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        else:
            return super().build_test_loader(cfg, dataset_name)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = args.output
    if args.model_path != "":
        cfg.MODEL.WEIGHTS = args.model_path
    if args.no_finetune:
        cfg.MODEL.BACKBONE.FREEZE_AT = 5
    if "bdd100k" not in cfg.DATASETS.TRAIN[0]:
        cfg.INPUT.FLIP.ENABLED = not args.no_flip
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--output", default="./output")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--no_finetune", action="store_true")
    parser.add_argument("--no_flip", action="store_true")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
