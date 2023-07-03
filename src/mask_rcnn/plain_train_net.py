#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
import argparse
import sys
from collections import OrderedDict
import pickle
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_setup, default_writers, launch
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

from habitat_baselines.common.constants import scenes, coco_categories, coco_categories_mapping

logger = logging.getLogger("detectron2")
# use cuda if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

coco_categories_objects = ["chair", "couch", "potted plant", "bed", "toilet", "tv"]


def register_trainset(args):
    dataset_dicts = []
    for scene_name in scenes['train']:
        new_dict = get_gibson_dicts(scene_name, max_frames=10000, args=args, mode="train")
        dataset_dicts.extend(new_dict)
    return dataset_dicts

def register_testset():
    dataset_dicts = []
    for scene_name in scenes['val']:
        new_dict = get_gibson_dicts(scene_name, max_frames=1000, mode="test")
        dataset_dicts.extend(new_dict)
    return dataset_dicts

def register_holdout():
    dataset_dicts = []
    for scene_name in scenes['train']:
        new_dict = get_gibson_dicts(scene_name, max_frames=200, mode="holdout")
        dataset_dicts.extend(new_dict)
    return dataset_dicts

def register_dataset(args=None, mode='train'):
    if mode == 'test':
        DatasetCatalog.register("gibson_test", lambda: register_testset())
        MetadataCatalog.get("gibson_test").set(thing_classes=coco_categories_objects)
    elif mode == 'holdout':
        DatasetCatalog.register("gibson_holdout", lambda: register_holdout())
        MetadataCatalog.get("gibson_holdout").set(thing_classes=coco_categories_objects)
    elif mode == 'train':
        DatasetCatalog.register("gibson_train", lambda: register_trainset(args))
        MetadataCatalog.get("gibson_train").set(thing_classes=coco_categories_objects)
    else:
        raise NotImplementedError


def get_gibson_dicts(scene_str, max_frames, args=None, mode="train"):
    dataset_dicts = []

    # evaluation setting: holdout or test
    if mode == "holdout":
        for idx in range(max_frames):
            with open("data/eval/holdout/%s/dict/%03d.pkl" % (scene_str, idx), 'rb') as f:
                dict_obj = pickle.load(f)
            dataset_dicts.append(dict_obj)
    
    elif mode == "test":
        for idx in range(max_frames):
            with open("data/eval/test/%s/dict/%03d.pkl" % (scene_str, idx), 'rb') as f:
                dict_obj = pickle.load(f)
            dataset_dicts.append(dict_obj)
    
    elif mode == "train" and args is not None:
        for idx in range(max_frames):
            with open(args.dataset_dir + "/%s/dict/%05d.pkl" % (scene_str, idx), 'rb') as f:
                dict_obj = pickle.load(f)
            dataset_dicts.append(dict_obj)
    
    else:
        raise NotImplementedError

    return dataset_dicts


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    # evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    return COCOEvaluator(dataset_name, output_dir=output_folder)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # dataset configuration
    cfg.DATASETS.TRAIN = ("gibson_train")
    cfg.DATASETS.TEST = ("gibson_holdout", "gibson_test")

    # input configurations
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.INPUT.MIN_SIZE_TRAIN = (256,)
    cfg.INPUT.MIN_SIZE_TEST = (256,)
    cfg.INPUT.RANDOM_FLIP = "none"

    # number of GPUs in usage
    cfg.DATALOADER.NUM_WORKERS = args.num_gpus * 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size

    # learning rate schedule
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000
    cfg.SOLVER.STEPS = ()

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # set threshold for this model (default: 0.05)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6

    # training end-to-end
    cfg.MODEL.BACKBONE.FREEZE_AT = 0

    # pretrained weights and logging
    # data_{ddppo_rnd, ddppo_rnd_alp, ddppo_crl, ddppo_crl_alp}
    if args.pretrain_weights == "imagenet-sup":
        cfg.OUTPUT_DIR = os.path.join("logs", args.date, "mask_rcnn", "data_{}".format(args.dataset_dir.split("/")[2]), "repr_imagenet_sup")
    elif args.pretrain_weights == "random":
        cfg.OUTPUT_DIR = os.path.join("logs", args.date, "mask_rcnn", "data_{}".format(args.dataset_dir.split("/")[2]), "repr_random_init")
    elif args.pretrain_weights == "sim-pretrain":
        cfg.MODEL.WEIGHTS = args.pretrain_path.replace(".pth", "_convert.pth")
        cfg.OUTPUT_DIR = os.path.join("logs", args.date, "mask_rcnn", "data_{}".format(args.dataset_dir.split("/")[2]), "repr_{}".format(args.pretrain_path.split("/")[2]))
    else:
        raise NotImplementedError
    
    if args.eval_only:
        # evaluate custom model checkpoint
        cfg.MODEL.WEIGHTS = args.model_path
        # inference separately on each gpu and accumulate results
        cfg.MODEL.RESNETS.NORM = "BN"

    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    # configurations: representation and samples
    parser.add_argument("--pretrain-weights", type=str, choices=["random", "imagenet-sup", "sim-pretrain"], help="load backbone pretrained representation weights")
    parser.add_argument("--pretrain-path", type=str, required=False, help="path to load simulator pretrained representation weights")
    parser.add_argument("--model-path", type=str, required=False, help="path to load perception model for downstream evaluation")
    parser.add_argument('--dataset-dir', type=str, help='path to training samples directory for supervised training')

    # hyperparameters
    parser.add_argument('--max-iter', type=int, default=150000, help='maximum training iterations')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')

    # logging
    parser.add_argument("--date", type=str, help="date of experiments for logging")
    return parser


def main(args):
    # register dataset
    register_dataset(args=args, mode="train")
    register_dataset(mode="test")
    register_dataset(mode="holdout")

    cfg = setup(args)

    model = build_model(cfg)

    if (not args.eval_only) and args.pretrain_weights == "sim-pretrain":
        # load resnet from policy, but random init remaining 
        new_ckpt_dict = model.state_dict()
        # load resnet weight
        resnet_ckpt_dict = torch.load(args.pretrain_path, map_location='cpu')

        for k, v in resnet_ckpt_dict.items():
            new_k = "backbone.bottom_up." + k
            new_ckpt_dict[new_k] = v
        
        if comm.is_main_process():
            torch.save(new_ckpt_dict, args.pretrain_path.replace(".pth", "_convert.pth"))
            logger.info("Dumping simulator pretrained weights to {}".format(args.pretrain_path.replace(".pth", "_convert.pth")))

    logger.info("Model:\n{}".format(model))

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    if args.pretrain_weights == "imagenet-sup":
        args.config_file = "configs/mask_rcnn/mask_rcnn_R_50_FPN_3x.yaml"
    else:
        args.config_file = "configs/mask_rcnn/scratch_mask_rcnn_R_50_FPN_3x_syncbn.yaml"

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )