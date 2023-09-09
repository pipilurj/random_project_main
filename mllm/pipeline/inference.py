import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import logging
import pathlib
import typing
import warnings
import traceback
from torchvision.ops import box_iou as box_iou_calculator

# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#
#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))
# warnings.showwarning = warn_with_traceback

SLURM_ENV = {k: v for k, v in os.environ.items() if 'SLURM' in k}
if SLURM_ENV:
    print(f"SLURM_ENV: {SLURM_ENV}")
project_path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_path))

import torch
import torch.cuda
import argparse

torch.set_printoptions(precision=3)
from mllm.config import prepare_args
from mllm.models import load_pretrained
from mllm.utils import print_trainable_params
from mllm.engine import prepare_trainer_collator
from mllm.dataset import prepare_data, prepare_target_processor
from mllm.utils.utils import *
import shutil
import tqdm
import deepspeed
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import transformers
from transformers import TrainerCallback
import time
from torch.utils.tensorboard import SummaryWriter
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)
import os


# os.environ["CUDA_VISIBLE_DEVICES"]='7'


def plot_images(real, gen, coord_gt, coord_pred, mask_pred_gt=None, save_path="", imgid=0):
    os.makedirs(save_path, exist_ok=True)
    w, h = real.shape
    x1_gt, y1_gt, x2_gt, y2_gt = coord_gt[0] * w, coord_gt[1] * h, coord_gt[2] * w, coord_gt[3] * h
    x1_pred, y1_pred, x2_pred, y2_pred = coord_pred[0] * w, coord_pred[1] * h, coord_pred[2] * w, coord_pred[3] * h
    # Assuming you have three images in NumPy format: image1, image2, image3

    # Create a figure with three subplots arranged horizontally
    if mask_pred_gt is not None:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Display the first image in the first subplot
    axs[0].imshow(real, cmap='gray')
    rect = patches.Rectangle((x1_gt, y1_gt), x2_gt - x1_gt, y2_gt - y1_gt, linewidth=2, edgecolor='r', facecolor='none')
    # Add the rectangle patch to the axes
    axs[0].add_patch(rect)
    axs[0].set_title("real")
    axs[0].axis('off')

    # Display the second image in the second subplot
    axs[1].imshow(gen, cmap='gray')
    axs[1].set_title("gen")
    rect = patches.Rectangle((x1_pred, y1_pred), x2_pred - x1_pred, y2_pred - y1_pred, linewidth=2, edgecolor='r',
                             facecolor='none')
    axs[1].add_patch(rect)
    axs[1].axis('off')

    if mask_pred_gt is not None:
        axs[2].imshow(mask_pred_gt, cmap='gray')
        axs[2].set_title("gen_gt")
        axs[2].axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the figure as an image file
    plt.savefig(os.path.join(save_path, f"masks_{imgid}.png"), dpi=300)  # Change the filename and format as desired
    plt.close()


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params):
    to_return = {k: t for k, t in named_params if "lora_" in k}
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return



def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    boxes_iou_meter = AverageMeter("boxIoU", ":6.3f")
    boxes_acc_meter = AverageMeter("boxAcc", ":6.3f")
    autoencoder = model_engine.autoencoder
    model_engine.eval()
    autoencoder.eval()
    # model_engine.train()
    with torch.no_grad():
        dtype = torch.float32
        if args.fp16:
            dtype = torch.float16
        if args.bf16:
            dtype = torch.bfloat16
        for i, input_dict in enumerate(tqdm.tqdm(val_loader)):
            input_dict = dict_to_cuda(input_dict)
            # with torch.cuda.amp.autocast(dtype=dtype):
            _ = model_engine(**input_dict)
        autoencoder.eval()
        for i, input_dict in enumerate(tqdm.tqdm(val_loader)):
            input_dict = dict_to_cuda(input_dict)
            with torch.cuda.amp.autocast(dtype=dtype):
                output_dict = model_engine(**input_dict)
                masks_list = input_dict["masks_seq"]
                masks_list = torch.cat(
                    [torch.cat([torch.cat([masks for masks in masks_seq]) for masks_seq in masks_seqs]) for masks_seqs in
                     masks_list]).cuda()
                masks_list = (masks_list > 0.5).float()
                mask_decode_gt = autoencoder(masks_list.unsqueeze(dim=1)).sigmoid()
            # evaluate mask
            pred_masks = output_dict["pred_masks"]
            pred_boxes = output_dict["pred_boxes"]
            target_boxes = input_dict["boxes_seq"]
            target_boxes = torch.tensor(target_boxes).cuda().reshape(-1, 4)
            output_list = (pred_masks > 0.5).float().squeeze()
            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip((masks_list>0.5).int(), output_list.int()):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]

            # evaluate box
            box_ious = box_iou_calculator(pred_boxes * 1000, target_boxes * 1000)
            box_ious = torch.einsum('i i -> i', box_ious)  # take diag elem
            # NOTE: please note iou only calculate for success target
            box_iou = box_ious.mean().item()
            correct_rate = (box_ious > 0.5).sum().item() / target_boxes.shape[0]

            intersection_meter.update(intersection, n=masks_list.shape[0]), union_meter.update(union,
                                                                                               n=masks_list.shape[
                                                                                                   0]), acc_iou_meter.update(
                acc_iou, n=masks_list.shape[0])
            boxes_iou_meter.update(box_iou, n=target_boxes.shape[0]), boxes_acc_meter.update(correct_rate,
                                                                                             n=target_boxes.shape[0])
            for k, (mask_gt, mask_pred, mask_pred_gt, box_gt, box_pred) in enumerate(zip(masks_list, output_list, mask_decode_gt, target_boxes, pred_boxes)):
                plot_images(mask_gt.to(torch.float32).squeeze().detach().cpu().numpy(),
                            mask_pred.to(torch.float32).squeeze().detach().cpu().numpy(),
                            box_gt.to(torch.float32).squeeze().detach().cpu().numpy(),
                            box_pred.to(torch.float32).squeeze().detach().cpu().numpy(),
                            mask_pred_gt=mask_pred_gt.to(torch.float32).squeeze().detach().cpu().numpy(),
                            save_path=os.path.join(args.output_dir, "images_val"),
                            imgid=f"epoch{epoch}-iter{i}-{k}")

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    boxiou = boxes_iou_meter.avg
    boxacc = boxes_acc_meter.avg

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        writer.add_scalar("val/boxiou", boxiou, epoch)
        writer.add_scalar("val/boxacc", boxacc, epoch)
        print("giou: {:.4f}, ciou: {:.4f}, boxiou: {:.4f}, boxacc: {:.3f}".format(giou, ciou, boxiou, boxacc))

    return giou, ciou, boxiou, boxacc

curr_global_steps = 0
def main():
    cfg, training_args = prepare_args()
    # if training_args.local_rank == 0:
    #     os.makedirs(training_args.output_dir, exist_ok=True)
    #     writer = SummaryWriter(training_args.output_dir)
    # else:
    writer = None
    model, preprocessor = load_pretrained(cfg.model_args, training_args)
    tokenizer = preprocessor["text"]
    world_size = torch.cuda.device_count()
    distributed = world_size > 1
    training_args.distributed = distributed
    # Some ugly codes to inject target_processor into preprocessor.
    # maybe effect model. (e.g. add special token; resize embedding)
    model, preprocessor = prepare_target_processor(model, preprocessor, cfg.model_args, training_args)
    model = model.cuda()
    print_trainable_params(model)
    # Prepare data_collator
    collator_kwargs = cfg.data_args.collator_kwargs
    trainer_cls, data_collator_dict = prepare_trainer_collator(cfg.model_args, preprocessor, collator_kwargs)
    dataset, compute_metrics = prepare_data(cfg.data_args, cfg.model_args, training_args, preprocessor)
    train_dataset, val_dataset = dataset['train'], dataset['validation']


    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_args.per_device_eval_batch_size * world_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        sampler=None,
        collate_fn=data_collator_dict["eval_collator"],
    )
    giou, ciou, boxiou, boxacc = validate(val_loader, model, 1, writer, training_args)


# noinspection PyUnusedLocal
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
