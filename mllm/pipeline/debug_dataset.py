import os
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
import sys
import logging
import pathlib
import typing
import warnings
os.environ['CUDA_VISIBLE_DEVICES']="7"
SLURM_ENV = {k: v for k, v in os.environ.items() if 'SLURM' in k}
if SLURM_ENV:
    print(f"SLURM_ENV: {SLURM_ENV}")
project_path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_path))

import torch
import torch.cuda

from mllm.config import prepare_args
from mllm.models import load_pretrained
from mllm.utils import print_trainable_params
from mllm.engine import prepare_trainer_collator
from mllm.dataset import prepare_data, prepare_target_processor
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import transformers
from transformers import TrainerCallback
from mllm.dataset.root import DATASETS, METRICS, TRANSFORMS, FUNCTIONS
from mllm.dataset.utils.transform import norm_box_xyxy
import matplotlib.patches as patches
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

def plot_images(image, mask,  expr, box):
    import matplotlib.pyplot as plt

    # Assuming you have three images in NumPy format: image1, image2, image
    ori_size = image.size
    x1_gt, y1_gt, x2_gt, y2_gt = box
    # Create a figure with three subplots arranged horizontally
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    rect = patches.Rectangle((x1_gt, y1_gt), x2_gt - x1_gt, y2_gt - y1_gt, linewidth=2, edgecolor='r', facecolor='none')
    axs[0].add_patch(rect)
    # x1_gt, y1_gt, x2_gt, y2_gt = box
    # Display the first image in the first subplot
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("real")
    axs[0].axis('off')

    # Display the second image in the second subplot
    mask_w, mask_h = mask.shape
    x1_gt, y1_gt, x2_gt, y2_gt = norm_box_xyxy(box, w=ori_size[0], h=ori_size[1])
    x1_gt, y1_gt, x2_gt, y2_gt = x1_gt*mask_w, y1_gt*mask_h, x2_gt*mask_w, y2_gt*mask_h
    axs[1].imshow(mask)
    # rect = patches.Rectangle((x1_gt, y1_gt), x2_gt - x1_gt, y2_gt - y1_gt, linewidth=2, edgecolor='r', facecolor='none')
    rect = patches.Rectangle((x1_gt, y1_gt), x2_gt - x1_gt, y2_gt - y1_gt, linewidth=2, edgecolor='r', facecolor='none')
    axs[1].add_patch(rect)
    axs[1].set_title("gen")
    axs[1].axis('off')


    # Adjust the spacing between subplots
    plt.tight_layout()
    fig.suptitle(f"{expr}", fontsize=12)
    # Save the figure as an image file
    plt.show()  # Change the filename and format as desired
    plt.close()

def main():
    cfg, training_args = prepare_args()
    dataset = DATASETS.build(cfg.data_args.train)
    for ret in dataset:
        image = ret['image']
        mask = ret['target']["masks"]
        expr = ret['target']["expr"]
        box = ret['target']["boxes"]
        plot_images(image, mask[0].squeeze().cpu().numpy(), expr[0], box[0])



# noinspection PyUnusedLocal
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
