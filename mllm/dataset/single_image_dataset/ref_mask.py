import sys
import logging
import warnings
from typing import Dict, Any, Sequence

import numpy
from transformers import EvalPrediction
import torch
from torchvision.ops import box_iou
from mllm.utils.common import decode_generate_ids
import os.path as osp
from PIL import Image
from torchvision import transforms
from ..utils import (
    MInstrDataset,
    BaseComputeMetrics,
)

from ..process_function import (
    BoxFormatter,
)

from ..root import (
    DATASETS,
    METRICS,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    MASKS_PLACEHOLDER,
    EXPR_PLACEHOLDER,
    OBJ_TEXT_START,
    OBJ_TEXT_END,
    OBJ_VISUAL_START,
    OBJ_VISUAL_END
)
import cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

def expand2square(pil_img, background_color=0):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def xywh2xyxy(box):
    x,y,w,h = box
    return [x,y, x+w,y+h]

@DATASETS.register_module()
class REFMaskDataset(MInstrDataset):
    def __init__(self, mask_dir = None, *args, **kwargs):
        self.mask_dir = mask_dir
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((224, 224), antialias=True)
            transforms.Resize((256, 256), antialias=True)
        ])
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))

    def get_mask(self, mask_id, dataset_name):
        mask_path = osp.join(self.mask_dir, dataset_name, f"{mask_id}.png")
        mask = Image.open(mask_path).convert("1")
        mask = expand2square(mask)
        mask = np.array(mask).astype(np.float32)
        mask = self.mask_transform(mask)
        return mask

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = item['expression']
        bbox = xywh2xyxy(item['bbox'])
        mask_id = item['segment_id']
        dataset_name = item['dataset_name']
        mask = self.get_mask(mask_id, dataset_name)
        image = self.get_image(img_path)
        segmentation = item['segmentation']
        question = self.get_template().replace(EXPR_PLACEHOLDER, expr)
        if expr == "{}":
            expr = "object"
        ret = {
            'image': image,
            'target': {
                'boxes': [bbox],
                'masks': [mask],
                "expr": [expr],
                "segments": [segmentation]
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f'Answer: The segmentation mask of {OBJ_TEXT_START}{expr}{OBJ_TEXT_END} is {OBJ_VISUAL_START}{MASKS_PLACEHOLDER}{OBJ_VISUAL_END}.',
                    'boxes_seq': [[0]],
                    'segments_seq': [[0]],
                }
            ]
        }
        return ret


@METRICS.register_module()
class RECMaskComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        failed = 0
        target_failed = 0

        pred_boxes, target_boxes = [], []
        for pred, target in zip(preds, targets):
            extract_pred = self.extract_ans(pred)
            extract_target = self.extract_ans(target)
            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans for target: {target}")
                continue
            if extract_pred is None:
                failed += 1
                logger.warning(f"failed to extract ans for pred: {pred}")
                extract_pred = [0, 0, 0, 0]
            target_boxes.append(extract_target)
            pred_boxes.append(extract_pred)

        with torch.no_grad():
            target_boxes = torch.tensor(target_boxes)
            pred_boxes = torch.tensor(pred_boxes)
            # normalized box value is too small, so that the area is 0.
            ious = box_iou(pred_boxes * 1000, target_boxes * 1000)
            ious = torch.einsum('i i -> i', ious)  # take diag elem
            # NOTE: please note iou only calculate for success target
            iou = ious.mean().item()
            correct = (ious > 0.5).sum().item()

        # HACK: currently we expand image to square. so this iou is the real iou.
        warn_message = "this iou is calculate on normalized box. just for non-rigorous training progress checking." \
                       "the value is consistent with real iou only if image.width == image.height."
        warnings.warn(warn_message)

        return {
            'accuracy': 1.0 * correct / len(targets),
            'target_failed': target_failed,
            'failed': failed,
            'iou': iou,
            'warning': warn_message,
        }

    def extract_ans(self, string: str):
        try:
            list_of_boxes = self.box_formatter.extract(string)
            if len(list_of_boxes) != 1 or len(list_of_boxes[0]) != 1:
                return None
            box = list_of_boxes[0][0]
            if len(box) != 4:
                return None
            return box
        except Exception as e:
            logger.warning(f"extract_ans for {string} but get exception: {e}")
            return None

