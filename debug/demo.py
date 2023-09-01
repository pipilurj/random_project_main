import os
import sys
import logging
import pathlib
from mllm.utils.common import decode_generate_ids
import typing
import warnings
os.environ["CUDA_VISIBLE_DEVICES"]='7'
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
from debug.process_data import process_input

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


def main(model, preprocessor, image_path,  question = ""):
    image, input_ids = process_input(image_path, question, preprocessor['text'], preprocessor['image'])
    with torch.no_grad():
        model = model.cuda()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            results = model.generate(input_ids= input_ids, images = image,
                                     max_new_tokens=4096, num_beams=1)
    return decode_generate_ids(preprocessor['text'], results)

if __name__ == "__main__":
    cfg, training_args = prepare_args()
    model, preprocessor = load_pretrained(cfg.model_args, training_args)
    # Some ugly codes to inject target_processor into preprocessor.
    # maybe effect model. (e.g. add special token; resize embedding)
    model, preprocessor = prepare_target_processor(model, preprocessor, cfg.model_args, training_args)
    print_trainable_params(model)
    model.eval()
    main(model, preprocessor, image_path = "debug/images/refridgerator.png", question= "In the image <image>, I need to find the refridgeator and know its coordinates. Can you please help?")
    main(model, preprocessor, image_path = "debug/images/crossing-street.jpg", question= "Can you provide a description of the image <image> and include the coordinates [x0,y0,x1,y1] for each mentioned object?")
