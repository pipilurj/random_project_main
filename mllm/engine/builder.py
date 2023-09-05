from functools import partial
from typing import Tuple, Dict, Any, Type

from transformers.trainer import DataCollator

from .shikra import ShikraTrainer
from .shikra2 import ShikraTrainer2
from .shikra_mask import ShikraTrainerMask
from .shikra_continuous import ShikraTrainerContinuous
from .base_engine import TrainerForMMLLM, Seq2Seq2DataCollatorWithImage

TYPE2TRAINER = {
    'shikra': ShikraTrainer,
    'shikra2': ShikraTrainer2,
    'shikra3': ShikraTrainer2,
    'shikra_mask': ShikraTrainerMask,
    "shikra_detr": ShikraTrainer2,
    "shikra_continuous": ShikraTrainerContinuous,
}


def prepare_trainer_collator(
        model_args,
        preprocessor: Dict[str, Any],
        collator_kwargs: Dict[str, Any]
) -> Tuple[Type[TrainerForMMLLM], Dict[str, DataCollator]]:
    type_ = model_args.type
    trainer_cls = TYPE2TRAINER[type_]
    data_collator_func = partial(
        Seq2Seq2DataCollatorWithImage,
        preprocessor=preprocessor,
        **collator_kwargs,
    )
    data_collator_dict = {
        "train_collator": data_collator_func(inference_mode=False),
        "eval_collator": data_collator_func(inference_mode=True),
    }
    return trainer_cls, data_collator_dict
