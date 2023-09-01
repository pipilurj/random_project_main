import os
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
import sys
import logging
import pathlib
import typing
import warnings
import traceback

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
torch.set_printoptions(precision=3)
from mllm.config import prepare_args
from mllm.models import load_pretrained
from mllm.utils import print_trainable_params
from mllm.engine import prepare_trainer_collator
from mllm.dataset import prepare_data, prepare_target_processor
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import transformers
from transformers import TrainerCallback
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

class LossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs.update(state.additional_losses)

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

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def main():
    cfg, training_args = prepare_args()
    model, preprocessor = load_pretrained(cfg.model_args, training_args)
    # Some ugly codes to inject target_processor into preprocessor.
    # maybe effect model. (e.g. add special token; resize embedding)
    model, preprocessor = prepare_target_processor(model, preprocessor, cfg.model_args, training_args)
    if training_args.lora_enable:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    if training_args.lora_enable:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=training_args.lora_r, lora_alpha= training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout, target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, lora_config)
    print_trainable_params(model)

    # Prepare data_collator
    collator_kwargs = cfg.data_args.collator_kwargs
    trainer_cls, data_collator_dict = prepare_trainer_collator(cfg.model_args, preprocessor, collator_kwargs)
    dataset, compute_metrics = prepare_data(cfg.data_args, cfg.model_args, training_args, preprocessor)

    # Initialize Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        tokenizer=preprocessor['text'],
        train_dataset=dataset['train'] if training_args.do_train else None,
        eval_dataset=dataset['validation'] if training_args.do_eval else None,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[LossCallback],
        **data_collator_dict,
    )

    # Training
    if training_args.do_train:
        try:
            if (not training_args.overwrite_output_dir) and list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
                train_result = trainer.train(resume_from_checkpoint=True)
            else:
                train_result = trainer.train()
            trainer.log_metrics("train", train_result.metrics)  # noqa
            trainer.save_metrics("train", train_result.metrics)  # noqa
            if training_args.lora_enable:
                state_dict = get_peft_state_maybe_zero_3(
                    model.named_parameters()
                )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    model.named_parameters()
                )
                if training_args.local_rank == 0 or training_args.local_rank == -1:
                    model.config.save_pretrained(training_args.output_dir)
                    model.save_pretrained(training_args.output_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
                    model = model.merge_and_unload()
                    model.save_pretrained(os.path.join(training_args.output_dir, "merged_weights"))
            else:
                safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)
        except RuntimeError as e:
            print(f"got RuntimeError: {e.args}")
            try:
                print(f"#### device {training_args.local_rank} summary ####\n{torch.cuda.memory_summary(training_args.local_rank)}")
            except Exception as inner_e:
                print(f"get Exception when show cuda summary: {inner_e.args}")
            raise e
        finally:
            trainer.save_state()  # noqa
            trainer.plot_loss()

    # save cfg to output_dir
    try:
        output_dir = training_args.output_dir
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        cfg.dump(os.path.join(output_dir, "cfg.py"))
    except Exception as e:
        warnings.warn(f'try to save cfg to output_dir, but get exception {e.args}')

    # Keyword arguments for `model.generate`
    gen_kwargs = dict(cfg.data_args.gen_kwargs)
    gen_kwargs.setdefault('use_cache', True)
    # important for use model.generate in batch mode. some model config with wrong special_token_id
    # (e.g. shikra generationConfig set pad_token_id to -1)
    if hasattr(cfg.model_args, 'gen_kwargs_set_pad_token_id') and cfg.model_args.gen_kwargs_set_pad_token_id:
        gen_kwargs['pad_token_id'] = preprocessor['text'].pad_token_id
    if hasattr(cfg.model_args, 'gen_kwargs_set_bos_token_id') and cfg.model_args.gen_kwargs_set_bos_token_id:
        gen_kwargs['bos_token_id'] = preprocessor['text'].bos_token_id
    if hasattr(cfg.model_args, 'gen_kwargs_set_eos_token_id') and cfg.model_args.gen_kwargs_set_eos_token_id:
        gen_kwargs['eos_token_id'] = preprocessor['text'].eos_token_id

    # Evaluation
    if training_args.do_eval:
        if hasattr(trainer, '_test_collator') and hasattr(trainer, '_eval_collator') \
                and trainer._test_collator != trainer._eval_collator:  # noqa
            warnings.warn('[WARNING!!!] use different collator for eval and test. but do_eval and '
                          'do_predict both use trainer.predict (i.e. only test_collator is used.)')
        eval_results = trainer.predict(dataset['validation'], metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", eval_results.metrics)  # noqa
        trainer.save_metrics("eval", eval_results.metrics)  # noqa
        trainer.save_prediction(eval_results, file_key_prefix='eval')

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset['test'], metric_key_prefix="test", **gen_kwargs)
        trainer.log_metrics("test", predict_results.metrics)  # noqa
        trainer.save_metrics("test", predict_results.metrics)  # noqa
        trainer.save_prediction(predict_results, file_key_prefix='test')

    # Multi Predict
    if training_args.do_multi_predict:
        old_compute_metrics = trainer.compute_metrics
        multitest = dataset['multitest']
        multitest = typing.cast(dict, multitest)
        for _idx, (k, item) in enumerate(multitest.items()):
            print(f'processing multitest set {_idx}/{len(multitest)}: {k}')
            _ds = item['dataset']
            _compute_metrics = item['compute_metric']
            _prefix = f"multitest_{k}"

            trainer.compute_metrics = _compute_metrics
            _pred_results = trainer.predict(_ds, metric_key_prefix=_prefix, **gen_kwargs)
            trainer.log_metrics(_prefix, _pred_results.metrics)  # noqa
            trainer.save_metrics(_prefix, _pred_results.metrics)  # noqa
            trainer.save_prediction(_pred_results, file_key_prefix=_prefix)
        trainer.compute_metrics = old_compute_metrics


# noinspection PyUnusedLocal
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
