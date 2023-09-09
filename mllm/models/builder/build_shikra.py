from typing import Dict, Any, Tuple

import torch
import transformers
from torch import nn

from ..shikra import ShikraLlamaForCausalLM, ShikraLlamaForCausalLM2, ShikraLlamaForCausalLM3, ShikraLlamaForCausalLMMask, ShikraLlamaForCausalLMDetr, ShikraLlamaForCausalContinous
from peft import get_peft_config, LoraConfig, TaskType
from mllm.models.shikra.peft_for_shikra import get_peft_model
PREPROCESSOR = Dict[str, Any]
from mllm.dataset.root import (
    FUNCTIONS,
    BaseTargetProcessFunc,
    BOXES_PLACEHOLDER,
    MASKS_PLACEHOLDER,
    BOXES_PROCESSOR,
    POINTS_PLACEHOLDER,
    OBJ_TEXT_START,
    OBJ_TEXT_END,
    OBJ_VISUAL_START,
    OBJ_VISUAL_END,
)
from mllm.models.shikra.peft_for_shikra import PeftModelForCausalLMShikra
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def load_pretrained_shikra(model_args, training_args) -> Tuple[nn.Module, PREPROCESSOR]:
    if getattr(model_args, "type", "shikra") == "shikra":
        model = ShikraLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    elif getattr(model_args, "type", "shikra") == "shikra3":
        model = ShikraLlamaForCausalLM3.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    elif getattr(model_args, "type", "shikra") == "shikra_continuous":
        model = ShikraLlamaForCausalContinous.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    elif getattr(model_args, "type", "shikra") == "shikra_mask":
        model = ShikraLlamaForCausalLMMask.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    elif getattr(model_args, "type", "shikra") == "shikra_detr":
        model = ShikraLlamaForCausalLMDetr.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    else:
        model = ShikraLlamaForCausalLM2.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    assert model_args.version == 'v1'
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token

    model_vision_dict = model.model.initialize_vision_modules(
        vision_tower=model_args.vision_tower,
        mm_vision_select_layer=model_args.mm_vision_select_layer,
        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
    )
    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16
    # model.model.get_vision_tower().to(dtype=dtype, device=training_args.device)
    print("here")
    vision_config = model_vision_dict['vision_config']

    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.model.mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = model_args.freeze_mm_mlp_adapter
    if model_args.freeze_mm_mlp_adapter:
        for p in model.model.mm_projector.parameters():
            p.requires_grad = False

    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    vision_config.use_im_start_end = model_args.mm_use_im_start_end
    model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end,
                                      tokenizer=tokenizer,
                                      device=training_args.device,
                                      tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
                                      pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)
    # # initialize autoencoder (debug)
    if model_args.pretrained_autoencoder is not None:
        model.load_autoencoder_pretrained(model_args)
    # model.get_autoencoder().to(dtype=dtype, device=training_args.device)
    # model.get_autoencoder().to(dtype=torch.float32, device=training_args.device)
    # set loss weights
    model.set_loss_weights(model_args)
    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'.format(len(params_no_grad),
                                                                                                                 params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.format(
                    len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen parameters, this is experimental.")
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)

                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    preprocessor = dict(
        image=model_vision_dict['image_processor'],
        text=tokenizer,
        conv=dict(
            image_token_len=model_args.image_token_len,
            sep_image_conv_front=model_args.sep_image_conv_front,
            use_im_start_end=model_args.mm_use_im_start_end,
        )
    )
    tokenizer = preprocessor['text']
    # for x, y in zip(range(0, total_bin_num), range(0, total_bin_num)):
    #     tokenizer.add_tokens(f"<bin_{x}_{y}>")
    additional_special_tokens = []
    additional_special_tokens.append(f'<mask>')
    additional_special_tokens.append(f'<box>')
    additional_special_tokens.append(OBJ_VISUAL_START)
    additional_special_tokens.append(OBJ_VISUAL_END)
    additional_special_tokens.append(OBJ_TEXT_START)
    additional_special_tokens.append(OBJ_TEXT_END)
    smart_tokenizer_and_embedding_resize(
        {'additional_special_tokens': additional_special_tokens},
        tokenizer,
        model,
    )
    # model.resize_token_embeddings(len(tokenizer))
    model.record_loc_token_id(tokenizer)
    if model_args.lora_enable:
        print(f"lora enable")
        if hasattr(model, "enable_input_require_grads"):
            model.model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=model_args.lora_r, lora_alpha= model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout, target_modules=["q_proj", "v_proj"]
        )
        model.model = get_peft_model(model.model, lora_config)
        for n, p in model.model.named_parameters():
            # if any([x in n for x in ["lm_head", "embed_tokens"]]) and p.shape[0] == len(
            #         tokenizer
            # ):
            if any([x in n for x in ["lm_head", "embed_tokens"]]):
                p.requires_grad = True

    return model, preprocessor


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
