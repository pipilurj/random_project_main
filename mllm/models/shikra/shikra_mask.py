import math
import os.path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, CLIPVisionModel, CLIPImageProcessor
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from mllm.models.utils.modeling_outputs import CausalLMOutputWithPastCustom, GreedySearchDecoderOnlyOutputCustom
# from mllm.models.autoencoder.model.resnet import ResNet50
from mllm.models.autoencoder.model.resnet_layernorm import ResNet50
from mllm.utils.dice_loss import dice_loss
from transformers.generation.logits_process import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
)
from transformers.generation.utils import (
    GreedySearchEncoderDecoderOutput,
    GreedySearchDecoderOnlyOutput,
    StoppingCriteriaList
)
import torch.distributed as dist
import matplotlib.pyplot as plt
from mllm.dataset.root import (
    OBJ_TEXT_START,
    OBJ_TEXT_END,
    OBJ_VISUAL_START,
    OBJ_VISUAL_END
)
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
LOC_TOKENS = "<bin_{}>"

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, ifsigmoid=False):
        super().__init__()
        self.num_layers = num_layers
        self.ifsigmoid = ifsigmoid
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.ifsigmoid:
            return F.sigmoid(x)
        else:
            return x

class ShikraConfig(LlamaConfig):
    model_type = "shikra"


class ShikraLlamaModel(LlamaModel):
    config_class = ShikraConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super(ShikraLlamaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            # HACK: for FSDP
            self.vision_tower = [CLIPVisionModel.from_pretrained(config.mm_vision_tower)]
            # self.vision_tower = CLIPVisionModel.from_pretrained(config.mm_vision_tower)

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, vision_tower, mm_vision_select_layer,
                                  pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False, fsdp=None):
        self.config.mm_vision_tower = vision_tower

        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        if not hasattr(self, 'vision_tower'):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower[0]
        vision_tower.requires_grad_(False)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config
        )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels=None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            loc_embedding_mapped=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        if loc_embedding_mapped is None:
            loc_embedding_mapped = [None]  * len(input_ids)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower = self.get_vision_tower()
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            with torch.no_grad():
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        image_forward_out = vision_tower(image.unsqueeze(0), output_hidden_states=True)
                        select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                        select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
                        image_feature = select_hidden_state[:, 1:]
                        image_features.append(image_feature)
                else:
                    image_forward_outs = vision_tower(images.to(vision_tower.dtype), output_hidden_states=True)
                    select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                    select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                    image_features = select_hidden_state[:, 1:].to(images.dtype)
            self.image_features = image_features
            if type(images) is list:
                image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
            else:
                image_features = self.mm_projector(image_features)
            dummy_image_features = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds, loc_embedding in zip(input_ids, inputs_embeds, loc_embedding_mapped):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if vision_tower.config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_start_token).sum() != (
                            cur_input_ids == vision_tower.config.im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    image_start_tokens = torch.where(cur_input_ids == vision_tower.config.im_start_token)[0]
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != vision_tower.config.im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(),
                                                              cur_input_embeds[
                                                              image_start_token_pos:image_start_token_pos + 1],
                                                              cur_image_features, cur_input_embeds[
                                                                                  image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2],
                                                              cur_input_embeds[
                                                              image_start_token_pos + num_patches + 2:].detach()),
                                                             dim=0)
                        else:
                            cur_new_input_embeds = torch.cat(
                                (cur_input_embeds[:image_start_token_pos + 1], cur_image_features,
                                 cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                        cur_image_idx += 1
                    if loc_embedding is not None:
                        cur_new_input_embeds[cur_input_ids==self.mask_token_ids] = loc_embedding.to(cur_new_input_embeds.dtype)
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_patch_token).sum() != num_patches:
                        raise ValueError(
                            "The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(cur_input_ids == vision_tower.config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start + num_patches,
                                                       device=masked_indices.device,
                                                       dtype=masked_indices.dtype)).any():
                        raise ValueError("The image patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat(
                            (cur_input_embeds[:mask_index_start].detach(), cur_image_features,
                             cur_input_embeds[mask_index_start + num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat(
                            (cur_input_embeds[:mask_index_start], cur_image_features,
                             cur_input_embeds[mask_index_start + num_patches:]),
                            dim=0)
                    if loc_embedding is not None:
                        cur_new_input_embeds[cur_input_ids==self.mask_token_ids] = loc_embedding.to(cur_new_input_embeds.dtype)
                    new_input_embeds.append(cur_new_input_embeds)

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(ShikraLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class ShikraLlamaForCausalLMMask(LlamaForCausalLM):
    config_class = ShikraConfig

    def __init__(self, config: ShikraConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ShikraLlamaModel(config)
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # define bbox head and mask head
        # Initialize weights and apply final processing
        self.post_init()
        self.init_bbox_head(config)
        self.init_autoencoder()

    def record_loc_token_id(self, tokenizer):
        self.tokenizer = tokenizer
        self.mask_token = "<mask>"
        self.box_token = "<box>"
        self.mask_token_ids = tokenizer.encode(self.mask_token, add_special_tokens=False)[0]
        self.box_token_ids = tokenizer.encode(self.box_token, add_special_tokens=False)[0]
        self.obj_visual_start_id = tokenizer.encode(OBJ_VISUAL_START, add_special_tokens=False)[0]
        self.obj_visual_end_id = tokenizer.encode(OBJ_VISUAL_END, add_special_tokens=False)[0]
        self.model.mask_token_ids = self.mask_token_ids

    def init_autoencoder(self):
        self.autoencoder = ResNet50()
        self.decoder_mapping = nn.Linear(4096, 4096)
        self.encoder_mapping = nn.Linear(4096, 4096)

    def load_autoencoder_pretrained(self, config):
        # TODO: write config, pretrained path, etc.
        print( f"pretrained {config.pretrained_autoencoder}")
        print( f"freeze autoencoder {config.freeze_autoencoder}")
        pretrained = config.pretrained_autoencoder
        if pretrained is not None:
            self.autoencoder.load_state_dict(torch.load(pretrained))
        if config.freeze_autoencoder:
            self.autoencoder.eval()
            for p in self.autoencoder.parameters():
                p.requires_grad = False

    def get_model(self):
        return self.model

    def get_autoencoder(self):
        return self.autoencoder

    def set_loss_weights(self, model_args):
        self.lm_loss_weight = getattr(model_args, "lm_loss_weight", 1)
        self.recon_loss_weight = getattr(model_args, "recon_loss_weight", 1)
        self.l2_loss_weight = getattr(model_args, "l2_loss_weight", 1)
        self.box_loss_weight = getattr(model_args, "box_loss_weight", 1)
        print(f"lm_loss_weight {self.lm_loss_weight}")
        print(f"recon_loss_weight {self.recon_loss_weight}")
        print(f"l2_loss_weight {self.l2_loss_weight}")
        print(f"box_loss_weight {self.box_loss_weight}")

    def get_vision_tower(self):
        model = self.get_model()
        vision_tower = model.vision_tower
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower


    def init_bbox_head(self, config):
        self.bbox_head = MLP(config.hidden_size, config.hidden_size, 4, 3, ifsigmoid=True)
        nn.init.constant_(self.bbox_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_head.layers[-1].bias.data, 0)

    def mask_loss(self, reconstruction, mask_pixel, pred_embed, gt_embed):
        dice = dice_loss(F.sigmoid(reconstruction.squeeze(1)), mask_pixel.squeeze(1).float(), multiclass=False)
        reconstruction = reconstruction.view(-1)
        target = mask_pixel.view(-1)
        loss_reconstruction = F.binary_cross_entropy_with_logits(reconstruction, target) + dice
        l2_loss = F.mse_loss(pred_embed, gt_embed)
        return loss_reconstruction, l2_loss

    def box_loss(self, pred, gt):
        # losses = bbox_losses(pred, gt)
        # l1_loss, giou_loss = losses['loss_bbox'], losses['loss_giou']
        # return l1_loss# + giou_loss
        return nn.L1Loss(reduction="mean")(pred, gt)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            boxes_seq=None,
            masks_seq=None,
            points_seq=None,
            loc_embedding_mapped=None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        device = next(self.parameters()).device
        masks_seq_batch = masks_seq
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # if self.training:
        masks_seq = torch.cat([torch.cat([torch.cat([masks for masks in masks_seq]) for masks_seq in masks_seqs]) for masks_seqs in masks_seq_batch]).to(device)
        _, pretrained_loc_embedding = self.autoencoder(masks_seq.unsqueeze(dim=1), return_embedding=True)
        loc_embedding_mapped = self.encoder_mapping(pretrained_loc_embedding)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            loc_embedding_mapped=loc_embedding_mapped
        )
        hidden_states = outputs[0]
        loss, loss_lm, box_loss, mask_loss, l2_loss, recon_loss = 0, 0, 0, 0, 0, 0
        logits = self.lm_head(hidden_states)
        mask_decoded = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # input: <xxx><start><mask><end>;  target: <start><mask><end><xxx>
            # shift_labels[shift_labels == self.mask_token_ids] = -100
            # shift_labels[shift_labels == self.obj_visual_end_id] = -100
            # Enable model/pipeleine parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss_lm = loss_fct(shift_logits, shift_labels)
            # loss += loss_lm
            loss += loss_lm * self.lm_loss_weight
        # if self.training:
        # pos_of_loc_tokens = torch.where(input_ids==self.obj_visual_start_id)
        pos_of_loc_tokens = torch.where(input_ids==self.mask_token_ids)
        # loc_embeddings = self.encoding_mapping(hidden_states[pos_of_loc_tokens[0], pos_of_loc_tokens[1]-1])
        loc_embeddings = self.decoder_mapping(hidden_states[pos_of_loc_tokens[0], pos_of_loc_tokens[1] - 1])
        box_pred = self.bbox_head(loc_embeddings)
        boxes_seq = torch.tensor(boxes_seq).to(device).reshape(-1, 4)
        box_loss = self.box_loss(box_pred, boxes_seq) * self.box_loss_weight
        loss += box_loss
        mask_decoded = self.autoencoder.generate(loc_embeddings, ifsigmoid=False)
        recon_loss, l2_loss = self.mask_loss(mask_decoded, masks_seq, loc_embeddings, pretrained_loc_embedding)
        mask_loss = recon_loss * self.recon_loss_weight + l2_loss * self.l2_loss_weight
        loss += mask_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastCustom(
            loss=loss,
            loss_lm=loss_lm,
            loss_mask=mask_loss,
            loss_l2=l2_loss,
            loss_recon=recon_loss,
            loss_bbox=box_loss,
            pred_masks=mask_decoded.sigmoid(),
            pred_boxes=box_pred,
            # loss_iou=iou_loss,
            logits=logits,
            # logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )

    def greedy_search(
            self,
            input_ids: torch.LongTensor,
            logits_processor=None,
            stopping_criteria=None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            **model_kwargs,
    ):
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        boxes_seq, masks_seq, points_seq, loc_embeds = [[] for _ in range(len(input_ids))], [[] for _ in range(len(input_ids))], [[] for _ in range(len(input_ids))], [[] for _ in range(len(input_ids))]
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs.update({"boxes_seq": boxes_seq})
            model_inputs.update({"masks_seq": masks_seq})
            model_inputs.update({"points_seq": points_seq})
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            last_hidden_state = outputs.last_hidden_state
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            # debug
            for i, tok in enumerate(next_tokens):
                if tok == self.mask_token_ids:
                    loc_embeds[i].append(tok)
                    pos_of_loc_tokens = torch.where(input_ids==self.mask_token_ids)
                    # loc_embeddings = self.encoding_mapping(hidden_states[pos_of_loc_tokens[0], pos_of_loc_tokens[1]-1])
                    loc_embeddings = self.decoder_mapping(last_hidden_state[i, -1])
                    masks_seq[i].append(self.autoencoder.generate(encoder_repr=loc_embeddings))
                    boxes_seq[i].append(self.bbox_head(loc_embeddings))


            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            return GreedySearchDecoderOnlyOutputCustom(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                boxes_seq = boxes_seq,
                masks_seq = masks_seq
            )
        else:
            return input_ids, masks_seq, boxes_seq

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        vision_config = self.get_vision_tower().config
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.model.orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
