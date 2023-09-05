import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import re
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, CLIPVisionModel, CLIPImageProcessor
import torch.nn.functional as F
import torchvision
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from mllm.models.utils.modeling_outputs import CausalLMOutputWithPastCustom, GreedySearchDecoderOnlyOutputCustom
from mllm.utils.box_utils import bbox_losses
from transformers.models.llama.modeling_llama import _expand_mask
from torchvision.ops.boxes import box_area, box_convert
import itertools
import numpy as np
import torch.distributed as dist
from transformers.generation.logits_process import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
)
from transformers.generation.utils import (
    GreedySearchEncoderDecoderOutput,
    GreedySearchDecoderOnlyOutput,
    StoppingCriteriaList
)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
LOC_TOKENS = "<bin_{}>"


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, ifsigmoid=False):
        super().__init__()
        self.ifsigmoid = ifsigmoid
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.ifsigmoid:
            x = F.sigmoid(x)
        return x


# def xywh_to_xyxy(box):
#     # box: (x, y, w, h)
#     x1 = max(0, box[0]-box[2]/2)
#     y1 = max(0, box[1]-box[3]/2)
#     x2 = min(1, box[0]+box[2]/2)
#     y2 = min(1, box[1]+box[3]/2)
#     return

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
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            locs: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

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
            if type(images) is list:
                image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
            else:
                image_features = self.mm_projector(image_features)
            dummy_image_features = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds, cur_loc_embeds in zip(input_ids, inputs_embeds, locs):
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
                    if len(cur_loc_embeds) > 0:
                        loc_indices = torch.nonzero(cur_input_ids == self.loc_tokens_ids).flatten()
                        cur_new_input_embeds[loc_indices] = cur_loc_embeds
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
                    if len(cur_loc_embeds) > 0:
                        loc_indices = torch.nonzero(cur_input_ids == self.loc_tokens_ids).flatten()
                        cur_new_input_embeds[loc_indices] = cur_loc_embeds
                    new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(ShikraLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class ShikraLlamaForCausalContinous(LlamaForCausalLM):
    config_class = ShikraConfig

    def __init__(self, config: ShikraConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.config = config
        self.model = ShikraLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # define bbox head and mask head
        self.init_bbox_head(config)
        # Initialize weights and apply final processing
        self.post_init()

    def record_loc_token_id(self, tokenizer, num_loc_tokens):
        self.num_bins = num_loc_tokens
        self.tokenizer = tokenizer
        self.loc_tokens = "<loc>"
        self.loc_start_tokens = "<loc_start>"
        self.loc_tokens_ids = tokenizer.encode(self.loc_tokens, add_special_tokens=False)[0]
        self.model.loc_tokens_ids = self.loc_tokens_ids
        self.loc_to_locid, self.locid_to_loc = dict(), dict()
        num = 0
        for x in range(0, num_loc_tokens):
            for y in range(0, num_loc_tokens):
                self.loc_to_locid.update({f"<bin_{x}_{y}>": num})
                self.locid_to_loc.update({num: f"<bin_{x}_{y}>"})
                num += 1
        self.loc_embeds = torch.nn.Embedding((num_loc_tokens) ** 2, self.config.hidden_size)
        # input_embeddings = self.get_input_embeddings().weight.data
        # output_embeddings = self.get_output_embeddings().weight.data
        #
        # input_embeddings_avg = input_embeddings[:-1].mean(dim=0, keepdim=True)
        # output_embeddings_avg = output_embeddings[:-1].mean(dim=0, keepdim=True)
        #
        # input_embeddings[-1:] = input_embeddings_avg
        # output_embeddings[-1:] = output_embeddings_avg
        # self.loc_embeds.weight.data = input_embeddings_avg.repeat(self.loc_embeds.weight.size(0), 1)

    def record_ret_token_id(self, tokenizer):
        self.tokenizer = tokenizer
        self.ret_token = "<ret>"
        self.ret_tokens_ids = tokenizer.encode(self.ret_token, add_special_tokens=False)[0]
        self.model.ret_tokens_ids = self.ret_tokens_ids

    def init_bbox_head(self, config):
        self.point_head = MLP(config.hidden_size, config.hidden_size, 2, 3, ifsigmoid=True)
        nn.init.constant_(self.point_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.point_head.layers[-1].bias.data, 0)
        # self.point_head = nn.Linear(config.hidden_size, 2)

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_vision_tower(self):
        model = self.get_model()
        vision_tower = model.vision_tower
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def box_reg_loss(self, pred, gt):
        return nn.L1Loss(reduction="mean")(pred, gt)

    def format_loc(self, locations):
        quant_loc11_list, quant_loc21_list, quant_loc12_list, quant_loc22_list, delta_x1_list, delta_y1_list, delta_x2_list, delta_y2_list, region_coord11_list, region_coord21_list, region_coord12_list, region_coord22_list = \
            [], [], [], [], [], [], [], [], [], [], [], []
        for loc in locations:
            loc = [loc[i] * (self.num_bins - 1) for i in range(2)]
            quant_loc11_list.append([math.floor(loc[0]), math.floor(loc[1])])
            quant_loc21_list.append([math.ceil(loc[0]), math.floor(loc[1])])
            quant_loc12_list.append([math.floor(loc[0]), math.ceil(loc[1])])
            quant_loc22_list.append([math.ceil(loc[0]), math.ceil(loc[1])])
            delta_x1_list.append(loc[0] - math.floor(loc[0]))
            delta_y1_list.append(loc[1] - math.floor(loc[1]))
            delta_x2_list.append(1 - loc[0] + math.floor(loc[0]))
            delta_y2_list.append(1 - loc[1] + math.floor(loc[1]))
            region_coord11_list.append(f"<bin_{int(quant_loc11_list[-1][0])}_{int(quant_loc11_list[-1][1])}>")
            region_coord21_list.append(f"<bin_{int(quant_loc21_list[-1][0])}_{int(quant_loc21_list[-1][1])}>")
            region_coord12_list.append(f"<bin_{int(quant_loc12_list[-1][0])}_{int(quant_loc12_list[-1][1])}>")
            region_coord22_list.append(f"<bin_{int(quant_loc22_list[-1][0])}_{int(quant_loc22_list[-1][1])}>")
        return quant_loc11_list, quant_loc21_list, quant_loc12_list, quant_loc22_list, delta_x1_list, delta_y1_list, delta_x2_list, delta_y2_list, region_coord11_list, region_coord21_list, region_coord12_list, region_coord22_list

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
            mask=None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # boxes_seq: batch x [num seq x [ num boxes []]]
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if boxes_seq is not None:
            loc_embedding_11_id_batch, loc_embedding_12_id_batch, loc_embedding_21_id_batch, loc_embedding_22_id_batch, delta_x1_batch, delta_y1_batch, delta_x2_batch, delta_y2_batch, coords_batch = \
                [], [], [], [], [], [], [], [], []
            loc_seq_batch = boxes_seq
            for locs in loc_seq_batch:
                if len(locs) > 0:
                    loc_embedding_11_id, loc_embedding_12_id, loc_embedding_21_id, loc_embedding_22_id, delta_x1, delta_y1, delta_x2, delta_y2, coords = \
                        [], [], [], [], [], [], [], [], []
                    # for locs in locs_seq:
                    quant_loc11, quant_loc21, quant_loc12, quant_loc22, d_x1, d_y1, d_x2, d_y2, region_coord11, region_coord21, region_coord12, region_coord22 = self.format_loc(
                        locs)
                    loc_embedding_11_id.extend(
                        [self.loc_to_locid[r] for r in
                         region_coord11])
                    loc_embedding_12_id.extend(
                        [self.loc_to_locid[r] for r in
                         region_coord12])
                    loc_embedding_21_id.extend(
                        [self.loc_to_locid[r] for r in
                         region_coord21])
                    loc_embedding_22_id.extend(
                        [self.loc_to_locid[r] for r in
                         region_coord22])
                    delta_x1.extend(d_x1)
                    delta_y1.extend(d_y1)
                    delta_x2.extend(d_x2)
                    delta_y2.extend(d_y2)
                    loc_embedding_11_id_batch.append(loc_embedding_11_id)
                    loc_embedding_12_id_batch.append(loc_embedding_12_id)
                    loc_embedding_21_id_batch.append(loc_embedding_21_id)
                    loc_embedding_22_id_batch.append(loc_embedding_22_id)
                    delta_x1_batch.append(delta_x1)
                    delta_x2_batch.append(delta_x2)
                    delta_y1_batch.append(delta_y1)
                    delta_y2_batch.append(delta_y2)
            if sum([len(x) for x in loc_seq_batch]) > 0:
                loc_embedding_11 = self.loc_embeds(
                    torch.cat([torch.tensor(x) for x in loc_embedding_11_id_batch]).to(input_ids.device))
                loc_embedding_12 = self.loc_embeds(
                    torch.cat([torch.tensor(x) for x in loc_embedding_12_id_batch]).to(input_ids.device))
                loc_embedding_21 = self.loc_embeds(
                    torch.cat([torch.tensor(x) for x in loc_embedding_21_id_batch]).to(input_ids.device))
                loc_embedding_22 = self.loc_embeds(
                    torch.cat([torch.tensor(x) for x in loc_embedding_22_id_batch]).to(input_ids.device))

                delta_x1_concat, delta_x2_concat, delta_y1_concat, delta_y2_concat = \
                    torch.cat([torch.tensor(d) for d in delta_x1_batch]).to(input_ids.device).unsqueeze(-1).repeat( 1,
                                                                                                                   loc_embedding_11.shape[
                                                                                                                       -1]), torch.cat(
                        [torch.tensor(d) for d in delta_x2_batch]).to(input_ids.device).unsqueeze(-1).repeat(1,
                                                                                                             loc_embedding_11.shape[
                                                                                                                 -1]), \
                    torch.cat([torch.tensor(d) for d in delta_y1_batch]).to(input_ids.device).unsqueeze(-1).repeat(1,
                                                                                                                   loc_embedding_11.shape[
                                                                                                                       -1]), torch.cat(
                        [torch.tensor(d) for d in delta_y2_batch]).to(input_ids.device).unsqueeze(-1).repeat(1,
                                                                                                             loc_embedding_11.shape[
                                                                                                                 -1])
                delta_x1_concat, delta_x2_concat, delta_y1_concat, delta_y2_concat = delta_x1_concat.to(
                    loc_embedding_22.dtype), delta_x2_concat.to(loc_embedding_22.dtype), delta_y1_concat.to(
                    loc_embedding_22.dtype), delta_y2_concat.to(loc_embedding_22.dtype)
                all_loc_embeddings = loc_embedding_11 * delta_x2_concat * delta_y2_concat + loc_embedding_12 * delta_x2_concat * delta_y1_concat + \
                                     loc_embedding_21 * delta_x1_concat * delta_y2_concat + loc_embedding_22 * delta_x1_concat * delta_y1_concat

                # batchify loc embedding:
                loc_embedding_batched = []
                start_index = 0
                for seq in loc_embedding_11_id_batch:
                    loc_embedding_batched.append(all_loc_embeddings[start_index:start_index + len(seq)])
                    start_index += len(seq)
            else:
                loc_embedding_batched = [[] for _ in range(len(input_ids))]
        else:
            loc_embedding_batched = [[] for _ in range(len(input_ids))]

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
            locs=loc_embedding_batched
            # locs=[[] for _ in range(len(input_ids))]
        )

        hidden_states = outputs[0]
        loss, loss_lm, bbox_loss = 0, 0, 0
        logits = self.lm_head(hidden_states)
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeleine parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss_lm = loss_fct(shift_logits, shift_labels)
            loss += loss_lm

        if self.training:
            # DEBUG: the position may be one position forward?
            pos_of_loc_tokens = torch.where(input_ids==self.loc_tokens_ids)
            loc_embedding_out = hidden_states[pos_of_loc_tokens[0], pos_of_loc_tokens[1]-1]
            coords_gt = torch.cat([torch.tensor(c) for c in loc_seq_batch]).to(input_ids.device)
            coords_pred = self.point_head(loc_embedding_out)  # .sigmoid() # x, y, w, h
            # box_loc_pred_reformat = box_convert(box_loc_pred, "xywh", "xyxy")
            bbox_loss = self.box_reg_loss(coords_pred, coords_gt)
            loss += bbox_loss
            box_loc_pred = coords_pred.reshape(-1, 4)
            for i, (gt, pred) in enumerate(zip(coords_gt.view(-1,4).data, box_loc_pred.data)):
                print(f"pred: {pred}, gt: {gt}")
                if i > 3:
                    break
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastCustom(
            loss=loss,
            loss_lm=loss_lm,
            loss_bbox=bbox_loss,
            # loss_iou=iou_loss,
            logits=logits,
            # logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )

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
        locs_seq, loc_embeds = [[] for _ in range(len(input_ids))], [[] for _ in range(len(input_ids))]
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
            model_inputs.update({"boxes_seq": locs_seq})
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
            for i, tok in enumerate(next_tokens):
                if tok == self.loc_tokens_ids:
                    loc_embeds[i].append(tok)
                    locs_seq[i].append(self.point_head(last_hidden_state[i][-1]).detach().to(torch.float32).cpu().numpy().tolist())


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
                boxes_seq = locs_seq
            )
        else:
            return input_ids, locs_seq
