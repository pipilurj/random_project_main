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
from mllm.models.utils.modeling_outputs import CausalLMOutputWithPastCustom
from mllm.utils.box_utils import bbox_losses
from transformers.models.llama.modeling_llama import _expand_mask
from torchvision.ops.boxes import box_area, box_convert
import itertools
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
LOC_TOKENS = "<bin_{}>"


def xywh2xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def loc_to_coords(loc, num_locations):
    match = re.search(r"<bin_(\d+)>", loc)
    token_num = float(match.group(1))
    xc, yc = token_num - math.floor((token_num - 1e-3) / num_locations)*num_locations, math.ceil(token_num / num_locations)
    xc_coord, yc_coord = (xc-0.5)*1/num_locations, (yc-0.5)*1/num_locations
    return xc_coord, yc_coord

def box_loc_to_offsets(box, loc, num_locations):
    xc_coord, yc_coord = loc_to_coords(loc, num_locations)
    x1_off, y1_off, x2_off, y2_off = xc_coord - box[0], yc_coord - box[1], box[2]-xc_coord, box[3]-yc_coord
    return [x1_off, y1_off, x2_off, y2_off]

def loc_offsets_to_bbox(loc, offsets, num_locations):
    xc_coord, yc_coord = loc_to_coords(loc, num_locations)
    x1,y1,x2,y2 = xc_coord - offsets[0], yc_coord - offsets[1], xc_coord + offsets[2], yc_coord + offsets[3]
    return [x1,y1,x2,y2]

# def box_coord_to_offsets(box, coord):
#     xc_coord, yc_coord = coord.unbind(-1)
#     x1, y1, x2, y2 = box.unbind(-1)
#     x1_off, y1_off, x2_off, y2_off = xc_coord - x1, yc_coord - y1, x2-xc_coord, y2-yc_coord
#     offsets = [x1_off, y1_off,
#          x2_off, y2_off]
#     return torch.stack(offsets, dim=-1)
#
# def coord_offsets_to_bbox(coord, offsets):
#     xc_coord, yc_coord = coord.unbind(-1)
#     x1_off, y1_off, x2_off, y2_off = offsets.unbind(-1)
#     boxes = [xc_coord-x1_off, yc_coord-y1_off,
#              xc_coord+x2_off, yc_coord+ y2_off]
#     return torch.stack(boxes, dim=-1)

def box_coord_to_offsets(box, coord):
    xc_coord, yc_coord = coord.unbind(-1)
    x1, y1, x2, y2 = box.unbind(-1)
    x1_off, y1_off, x2_off, y2_off = x1 - xc_coord, y1 - yc_coord, x2-xc_coord, y2-yc_coord
    offsets = [x1_off, y1_off,
               x2_off, y2_off]
    return torch.stack(offsets, dim=-1)

def coord_offsets_to_bbox(coord, offsets):
    xc_coord, yc_coord = coord.unbind(-1)
    x1_off, y1_off, x2_off, y2_off = offsets.unbind(-1)
    boxes = [xc_coord+x1_off, yc_coord+y1_off,
             xc_coord+x2_off, yc_coord+ y2_off]
    return torch.stack(boxes, dim=-1)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
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
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
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
                                                              cur_input_embeds[image_start_token_pos:image_start_token_pos + 1],
                                                              cur_image_features, cur_input_embeds[
                                                                                  image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2],
                                                              cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos + 1], cur_image_features,
                                                              cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_patch_token).sum() != num_patches:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(cur_input_ids == vision_tower.config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start + num_patches, device=masked_indices.device,
                                                       dtype=masked_indices.dtype)).any():
                        raise ValueError("The image patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_image_features,
                                                          cur_input_embeds[mask_index_start + num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat(
                            (cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start + num_patches:]),
                            dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(ShikraLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class ShikraLlamaForCausalLM3(LlamaForCausalLM):
    config_class = ShikraConfig

    def __init__(self, config: ShikraConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ShikraLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # define bbox head and mask head
        self.init_bbox_head(config)
        # Initialize weights and apply final processing
        self.post_init()

    def record_loc_token_id(self, tokenizer, num_loc_tokens):
        self.tokenizer = tokenizer
        self.loc_tokens = [f"<bin_{i+1}>" for i in range(num_loc_tokens**2)]
        self.loc_tokens_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in self.loc_tokens]
        self.loc_id_to_coord, self.loc_token_to_coord = {}, {}
        for loc_token, loc_id in zip(self.loc_tokens,  self.loc_tokens_ids):
            center_coord = loc_to_coords(loc_token, num_loc_tokens)
            self.loc_id_to_coord.update({loc_id: center_coord})
            self.loc_token_to_coord.update({loc_token: center_coord})

        self.model.loc_tokens_ids = self.loc_tokens_ids

    def record_ret_token_id(self, tokenizer):
        self.tokenizer = tokenizer
        self.ret_token = "<ret>"
        self.ret_tokens_ids = tokenizer.encode(self.ret_token, add_special_tokens=False)[0]
        self.model.ret_tokens_ids = self.ret_tokens_ids

    def init_bbox_head(self, config):
        self.bbox_head = MLP(config.hidden_size, config.hidden_size, 4, 3)

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
        # return nn.MSELoss(reduction="sum")(pred, gt)
        # return torchvision.ops.generalized_box_iou_loss(pred, gt, reduction="mean")
        # return giou_loss(pred, gt)
        # loss_bbox = nn.L1Loss(reduction="mean")(pred, gt)
        # xyxy_pred = xywh2xyxy(pred)
        # loss_bbox = F.l1_loss(xyxy_pred, gt, reduction='none')
        # loss_giou = giou_loss(xyxy_pred, gt, ifxyxy=True)
        # loss_bbox = loss_bbox.sum() / pred.shape[0]
        # loss_giou = loss_giou.sum() / pred.shape[0]
        return nn.L1Loss(reduction="mean")(pred, gt)

    def get_loc_ids_pos(self, input_ids):
        pos_of_loc_tokens, loc_token_ids = [], []
        input_ids = input_ids.detach().cpu().numpy()
        for ids in input_ids:
            pos, loc = [], []
            for i, id in enumerate(ids):
                if id in self.loc_tokens_ids:
                    pos.append(i)
                    loc.append(id)
            if len(loc) == 0:
                print(f"input ids have no location token! {ids}")
                print(f"decoded ids {self.tokenizer.decode(ids)}")
            pos_of_loc_tokens.append(pos)
            loc_token_ids.extend(loc)
        return pos_of_loc_tokens, loc_token_ids

    def decode_box_from_pred(self, center_coords, offset_pred):
        return coord_offsets_to_bbox(center_coords, offset_pred)

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
            boxes_seq = None,
            mask = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images
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
            pos_of_loc_tokens, loc_token_ids = self.get_loc_ids_pos(input_ids)
            boxes_seq = torch.cat(boxes_seq)
            # for ids in input_ids:
            #     pos_of_loc_tokens.append([i for i, id in enumerate(ids) if id in self.loc_tokens_ids])
            #     loc_token_ids.extend([id for i, id in enumerate(ids) if id in self.loc_tokens_ids])
                # pos_of_loc_tokens.append([-1 for i, id in enumerate(ids) if id in self.loc_tokens_ids])
                # pos_of_loc_tokens.append([i for i, id in enumerate(ids) if id == self.ret_tokens_ids])
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            center_coords = torch.tensor([self.loc_id_to_coord[id.item()] for id in loc_token_ids]).to(input_ids.device)
            offset_gt = box_coord_to_offsets(boxes_seq, center_coords)
            assert len(boxes_seq) == len(boxes_seq)
            # loc_embeddings = torch.stack(list(itertools.chain(*[[hidden_states[i,:x+1,:].mean(dim=0) for x in id] for i, id in enumerate(pos_of_loc_tokens)])))
            loc_embeddings = torch.cat([hidden_states[i,id,:] for i, id in enumerate(pos_of_loc_tokens)])
            # loc_embeddings = torch.stack([hidden_states[i,:id[0],:].mean(dim=0) for i, id in enumerate(pos_of_loc_tokens)])
            offset_pred = self.bbox_head(loc_embeddings)#.sigmoid() # x, y, w, h
            # box_loc_pred_reformat = box_convert(box_loc_pred, "xywh", "xyxy")
            bbox_loss = self.box_reg_loss(offset_pred, offset_gt)
            loss += bbox_loss
            box_loc_pred_reformat = self.decode_box_from_pred(center_coords, offset_pred)
            for i, (pred, gt, embed) in enumerate(zip(box_loc_pred_reformat.data, boxes_seq.data, loc_embeddings.data)):
                # print(f"pred: {pred}, gt: {gt}, embed mean: {embed.mean()}, embed sum: {embed.sum()}, embed std: {embed.std()}")
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
