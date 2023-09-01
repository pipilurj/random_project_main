import math
import os.path
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
import matplotlib.patches as patches
from mllm.models.autoencoder.model.transformer_mask import TransformerMask
from mllm.models.autoencoder.model.resnet import ResNet50
from mllm.utils.dice_loss import dice_loss
import matplotlib.pyplot as plt
from .detr_transformer import Transformer
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

def plot_images(real, gen, noise, coord_gt, coord_pred, save_path="", imgid=0):
    w, h = real.shape
    x1_gt, y1_gt, x2_gt, y2_gt = coord_gt[0]*w, coord_gt[1]*h, coord_gt[2]*w, coord_gt[3]*h
    x1_pred, y1_pred, x2_pred, y2_pred = coord_pred[0]*w, coord_pred[1]*h, coord_pred[2]*w, coord_pred[3]*h
    # Assuming you have three images in NumPy format: image1, image2, image3

    # Create a figure with three subplots arranged horizontally
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Display the first image in the first subplot
    axs[0].imshow(real)
    rect = patches.Rectangle((x1_gt, y1_gt), x2_gt - x1_gt, y2_gt - y1_gt, linewidth=2, edgecolor='r', facecolor='none')
    # Add the rectangle patch to the axes
    axs[0].add_patch(rect)
    axs[0].set_title("real")
    axs[0].axis('off')

    # Display the second image in the second subplot
    axs[1].imshow(gen)
    axs[1].set_title("gen")
    rect = patches.Rectangle((x1_pred, y1_pred), x2_pred - x1_pred, y2_pred - y1_pred, linewidth=2, edgecolor='r', facecolor='none')
    axs[1].add_patch(rect)
    axs[1].axis('off')

    # Display the third image in the third subplot
    axs[2].imshow(noise)
    axs[2].set_title("noise")
    axs[2].axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the figure as an image file
    plt.savefig(os.path.join(save_path, f"masks_{imgid}.png"), dpi=300)  # Change the filename and format as desired
    plt.close()

def xywh2xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def loc_to_coords(loc, num_locations):
    match = re.search(r"<bin_(\d+)>", loc)
    token_num = float(match.group(1))
    xc, yc = token_num - math.floor((token_num - 1e-3) / num_locations) * num_locations, math.ceil(
        token_num / num_locations)
    xc_coord, yc_coord = (xc - 0.5) * 1 / num_locations, (yc - 0.5) * 1 / num_locations
    return xc_coord, yc_coord


def box_loc_to_offsets(box, loc, num_locations):
    xc_coord, yc_coord = loc_to_coords(loc, num_locations)
    x1_off, y1_off, x2_off, y2_off = xc_coord - box[0], yc_coord - box[1], box[2] - xc_coord, box[3] - yc_coord
    return [x1_off, y1_off, x2_off, y2_off]


def loc_offsets_to_bbox(loc, offsets, num_locations):
    xc_coord, yc_coord = loc_to_coords(loc, num_locations)
    x1, y1, x2, y2 = xc_coord - offsets[0], yc_coord - offsets[1], xc_coord + offsets[2], yc_coord + offsets[3]
    return [x1, y1, x2, y2]


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
    x1_off, y1_off, x2_off, y2_off = x1 - xc_coord, y1 - yc_coord, x2 - xc_coord, y2 - yc_coord
    offsets = [x1_off, y1_off,
               x2_off, y2_off]
    return torch.stack(offsets, dim=-1)


def coord_offsets_to_bbox(coord, offsets):
    xc_coord, yc_coord = coord.unbind(-1)
    x1_off, y1_off, x2_off, y2_off = offsets.unbind(-1)
    boxes = [xc_coord + x1_off, yc_coord + y1_off,
             xc_coord + x2_off, yc_coord + y2_off]
    return torch.stack(boxes, dim=-1)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


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


class MaskHead(nn.Module):
    def __init__(
            self,
            hidden_dim=256,
            activation=nn.GELU,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim // 4),
            activation(),
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.query_mlp = MLP(hidden_dim, hidden_dim, hidden_dim // 8, 3)

    def forward(
            self,
            query: torch.Tensor,
            image_embeddings: torch.Tensor,
    ):
        masks = self.predict_masks(
            image_embeddings,
            query,
        )
        # Prepare output
        return masks

    def predict_masks(self, src, query):
        bs, hw, c = src.shape
        h, w = int(math.sqrt(hw)), int(math.sqrt(hw))
        src_cat = torch.cat([s.unsqueeze(0).repeat(len(q), 1, 1) for s, q in zip(src, query)])
        query_cat = torch.cat(query)
        src_reshaped = src_cat.view(len(src), h, w, -1).permute(0, 3, 1, 2)
        upscaled_embedding = self.output_upscaling(src_reshaped)
        query_in = self.query_mlp(query_cat)
        bs, c, h, w = upscaled_embedding.shape
        masks = torch.einsum("bc,bcd->bd", query_in, upscaled_embedding.view(bs, c, h * w)).view(bs, -1, h, w)
        return masks


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
                        cur_new_input_embeds[cur_input_ids==self.mask_token_ids] = loc_embedding
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
                        cur_new_input_embeds[cur_input_ids==self.mask_token_ids] = loc_embedding
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
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # define bbox head and mask head
        # Initialize weights and apply final processing
        self.post_init()
        self.init_bbox_head(config)
        # self.init_autoencoder(None)

    def record_loc_token_id(self, tokenizer):
        self.tokenizer = tokenizer
        self.mask_token = "<mask>"
        self.box_token = "<box>"
        self.mask_token_ids = tokenizer.encode(self.mask_token, add_special_tokens=False)[0]
        self.box_token_ids = tokenizer.encode(self.box_token, add_special_tokens=False)[0]
        self.obj_visual_start_id = tokenizer.encode(OBJ_VISUAL_START, add_special_tokens=False)[0]
        self.obj_visual_end_id = tokenizer.encode(OBJ_VISUAL_END, add_special_tokens=False)[0]
        self.model.mask_token_ids = self.mask_token_ids

    def set_image_dir(self, output_dir):
        self.img_dir = os.path.join(output_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)

    def init_autoencoder(self, config):
        # TODO: write config, pretrained path, etc.
        print( f"pretrained {config.pretrained_autoencoder}")
        print( f"freeze autoencoder {config.freeze_autoencoder}")
        pretrained = config.pretrained_autoencoder

        # self.autoencoder = TransformerMask(d_model=512,
        #                                    max_len=200,
        #                                    ffn_hidden=512,
        #                                    n_head=8,
        #                                    n_layers=6,
        #                                    add_mapping=False,
        #                                    num_bins=100,
        #                                    mode="cls",
        #                                    drop_prob=0.1,
        #                                    share_loc_embed=True,
        #                                    device="cuda")
        self.autoencoder = ResNet50()
        if pretrained is not None:
            self.autoencoder.load_state_dict(torch.load(pretrained))
        if config.freeze_autoencoder:
            self.autoencoder.eval()
            for p in self.autoencoder.parameters():
                p.requires_grad = False
        self.decoder_mapping = nn.Linear(4096, 4096)
        self.encoder_mapping = nn.Linear(4096, 4096)
        # self.encoding_mapping = MLP(4096, 1024, 4096, 2)

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
        masks_seq_batch = masks_seq
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.training:
            masks_seq = torch.cat([torch.cat([torch.cat([masks for masks in masks_seq]) for masks_seq in masks_seqs]) for masks_seqs in masks_seq_batch])
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
            # input: <xxx><start><mask><end>;  target: <start><mask><end><xxx>
            shift_labels[shift_labels == self.mask_token_ids] = -100
            # shift_labels[shift_labels == self.obj_visual_end_id] = -100
            # Enable model/pipeleine parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss_lm = loss_fct(shift_logits, shift_labels)
            # loss += loss_lm
            loss += loss_lm * self.lm_loss_weight
        if self.training:
            pos_of_loc_tokens = torch.where(input_ids==self.obj_visual_start_id)
            # loc_embeddings = self.encoding_mapping(hidden_states[pos_of_loc_tokens[0], pos_of_loc_tokens[1]-1])
            loc_embeddings = self.decoder_mapping(hidden_states[pos_of_loc_tokens[0], pos_of_loc_tokens[1]])
            box_pred = self.bbox_head(loc_embeddings)
            boxes_seq = torch.tensor(boxes_seq).to(loc_embeddings.device).reshape(-1, 4)
            box_loss = self.box_loss(box_pred, boxes_seq) * self.box_loss_weight
            loss += box_loss

            # mask_decoded = self.autoencoder(masks_seq.unsqueeze(dim=1))
            mask_decoded = self.autoencoder.generate(loc_embeddings, ifsigmoid=False)
            # mask_loss = self.mask_loss(loc_embeddings,pretrained_embedding)
            recon_loss, l2_loss = self.mask_loss(mask_decoded, masks_seq, loc_embeddings, pretrained_loc_embedding)
            mask_loss = recon_loss * self.recon_loss_weight + l2_loss * self.l2_loss_weight
            # mask_loss = recon_loss + l1_loss
            # mask_loss = l1_loss
            loss += mask_loss
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    learnt_generation = self.autoencoder.generate(encoder_repr=loc_embeddings)
                    pretrained_generation = self.autoencoder.generate(encoder_repr=pretrained_loc_embedding)
                    # pretrained_embedding_noise = pretrained_embedding + torch.rand_like(pretrained_embedding) * torch.sqrt(mask_loss)
                    # pretrained_embedding_noise = pretrained_embedding + torch.rand_like(pretrained_embedding) * mask_loss
                    # noise_loss = self.mask_loss(pretrained_embedding_noise, pretrained_embedding)
                    # pretrained_generation_addednoise = self.autoencoder.generate(encoder_repr=pretrained_embedding_noise)
                    if hasattr(self, "imgid"):
                        self.imgid+=1
                    else:
                        self.imgid=0
                    plot_images(masks_seq[0].to(torch.float32).squeeze().detach().cpu().numpy(), learnt_generation[0].to(torch.float32).squeeze().detach().cpu().numpy(), pretrained_generation[0].to(torch.float32).squeeze().detach().cpu().numpy(), boxes_seq[0].to(torch.float32).squeeze().detach().cpu().numpy(), box_pred[0].to(torch.float32).squeeze().detach().cpu().numpy(), save_path=self.img_dir, imgid = self.imgid)
                        # print(f"pred: {learnt_generation[0][0].flatten()} len {len(learnt_generation[0][0].flatten())} \n gt: {pretrained_generation[0][0].flatten()}  len {len(pretrained_generation[0][0].flatten())}\n gt with noise: {pretrained_generation_addednoise[0][0].flatten()} len {len(pretrained_generation_addednoise[0][0].flatten())} \n noise loss {noise_loss.item()}")

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
