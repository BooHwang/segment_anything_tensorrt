#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File      : sam_function.py
@Time      : 2023/06/01 13:50:45
@Author    : Huang Bo
@Contact   : cenahwang0304@gmail.com
@Desc      : None
'''

'''
all the function are from sam source code
'''

import torch
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize, to_pil_image
from torch.nn import functional as F

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def apply_coords(coords: np.ndarray, original_size: Tuple[int, ...], target_length: int = 1024) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(
        original_size[0], original_size[1], target_length
    )
    coords = coords.copy().astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords
    
def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)
        
def pre_processing(image: np.ndarray, 
                   img_size: int = 1024,
                   target_length: int = 1024, 
                   pixel_mean: List[float] = [123.675, 116.28, 103.53], 
                   pixel_std: List[float] = [58.395, 57.12, 57.375]):
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
    target_size = get_preprocess_shape(image.shape[0], image.shape[1], target_length)
    input_image = np.array(resize(to_pil_image(image), target_size))
    input_image_torch = torch.as_tensor(input_image)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    # Normalize colors
    input_image_torch = (input_image_torch - pixel_mean) / pixel_std

    # Pad
    h, w = input_image_torch.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
    return input_image_torch.numpy()

def resize_longest_image_size(
        input_image_size: torch.Tensor, longest_side: int
    ) -> torch.Tensor:
        input_image_size = input_image_size.to(torch.float32)
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
        return transformed_size
    
def mask_postprocessing(masks: np.array, orig_im_size: Tuple, img_size: int) -> torch.Tensor:
    masks = torch.from_numpy(masks[None, :, :, :]) # (4, 256, 256) -> (1, 4, 256, 256)
    orig_im_size = torch.tensor([orig_im_size[0], orig_im_size[1]], dtype=torch.int32)

    masks = F.interpolate(
        masks,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )

    prepadded_size = resize_longest_image_size(orig_im_size, img_size).to(torch.int64)
    masks = masks[..., : prepadded_size[0], : prepadded_size[1]]  # type: ignore

    orig_im_size = orig_im_size.to(torch.int64)
    h, w = orig_im_size[0], orig_im_size[1]
    masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
    return masks