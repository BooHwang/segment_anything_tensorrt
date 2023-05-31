#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File      : sam_onnx_inference.py
@Time      : 2023/05/31 11:16:23
@Author    : Huang Bo
@Contact   : cenahwang0304@gmail.com
@Desc      : None
'''


import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import onnxruntime
from torchvision.transforms.functional import resize, to_pil_image
from torch.nn import functional as F
from typing import Tuple, List


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("use sam onnx model inference")
    parser.add_argument("--img_path", type=str, default="images/truck.jpg", help="you want segment image")
    parser.add_argument("--img_onnx_model_path", type=str, default="embedding_onnx/sam_default_embedding.onnx")
    parser.add_argument("--sam_onnx_model_path", type=str, default="weights/sam_vit_h_4b8939.onnx", help="sam onnx model")
    parser.add_argument("--gpu_id", type=int, default=0, help="use which gpu to inference")
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu_id}"
    
    ort_embedding_session = onnxruntime.InferenceSession(args.img_onnx_model_path, provider=['CUDAExecutionProvider'])
    ort_sam_session = onnxruntime.InferenceSession(args.sam_onnx_model_path, provider=['CUDAExecutionProvider'])
    ort_embedding_session.set_providers(['CUDAExecutionProvider'],  provider_options=[{f'device_id': {args.gpu_id}}])
    ort_sam_session.set_providers(['CUDAExecutionProvider'],  provider_options=[{f'device_id': {args.gpu_id}}])
    
    image = cv2.imread(args.img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_inputs = pre_processing(image)
    ort_inputs = {"images": img_inputs}
    # Get image embedding, just extra once
    image_embeddings = ort_embedding_session.run(None, ort_inputs)[0]
    
    # Point prompt
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_coord = apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    
    ort_inputs = {
        "image_embeddings": image_embeddings,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.int32)
    }
    masks, scores, low_res_logits = ort_sam_session.run(None, ort_inputs)
    masks = masks > 0.0
    
    for i, (mask, score) in enumerate(zip(masks[0], scores[0])):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f"results/onnx_mask{i}.png", bbox_inches='tight', pad_inches=0)
        print(f"generate: results/onnx_mask{i}.png")
        # plt.show() 