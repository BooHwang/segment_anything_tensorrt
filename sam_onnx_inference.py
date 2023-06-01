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
import matplotlib.pyplot as plt
import onnxruntime
from utils import pre_processing, apply_coords, show_mask, show_points


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