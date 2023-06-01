#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File      : sam_torch_inference.py
@Time      : 2023/05/31 20:21:39
@Author    : Huang Bo
@Contact   : cenahwang0304@gmail.com
@Desc      : None
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from segment_anything import sam_model_registry, SamPredictor
from utils import show_mask, show_points, show_box


if __name__ == "__main__":
    parser = argparse.ArgumentParser("use sam torch model inference")
    parser.add_argument("--img_path", type=str, default="images/truck.jpg")
    parser.add_argument("--sam_checkpoint", type=str, default="weights/sam_vit_h_4b8939.pth")
    parser.add_argument("--model_type", type=str, default="vit_h")
    parser.add_argument("--gpu_id", type=int, default=0, help="use which gpu to inference")

    args = parser.parse_args()
    
    device = f"cuda:{args.gpu_id}"
    
    image = cv2.imread(args.img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    print("Use point prompt to segment ...")
    # Point prompt
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f"results/torch_mask{i}.png", bbox_inches='tight', pad_inches=0)
        print(f"generate: results/torch_mask{i}.png")
        # plt.show() 
       
    
    print("Use point and last segment mask prompt to segment ...")
    # use last inference mask as input
    input_point = np.array([[500, 375], [1125, 625]])
    input_label = np.array([1, 1])
    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )
    
    print("Use point and boxes prompt to segment ...")
    # use box method to segment
    input_box = np.array([425, 600, 700, 875])
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    
    
    input_box = np.array([425, 600, 700, 875])
    input_point = np.array([[575, 750]])
    input_label = np.array([0])
    
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )