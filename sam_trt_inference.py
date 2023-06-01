#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File      : sam_trt_inference.py
@Time      : 2023/05/31 11:13:08
@Author    : Huang Bo
@Contact   : cenahwang0304@gmail.com
@Desc      : None
'''

'''
Use tensorrt accerate segment anything model(SAM) inference
'''

import numpy as np
import cv2
from utils.common import TrtModel
import os
import argparse
import matplotlib.pyplot as plt
from utils import apply_coords, pre_processing, mask_postprocessing, show_mask, show_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser("use tensorrt to inference segment anything model")
    parser.add_argument("--img_path", type=str, default="images/truck.jpg", help="you want segment image")
    parser.add_argument("--sam_engine_file", type=str, default="weights/sam_default_prompt_mask.engine")
    parser.add_argument("--embedding_engine_file", type=str, default="embedding_onnx/sam_default_embedding.engine")
    parser.add_argument("--gpu_id", type=int, default=0, help="use which gpu to inference")
    parser.add_argument("--batch_size", type=int, default=1, help="use batch size img to inference")
    args = parser.parse_args()
    
    image = cv2.imread(args.img_path)
    orig_im_size = image.shape[:2]
    img_inputs = pre_processing(image)
    print(f'img input: {img_inputs.shape}')
    
    # embedding init
    embedding_inference = TrtModel(engine_path=args.embedding_engine_file, gpu_id=args.gpu_id, max_batch_size=args.batch_size)
    # sam init
    sam_inference = TrtModel(engine_path=args.sam_engine_file, gpu_id=args.gpu_id, max_batch_size=20)
    
    image_embedding = embedding_inference([img_inputs])[0].reshape(1, 256, 64, 64)
    print(f"img embedding: {image_embedding.shape}")
    
    # Point prompt
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_coord = apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    # print(image_embedding.shape)
    # print(onnx_coord.shape)
    # print(onnx_label.shape)
    # print(onnx_mask_input.shape)
    # print(onnx_has_mask_input.shape)

    input = [image_embedding, onnx_coord, onnx_label, onnx_mask_input, onnx_has_mask_input]
    shape_map = {'image_embeddings': image_embedding.shape,
                'point_coords': onnx_coord.shape,
                'point_labels': onnx_label.shape,
                'mask_input': onnx_mask_input.shape,
                'has_mask_input': onnx_has_mask_input.shape}
    
    output = sam_inference(input, binding_shape_map=shape_map)
    
    # print(output[0].shape)
    # print(output[1].shape)
    
    low_res_logits = output[0].reshape(args.batch_size, -1).reshape(4, 256, 256)
    scores =  output[1].reshape(args.batch_size, -1).squeeze(0)
    
    masks = mask_postprocessing(low_res_logits, orig_im_size, img_inputs.shape[2])
    masks = masks.numpy().squeeze(0)
    os.makedirs("results", exist_ok=True)
    # for i in range(masks.shape[0]):
    #     # mask_image = show_mask(masks[i]*255)
    #     cv2.imwrite(f"results/trt_mask{i}.png", masks[i]*255)
    #     print(f"Generate results/trt_mask{i}.png")
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask = mask > 0.0
        plt.figure(figsize=(10,10))
        plt.imshow(image[:, :, ::-1])
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f"results/trt_mask{i}.png", bbox_inches='tight', pad_inches=0)
        print(f"generate: results/trt_mask{i}.png")
        # plt.show() 
    