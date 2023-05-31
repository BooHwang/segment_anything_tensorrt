# Segment anything tensorrt

Use tensorrt accerate segment anything model ([SAM](https://github.com/facebookresearch/segment-anything)), which design by facebook research.



## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

```shell
# Install Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install requirement, maybe more than real use
pip install -r requirements.txt

# clone our repo 
git clone https://github.com/BooHwang/segment_anything_tensorrt.git
```

After clone the code, you should download sam model from ([SAM](https://github.com/facebookresearch/segment-anything)), and put it in `weights`.



## Model Transform

### Image embedding transform

- Transform image embedding pth from sam to onnx model

```shell
python scripts/onnx2trt.py --img_pt2onnx --sam_checkpoint weights/sam_vit_h_4b8939.pth --model_type default
```



- Transform image embedding onnx model to tensorrt engine

```shell
trtexec --onnx=embedding_onnx/sam_default_embedding.onnx --workspace=4096 --saveEngine=weights/sam_default_embedding.engine
```



- Or use code transform image embedding onnx model to tensorrt engine

```shell
python scripts/onnx2trt.py --img_onnx2trt --img_onnx_model_path embedding_onnx/sam_default_embedding.onnx 
```



### SAM model transform

**Notice:** opset set difference will get error while transfer onnx model to tensorrt engine, and it can set to 16 or 17 while my docker images is "nvidia/cuda:11.4.2-cudnn8-devel-ubuntu18.04"



- Transform sam pth model to onnx model

```shell
git clone https://github.com/facebookresearch/segment-anything.git

cd segment-anything

# Download weights of sam_vit_h_4b8939.pth, and change line 136 to "orig_im_size": torch.tensor([1500, 2250], dtype=torch.int32)
python scripts/export_onnx_model.py --checkpoint weights/sam_vit_h_4b8939.pth --output weights/sam_vit_h_4b8939.onnx --model-type default --opset 17

cp weights/sam_vit_h_4b8939.onnx <path_root>/segment_anything_tensorrt/weights
```



- Export onnx model to engine file

```shell
# Download TensorRT packet from Nvidia, and unzip to /root, here we use version: TensorRT-8.6.1.6
# pip install ~/TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp38-none-linux_x86_64.whl
export PATH=$HOME/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin:$PATH
export TENSORRT_DIR=$HOME/TensorRT-8.6.1.6:$TENSORRT_DIR
export LD_LIBRARY_PATH=$HOME/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH

trtexec --onnx=weights/sam_vit_h_4b8939.onnx --workspace=4096 --shapes=image_embeddings:1x256x64x64,point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1,orig_im_size:1200x1800 --minShapes=image_embeddings:1x256x64x64,point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1,orig_im_size:1200x1800 --optShapes=image_embeddings:1x256x64x64,point_coords:1x10x2,point_labels:1x10,mask_input:1x1x256x256,has_mask_input:1,orig_im_size:1200x1800 --maxShapes=image_embeddings:1x256x64x64,point_coords:1x20x2,point_labels:1x20,mask_input:1x1x256x256,has_mask_input:1,orig_im_size:1200x1800 --saveEngine=weights/sam_vit_h_4b8939.engine
```



- Or export onnx model to tensorrt engine file by code

```shell
python scripts/onnx2trt.py --sam_onnx2trt --sam_onnx_path ./weights/sam_vit_h_4b8939.onnx
```



## Inference

- Use **ONNX** model inference

```shell
python sam_onnx_inference.py
```



- Use **TensorRT** inference

```shell
python sam_trt_inference.py
```

