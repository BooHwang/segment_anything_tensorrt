# Segment anything tensorrt

Use tensorrt accerate segment anything model ([SAM](https://github.com/facebookresearch/segment-anything)), which design by facebook research. In this repo, we divide SAM into two parts for model transformation, one is `ImageEncoderViT` (also named img embedding in this repo), and other one is `MaskDecoder`, `PromptEncoder` (also named sam model in this repo). while image encoder just inference once, and the most process time waste in image embedding, so you can save image embedding, and input different point or boxes to segment as your wish.



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

# To avoid fixing the original size when exporting the model, it is necessary to modify some of the code
# change "forward" function in the file which is "segment_anything/utils/onnx.py",as follows:
def forward(
    self,
    image_embeddings: torch.Tensor,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    mask_input: torch.Tensor,
    has_mask_input: torch.Tensor
    # orig_im_size: torch.Tensor,
):
    sparse_embedding = self._embed_points(point_coords, point_labels)
    dense_embedding = self._embed_masks(mask_input, has_mask_input)

    masks, scores = self.model.mask_decoder.predict_masks(
        image_embeddings=image_embeddings,
        image_pe=self.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embedding,
        dense_prompt_embeddings=dense_embedding,
    )

    if self.use_stability_score:
        scores = calculate_stability_score(
            masks, self.model.mask_threshold, self.stability_score_offset
        )

    if self.return_single_mask:
        masks, scores = self.select_masks(masks, scores, point_coords.shape[1])

    return masks, scores
    # upscaled_masks = self.mask_postprocessing(masks, orig_im_size)

    # if self.return_extra_metrics:
    #     stability_scores = calculate_stability_score(
    #         upscaled_masks, self.model.mask_threshold, self.stability_score_offset
    #     )
    #     areas = (upscaled_masks > self.model.mask_threshold).sum(-1).sum(-1)
    #     return upscaled_masks, scores, stability_scores, areas, masks

    # return upscaled_masks, scores, masks

# Download weights of sam_vit_h_4b8939.pth
python scripts/onnx2trt.py --prompt_masks_pt2onnx
```



- Export onnx model to engine file

```shell
# Download TensorRT packet from Nvidia, and unzip to /root, here we use version: TensorRT-8.6.1.6
pip install ~/TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp38-none-linux_x86_64.whl
export PATH=$HOME/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin:$PATH
export TENSORRT_DIR=$HOME/TensorRT-8.6.1.6:$TENSORRT_DIR
export LD_LIBRARY_PATH=$HOME/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH

# transform prompt encoder and mask decoder onnx model to tensorrt engine
trtexec --onnx=weights/sam_default_prompt_mask.onnx --workspace=4096 --shapes=image_embeddings:1x256x64x64,point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1 --minShapes=image_embeddings:1x256x64x64,point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1 --optShapes=image_embeddings:1x256x64x64,point_coords:1x10x2,point_labels:1x10,mask_input:1x1x256x256,has_mask_input:1 --maxShapes=image_embeddings:1x256x64x64,point_coords:1x20x2,point_labels:1x20,mask_input:1x1x256x256,has_mask_input:1 --saveEngine=weights/sam_default_prompt_mask.engine
```



- Or export onnx model to tensorrt engine file by code

```shell
python scripts/onnx2trt.py --sam_onnx2trt --sam_onnx_path ./weights/sam_vit_h_4b8939.onnx
```



## Inference

- Use **Pytorch** model inference

```shell
python sam_torch_inference.py
```



- Use **ONNX** model inference

```shell
python sam_onnx_inference.py
```



- Use **TensorRT** inference

```shell
python sam_trt_inference.py
```

