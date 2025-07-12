# Test-Time Generative Augmentation (TTGA)

This is the official repository for "[Test-Time Generative Augmentation for Medical Image Segmentation](https://arxiv.org/abs/2406.17608)"

## Introduction
Test-Time Generative Augmentation (TTGA) is a novel approach to enhance medical image segmentation during test time. Instead of employing handcrafted transforms or functions on the input test image to create multiple views for test-time augmentation, this approach advocate for the utilization of an advanced domain-fine-tuned generative model, e.g., diffusion models, for test-time augmentation. Hence, by integrating the generative model into test-time augmentation, we can effectively generate multiple views of a given test sample, aligning with the content and appearance characteristics of the sample and the related local data distribution.

<img src="figs/fig-1.png">

## Pipeline
The proposed pipeline of three key workflows are presented. The test image is processed through a sequence of steps to generate a noise image at a designated step count. Using this noise image, a one-step denoising process is employed to refine a trainable null-text embedding, enabling the stable generation of results that closely resemble the initial image. In the augmentation generation phase, this null-text embedding, guided by semantic and regional information, is leveraged to produce a series of augmented images.
<img src="figs/fig-2.png">

## Installation
This repository requires CUDA 12.1, Python 3.9 and Pytorch 2.1. To install the latest version run:
1. Create Conda Environment and Install Dependencies
  ```
  conda create --name ttga python=3.9
  conda activate ttga
  pip install -r requirements.txt
  ```
2. Install diffusers
- Download diffusers==0.27.0 and extract its contents. You can do this using command-line tools:
```
wget https://pypi.org/packages/source/d/diffusers/diffusers-0.27.0.tar.gz
tar -xzf diffusers-0.27.0.tar.gz
mv diffusers-0.27.0 diffusers
cd diffusers
python setup.py install
cd ..
```

## Running
This section outlines the steps to run the processes described in this repository, using the **polyp** dataset as an example.

**Step 1: LoRA Fine-tuning of Stable Diffusion**
First, fine-tune Stable Diffusion using `LoRA` on the polyp dataset:
```
bash scripts/lora_polyp.sh
```

**Step 2: One-step Null-Text Optimization**
Next, perform one-step null-text optimization on test images to extract their identity information.

**Step 3: Generative Augmentation**
Finally, apply generative augmentation to the test images. 

**Batch Operations and Demos:**
- For **batch operations** of **Step 2** and **Step 3**, please refer to [`main.py`](main.py). This script provides a comprehensive implementation for processing multiple images.
- A **demonstration** showcasing these processes can be found in [`demo.ipynb`](demo.ipynb). This notebook provides an interactive example of how to apply these techniques.

## Augmentation
:sparkles: Optic Disc and Cup Segmentation

<img src=figs/fundus_aug.webp height=150 />

:sparkles: Polyp Segmentation

<img src=figs/polyp_aug.webp height=150 />

:sparkles: Skin Lesion Segmentation

<img src=figs/skin_aug.webp height=150 />

## Materials
:two_hearts: SOTA segmentation models with codes, datasets and open-source parameters. (Thanks!)

| Index | Physiology         | Dataset  | Paper    | Code |
| :----:| :----:             | :----:   | :----:   |:----:|
| 1     | Optic Disc and Cup | [REFUGE20](https://refuge.grand-challenge.org/Download/) | [Segtrain](https://arxiv.org/pdf/2105.09511)|[code](https://github.com/askerlee/segtran/)|
| 2     | Polyp              | [Kvasir](https://datasets.simula.no/kvasir/)<br>[CVC-ClinicDB](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)<br>[CVC-ColonDB](http://mv.cvc.uab.es/projects/colon-qa/cvc-colondb)<br>[CVC-300](http://adas.cvc.uab.es/endoscene)<br>[ETIS-LaribPolypDB](https://www.kaggle.com/datasets/nguyenvoquocduong/etis-laribpolypdb) | [HSNet](https://www.sciencedirect.com/science/article/abs/pii/S0010482522008812?fr=RR-2&ref=pdf_download&rr=89939d365e6984b1)|[code](https://github.com/baiboat/HSNet/)|
| 3     | Skin Lesion        | [ISIC 2017](https://challenge.isic-archive.com/data/#2017)<br>[ISIC 2018](https://challenge.isic-archive.com/data/#2018)<br>[PH2](https://www.fc.up.pt/addi/ph2%20database.html) | [TMUnet](https://arxiv.org/pdf/2203.01932)|[code](https://github.com/rezazad68/TMUnet/)|

## Citing
TO-DO.
