# Test-Time Generative Augmentation (TTGA)

This is the official repository for "Test-Time Generative Augmentation for Medical Image Segmentation"

## Introduction
Test-Time Generative Augmentation (TTGA) is a novel approach to enhance medical image segmentation during test time. Instead of employing handcrafted transforms or functions on the input test image to create multiple views for test-time augmentation, this approach advocate for the utilization of an advanced domain-fine-tuned generative model, e.g., diffusion models, for test-time augmentation. Hence, by integrating the generative model into test-time augmentation, we can effectively generate multiple views of a given test sample, aligning with the content and appearance characteristics of the sample and the related local data distribution.

<img src="figs/fig-1.png">

## Augmentation
- Optic Disc and Cup Segmentation
<p float="left">
  <img src=figs/fundus_org.png height=240 />
  <img src=figs/fundus_aug.gif height=240 /> 
</p>

- Polyp Segmentation
<p float="left">
  <img src=figs/polyp_org.png height=240 />
  <img src=figs/polyp_aug.gif height=240 /> 
</p>

- Skin Lesion Segmentation
<p float="left">
  <img src=figs/skin_org.png height=240 />
  <img src=figs/skin_aug.gif height=240 /> 
</p>

## Citing
TO-DO.
