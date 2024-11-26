<div align="center">
<h1>WS-SAM </h1>
<h3>WS-SAM: Self-Prompt SAM with Wavelet and Spatial Domain for OCTA Vessel Segmentation</h3>
</div>

## Abstract
Retinal vascular diseases often lead to vision impairment and blindness. Optical Coherence Tomography Angiography (OCTA) technology can accurately demonstrate the vascular structure of the eye  and assist in the diagnosis of retinal diseases. Accurate segmentation of retinal blood vessels and  ensuring connectivity are key challenges in retinal vessel segmentation. However, the OCTA images  have artifacts and noise, which makes vessels segmentation difficult. Recently, the Segment Anything Model (SAM) has impressed with its superior segmentation performance. To efficiently segment retinal vessels in OCTA images, we propose a self-prompt SAM based on Wavelets and spatial domains,  named WS-SAM. WS-SAM consists of five components: Wavelet Encoder, Pixel wise Image Encoder, Wavelet Space Fusion Moduel (WSFFormer), Meta Self-Prompter (MSPrompter) and Simplified Mask Decoder. To enrich the retinal data, we construct a new OCTA retinal vessel segmentation dataset,  named OCTA-RV. Compared with the existing OCTA retinal datasets, OCTA-RV contains multiple retinal vascular diseases and pays more attention on pixel-level label. The proposed WS-SAM  is conducted on the OCTA500 dataset and the new OCTA-RV dataset. The Dice Coefficient of the  proposed WS-SAM on OCTA500-3M, OCTA500-6M and OCTA-RV datasets are 0.7892, 0.8949 and 0.7412, respectively. Extensive experimental results demonstrate that WS-SAM performs competitively with state-of-the-art methods in OCTA retinal vessel segmentation.

## Overview
* **WS-SAM**
  
* **Pixel-wise Image Encoder(PWA-Former)**

<p align="center">
  <img src="assets/all_heatmap.png" alt="all_heatmap" width="40%">
</p>

