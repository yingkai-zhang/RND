# MEHSI
Multi-Exposure real HSI denoising (MEHSI) dataset for "*Real Noise Decoupling for Hyperspectral Image Denoising*" accepted at **AAAI 2026.**

## Details
We aim to capture a larger and more varied noise level dataset, so we control exposure time to 1/20, 1/50, and 1/100 of the reference image captured. We utilize an SOC710-VP hyperspectral camera to capture HSIs. 

The collected clean HSIs are averaged, and the paired data are manually aligned and calibrated. The collected data is processed to obtain 303 pairs of hyperspectral images with 34 spectral dimensions and a spatial resolution of $690\times512$.

We randomly select 273 pairs (91 scenes) for training and the remaining 30 (10 scenes) for testing.

## Train
We crop overlapped $128\times128$ spatial regions from the paired data for training.

## Inference
We crop center $512\times512$ spatial region from the paired data for test.

## Process
You can follow [SERT](https://github.com/MyuLi/SERT) to load HSIs.