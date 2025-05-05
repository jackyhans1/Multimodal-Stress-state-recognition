# StressID Multimodal Distillation Model

This repository contains the implementation of a multimodal learning pipeline using the [StressID dataset](https://project.inria.fr/stressid/), focused on recognizing human stress levels from various modalities: audio, video, ECG, EDA, and respiratory signals. The core objective is to train a lightweight student model using privileged information from multiple rich modalities via a knowledge distillation strategy.

## Research Objective

Stress detection is inherently multimodal. While audio-based models are practical for deployment, they often suffer in performance due to limited information. Our goal is to enhance unimodal (audio-only) performance by distilling knowledge from a richer multimodal teacher model.

## Key Ideas

- **Teacher Model**: A multimodal neural network trained using video, ECG, EDA, and RR signals to classify stress into 3 levels (Low, Medium, High).
- **Student Model**: A lightweight audio-only network trained using:
  - Supervision from ground-truth labels (cross-entropy).
  - Feature matching with the teacher's internal representations (MSE loss).
- **Privileged Learning**: During training, the student accesses the teacher’s features from richer modalities, which are not available during inference.

## Dataset: [StressID](https://project.inria.fr/stressid/)

* **Subjects**: 65 participants -> used data from 52 participants who has every moldality datas available
* **Modalities**:

  * Audio (.wav, log-mel converted PNG)
  * Video (.mp4)
  * ECG, EDA, RR (.png)
* **Label**: `affect3-class`

  * 0: Neutral
  * 1: Stress
  * 2: Amusement

### Split Strategy

* Data is split based on subject IDs
* Each subject's data belongs to one split only
* Distribution:

  * Train: 36 subjects
  * Val: 8 subjects
  * Test: 8 subjects
* Ensures near-original class ratio across splits

## Modalities & Preprocessing

- **Audio**: 
  - Raw `.wav` files are converted into **log-mel spectrogram** images before training.
  - These spectrograms serve as 2D input for CNN-based classification.

- **Physiological Signals** (ECG, EDA, RR):
  - Each signal is first segmented into 60-second windows.
  - Then, converted into **GAF (Gramian Angular Field)** images:
    - GAF encodes 1D time-series into 2D images by computing pairwise trigonometric relationships, preserving temporal correlation.
    - This transformation enables 1D physiological data to be used with 2D CNN architectures.

- **Video**:
  - 16 evenly spaced RGB frames are extracted per video and fed into a 3D CNN.

All modalities are padded within a batch to accommodate variable image sizes without distortion.

## Architecture

### Teacher Model
- **Video**: 3D CNN → 256-dim feature
- **Physiological**: Simple 2D CNNs (shared architecture) → 256-dim feature each
- **Classifier**: Fully connected layer on concatenated 1024-dim feature

### Student Model
- **Audio**: 2D CNN on log-mel spectrograms → 256-dim feature
- **Classifier**: Fully connected layer

### Distillation Loss
Total loss for the student:
```math
L = \text{CE}(y, \hat{y}) + \alpha \cdot \text{MSE}(f_s, f_v) + \beta \cdot \text{MSE}(f_s, f_{ecg}) + \gamma \cdot \text{MSE}(f_s, f_{eda}) + \delta \cdot \text{MSE}(f_s, f_{rr})
```

## Reference

This project is inspired by the privileged learning strategy proposed in:

**MARS: Motion-Augmented RGB Stream Distillation for Action Recognition**  
By T. Piergiovanni, A. Angelova  
[https://arxiv.org/abs/2007.12645](https://arxiv.org/abs/2007.12645)

In MARS, a single RGB stream learns from a more informative optical flow stream during training. Similarly, our student model (audio-only) learns from richer teacher features derived from video and physiological signals — modalities that are not available at test time. This concept of *distillation from privileged modalities* forms the basis of our training strategy.

## AI & Computer Vision lab at Konkuk University

### Advised by Prof. EunYi Kim
### Supported by Konkuk University