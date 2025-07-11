# StressID Multimodal Distillation Model

## My works are in folder jihan
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

  * 0: Low level Stress
  * 1: Mid level Stress
  * 2: High level Stress

### Split Strategy

* Data is split based on participant's IDs
* Each participant's data belongs to one split only
* Distribution:

  * Train: 36 participants
  * Val: 8 participants
  * Test: 8 participants
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
## Future Work

To further enhance the performance and extend the understanding of modality-specific contributions, we plan to explore the following directions:

- **Model Architecture Enhancements**:  
  Introduce advanced backbone networks for both teacher and student models to capture more complex representations.

- **Data Augmentation**:  
  Apply augmentation techniques tailored for each modality, such as:
  - SpecAugment or frequency masking for log-mel spectrograms,
  - Geometric and color jittering for GAF images,
  - Temporal jittering for video frames.

- **Ablation Study on Knowledge Distillation**:  
  Conduct systematic ablation experiments to evaluate the effectiveness of distillation from each modality (video, ECG, EDA, RR) into the audio-based student model. This will help quantify the individual impact of each privileged signal.

These improvements aim to deepen our understanding of cross-modal knowledge transfer and build more robust stress recognition systems in low-resource (single modality) settings.


## Reference

This project is inspired by the privileged learning strategy proposed in:

**MARS: Motion-Augmented RGB Stream for Action Recognition**  
By T. Piergiovanni, A. Angelova (2019, CVPR)
[https://arxiv.org/abs/2007.12645](https://arxiv.org/abs/2007.12645)

In MARS, a single RGB stream learns from a more informative optical flow stream during training. Similarly, our student model (audio-only) learns from richer teacher features derived from video and physiological signals — modalities that are not available at test time. This concept of *distillation from privileged modalities* forms the basis of our training strategy.

## AI & Computer Vision lab at Konkuk University

- Advised by Prof. EunYi Kim
- Supported by Konkuk University