# Speech Emotion Recognition with Diffusion Models

This project explores deep learning-based emotion recognition from speech spectrograms. We extract features using Mel-frequency cepstral coefficients (MFCCs), encode them using a ResNet-50 convolutional neural network, and decode them with a conditional diffusion model.

##  Project Overview

- **Feature Extraction**: MFCCs are extracted from audio data to represent emotional speech in the frequency domain.
- **Encoder**: A ResNet-50 CNN is used to produce latent representations from the spectrograms.
- **Decoder**: A conditional U-Net diffusion model generates spectrograms from emotional embeddings.
- **Loss**: L2 loss (mean squared error) is used for training the diffusion model.
- **Training**: Gradient accumulation and mixed-precision training with AMP are used to optimize GPU memory usage.

## 📊 Dataset

The dataset includes labeled speech samples across several emotion categories. (Note: the dataset itself is not included due to licensing restrictions.)

## 🔧 Setup

Install dependencies:

```bash
pip install -r requirements.txt
