DeepFake Detection with Vision Transformer (ViT)


The Fine Tuning Model is in here: 

https://drive.google.com/drive/folders/15BahTRQzD7EUeOBCj2xPFSDa4i3H249w?dmr=1&ec=wgc-drive-hero-goto


This project uses a Vision Transformer (ViT) model to detect DeepFake images by fine-tuning it on a custom dataset created from the FaceForensics DeepFake Detection dataset. Using OpenCV, frames are extracted from real and fake videos to create the dataset, which is then used to train a ViT model for AI-based image classification.


Project Overview：

DeepFake detection is essential in various fields to ensure the authenticity of visual content. This project leverages the Vision Transformer (ViT) model from Hugging Face’s Transformers library, fine-tuning it on a custom labeled dataset of real and fake frames extracted from videos in the FaceForensics DeepFake Detection dataset.

The project includes three main components:

1.Data Preprocessing: Extracting frames from real and fake videos for training.

2.Model Training: Fine-tuning a ViT model on the custom image dataset. 

3.Single Image Testing: Testing the trained model on individual images for classification.

Features：

1. Frame extraction from FaceForensics videos using OpenCV
2. Model fine-tuning using Vision Transformer
3. Single-image testing for detecting DeepFake images
4. GPU support for faster training and inference

Requirements：

Python 3.8 or higher

PyTorch

Hugging Face Transformers

OpenCV

Matplotlib

Torchvision
