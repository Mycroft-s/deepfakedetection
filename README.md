The Fine Tuning Model is in here: 
https://drive.google.com/drive/folders/15BahTRQzD7EUeOBCj2xPFSDa4i3H249w?dmr=1&ec=wgc-drive-hero-goto

DeepFake Detection with Vision Transformer (ViT)
This project uses a Vision Transformer (ViT) model to detect DeepFake images by fine-tuning it on a custom dataset created from the FaceForensics DeepFake Detection dataset. Using OpenCV, frames are extracted from real and fake videos to create the dataset, which is then used to train a ViT model for AI-based image classification.

Table of Contents
Project Overview
Features
Installation
Usage
Data Preprocessing
Model Training
Single Image Testing
File Structure
Requirements
Contributing
License


Project Overview
DeepFake detection is essential in various fields to ensure the authenticity of visual content. This project leverages the Vision Transformer (ViT) model from Hugging Face’s Transformers library, fine-tuning it on a custom labeled dataset of real and fake frames extracted from videos in the FaceForensics DeepFake Detection dataset.

The project includes three main components:

Data Preprocessing: Extracting frames from real and fake videos for training.
Model Training: Fine-tuning a ViT model on the custom image dataset.
Single Image Testing: Testing the trained model on individual images for classification.
Features
Frame extraction from FaceForensics videos using OpenCV
Model fine-tuning using Vision Transformer
Single-image testing for detecting DeepFake images
GPU support for faster training and inference

Installation
To set up the environment for this project, follow these steps:

Clone the repository:


git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Set up a virtual environment (optional but recommended):


python -m venv .venv
source .venv/bin/activate    # On Windows use `.venv\Scripts\activate`
Install the dependencies:


pip install -r requirements.txt
CUDA Setup (if using GPU): Ensure CUDA and cuDNN are properly configured. Check the torch and transformers documentation if needed.
