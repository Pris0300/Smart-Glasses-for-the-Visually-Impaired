# 🎯 Object Detection for Assistive Smart Glasses
This repository contains the Machine Learning model developed as part of a larger group project aimed at assisting visually impaired individuals through real-time object detection using smart glasses.

**🧩 This was one of three core components of the complete system:**

**1. Mobile Application** – Connecting to Vuzix Blade smart glasses to access video.

**2. Machine Learning Model** – Responsible for real-time object classification and recognition *(This is the part I contributed to and is the focus of this repository)*.

**3. Cloud Infrastructure** – Designed to support Federated Learning and remote processing.

## 🧠 Project Overview
The goal of the full project was to design a smart assistive device using Vuzix Blade smart glasses that could help blind or visually impaired users detect obstacles and objects around them. The system was intended to stream video from the glasses to a mobile phone, where it would be processed and analyzed using a custom object recognition model. The cloud component was planned to support federated learning to improve model performance while maintaining data privacy.

Due to development constraints, full integration between components wasn’t completed, but each sub-team worked independently on their assigned modules.

## 🤖 Machine Learning Model (My Contribution)
This repository includes:

A custom **Convolutional Neural Network (CNN)** developed and trained to perform object recognition on video frames.

Comparison between different model architectures to identify the most accurate and efficient for deployment on mobile devices.

Preprocessing and organization of image datasets.

Iterative model improvements through multiple testing phases.

While this module was originally intended to integrate with the app and cloud, it can be run independently to test object classification performance.

## 📂 Contents of This Repository
model/: Scripts and Jupyter notebooks for training, evaluating, and testing the model.

report/: Final documentation and report detailing the development and testing phases of the machine learning component.

## 🧪 Tools & Technologies
- Python
- TensorFlow / Keras
- CNN-based image classification
- Computer Vision techniques
- Git / GitHub for version control

## IMPORTANT
*Due to the collaborative nature of this project, only code and components I directly worked on are included.*

---
