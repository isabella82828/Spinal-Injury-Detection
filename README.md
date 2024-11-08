# 🧠 Spinal Injury Detection and Segmentation in Ultrasound Images

This project automates diagnostics in spinal cord injuries by leveraging computer vision and machine learning. Through integrated computer vision algorithms, this project enables the detection of relevant features in ultrasound images, supporting continuous evaluation of clinical metrics and aiding radiologists in patient monitoring and diagnostics.

## Project Overview

Traditional approaches to spinal cord injury diagnostics rely on infrequent imaging sessions, which limit real-time insights into a patient’s recovery and condition. By using ultrasound imaging combined with wearable technology, this project enables continuous and automated tracking of spinal cord health, capturing key metrics such as swelling, inflammation, and injury development.

The dataset, comprising over 10,000 spinal cord images (both injured and non-injured), was developed to train and evaluate machine learning models specifically for injury localization and automatic segmentation. With tools like fine-tuned TransUNet, clinicians can visualize soft tissue segmentation and spinal anatomy changes over time, making it possible to observe developments in real-time and reduce the workload for radiologists.

After training, the models can be used for: 

- 📍 Injury Localization: Detect specific injuries within the spinal cord.
- 🧬 Soft Tissue Segmentation: Analyze anatomical features and track injury progression.

## Key Features

- 🔍 **Object Detection**: Localizes injuries (e.g., hematomas) in ultrasound images, aiding in diagnosis and clinical assessment.
- 🧩 **Semantic Segmentation**: Enables soft tissue segmentation within spinal cord ultrasound images to track anatomical changes.
- 📈 **Continuous Monitoring**: Facilitates ongoing assessment of patient health trajectory post-surgery, providing clinicians with actionable insights.
  
##Repository Structure

- **`DataScaling`**: Contains scripts to preprocess and scale ultrasound B-Mode Sagittal images for consistent input across models.
  
- **`HematomaDetection`**: Contains code for training object detection models to localize hematomas in spinal cord images, enabling automated injury detection.

- **`Segmentation`**: Contains code for training semantic segmentation models focused on segmenting soft tissue within spinal cord ultrasound images.

## Dataset

The project relies on a custom dataset with over 10,000 spinal cord ultrasound images. The images are annotated for both injury localization and segmentation tasks, supporting the training and evaluation of machine learning models.

## Model Training

- 🔹 **Object Detection**: Models trained to detect hematomas and localize injury sites.
- 🔹 **Semantic Segmentation**: Models fine-tuned to segment spinal cord anatomy over multiple cardiac cycles, providing detailed insights into structural and pathological changes.

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/isabella82828/Spine-Injury-Detection.git
   cd Spine-Injury-Detection

2. **Set Up the Environment**
   ```bash
   pip install -r requirements.txt

3. **Data Preprocessing**:
- Run data scaling scripts to scale and prepare data

4. **Train Model**
- For object detection, navigate to the HematomaDetection directory and follow the instructions to train the hematoma localization models.
- For segmentation, use the scripts in the Segmentation directory to train models for soft tissue segmentation.

## Conclusions 
This project demonstrates the promising potential of deep learning models for automating diagnostics in spinal cord injury using ultrasound imaging. Key quantitative findings include:

- __Injury Localization__: The YOLOv8 model outperformed other object detection models, achieving a mean Average Precision (mAP50-95) score of 0.606 and an Average Recall (AR) of 0.644.
- __Semantic Segmentation__: The DeepLabv3 model had the highest segmentation accuracy on porcine spinal cord images, with a Mean Dice score of 0.587. For human spinal cord images, the SAMed model performed best, achieving a Mean Dice score of 0.445.
- __Implantability Scores__: Both YOLOv8 and DeepLabv3 showed the highest potential for deployment in wearable or implantable devices, making them suitable for continuous monitoring of patient health.

These findings suggest that deep learning can be effectively used to track and monitor post-surgical developments, such as swelling, inflammation, and injury progression, to optimize patient treatment and recovery. The integration of these models into clinical workflows can enhance real-time decision-making and facilitate personalized, proactive care.


## Future Improvements 

- __Increased Dataset Diversity__: Expand the dataset to include more human spinal cord images from diverse patient populations to improve model generalization and robustness.
Transfer Learning Enhancements: Implement domain adaptation techniques to better translate models trained on porcine data to human data, enhancing their performance in clinical applications.
- __Improved Image Augmentation__: Use augmentation techniques, including deformation, reverberation, and signal-to-noise adjustments tailored for ultrasound imaging to increase model resilience to noise and artifacts.
- __Integration with Clinical Tools__: Create a seamless interface that integrates the model’s outputs into clinical diagnostic tools or surgical guidance systems.
