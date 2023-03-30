# Video Object Detection and Tracking using the Attention Mechanism: A Comparative Study
The objective of this study is to explore the potential of advanced deep learning techniques for video object detection and tracking. Our focus is on DETR and its variants, including Deformable DETR, Efficient DETR, TransVOD, and TrackFormer. 

To accomplish this goal, we will evaluate the capabilities of these models using well-known benchmark datasets such as MOT20 and YouTube-Objects. By doing so, we aim to uncover the strengths and limitations of these models and provide valuable insights for future research in this field.

## Table of Contents
- [Video Object Detection and Tracking using the Attention Mechanism: A Comparative Study](#video-object-detection-and-tracking-using-the-attention-mechanism-a-comparative-study)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Project Layout](#project-layout)
  - [Installation](#installation)
  - [Usage](#usage)

## Introduction
The main goal of this project is to perform a comparative study of several recent extensions of DETR for video object detection and tracking. We will apply these models to video clips by breaking them down into frames and running the models on each frame.

## Project Layout
The following is the folder structure of the project, which provides an overview of the organization and purpose of each directory.
```
DETR-COMPARATIVE-STUDY
│
├── data                      # Dataset storage and processing
│   ├── MOT20                 # Contains [MOT20](https://motchallenge.net/data/MOT20/) dataset
│   │   └── MOT20
│   │       └── test
│   │       └── train
│   ├── YouTube-Objects       # Contains [YouTube-Objects "Car"](https://data.vision.ee.ethz.ch/cvl/youtube-objects/) dataset
│   │   └── car
│   └── processed_data        # Processed data for evaluation
│
├── testing_models                    # Pre-trained models for evaluation
│   ├── detr.py                 
│   ├── deformable_detr       
│   ├── efficient_detr        
│   ├── transvod              
│   └── trackformer           
│
├── notebooks                 # Jupyter notebooks for analysis and visualization
│
├── utils.py                  # Utility functions and classes
├── main.py                   # Main script for evaluation
├── setup.py                  # Main script for setting up data
│
├── results                   # Output folder for results and performance metrics
│            
├── .gitignore                  
├── README.md           
└── requirements.txt    
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/DETR-COMPARATIVE-STUDY.git
cd DETR-COMPARATIVE-STUDY
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Run the setup script to download datasets
```bash
python setup.py
```

## Usage
(To be updated with instructions on how to use the project)
