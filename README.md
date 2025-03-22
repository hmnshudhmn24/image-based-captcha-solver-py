# Image-Based CAPTCHA Solver

## Overview
This project trains a convolutional neural network (CNN) to solve image-based CAPTCHAs. The model learns to recognize alphanumeric characters from CAPTCHA images.

## Features
- Uses a CNN to classify CAPTCHA characters
- Supports alphanumeric CAPTCHAs (A-Z, 0-9)
- Trained on a dataset of labeled CAPTCHA images
- Can predict and solve new CAPTCHA images

## Installation
Install required dependencies:
```sh
pip install tensorflow numpy opencv-python scikit-learn
```

## Training the Model
Place CAPTCHA images in a folder (e.g., `captcha_dataset/`) and run:
```sh
python captcha_solver.py
```

## Testing the Model
To predict a CAPTCHA:
```sh
python captcha_solver.py
```
Modify `test_captcha.png` with the path to your test image.
