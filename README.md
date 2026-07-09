# Deepfake Video Detection Architecture

## Overview
This repository contains a deep learning pipeline designed to detect forged/deepfake videos. The project processes raw video data, extracts facial frames using advanced computer vision techniques, and leverages a time-distributed neural network architecture to classify videos as REAL or FAKE.

## Key Features & Engineering
* **Efficient Memory Management:** Utilizes NumPy's memory-mapping (`mmap_mode`) to load large batches of processed video slices without overwhelming system RAM.
* **Smart Frame Extraction:** Uses `dlib`'s frontal face detector to locate faces, applies a 1.3x padding factor to capture the full head, and standardizes inputs to 224x224 pixels. Exactly 15 frames are extracted per video to capture temporal inconsistencies.
* **Advanced Deep Learning Architectures:** The pipeline is built to support Time-Distributed CNNs (Xception, MobileNet, ResNet50) paired with Bidirectional LSTMs to analyze both spatial artifacts and temporal anomalies across frames.
* **Automated Data Visualizations:** Includes automated plotting of Original vs. Deepfake frames for visual verification and debugging.

## Tech Stack
* **Languages:** Python 3
* **Libraries/Frameworks:** TensorFlow, Keras, OpenCV (`cv2`), dlib, Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn.