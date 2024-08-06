# AIML-Face-Recognition-Project

This project implements a face recognition system using OpenCV with deep learning. It detects faces in images, visualizes facial landmarks, and compares facial features to determine identity similarity.

## Features

- **Face Detection:** Detects faces in reference and query images using the `face_detection_yunet` model.
- **Facial Landmark Visualization:** Highlights facial landmarks such as eyes, nose, and mouth.
- **Face Alignment and Feature Extraction:** Aligns faces and extracts features using the `face_recognition_sface` model.
- **Identity Comparison:** Compares features to determine if two faces are of the same identity using cosine and L2 distances.

## Technologies Used

- **OpenCV:** For face detection and recognition.
- **ONNX Models:** `face_detection_yunet` for face detection and `face_recognition_sface` for face recognition.

## Setup and Installation

- pip install opencv-python opencv-python-headless numpy
- python face_recognition_script.py --reference_image path/to/reference.jpg --query_image path/to/query.jpg

## Contributing

Contributions are welcome! If you have suggestions or find issues, please feel free to:

- **Submit a Pull Request:** Make your changes and submit a pull request for review.
- **Open an Issue:** Report bugs or request new features by opening an issue.

Thank you for contributing!
