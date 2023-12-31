# Banking-Digital-Onboard-System
Project Test Video: https://drive.google.com/file/d/1s0fVl7ggrhvYrzGOhPGdRy8j8GtuGsIF/view?usp=sharing
For a visual demonstration of the app in action, please view the Project Test Video.


# Identity Verification App
# Overview

This Android application is designed to provide a secure and user-friendly platform for identity verification. The app utilizes advanced image processing and AI technologies to ensure that personal identification is authenticated efficiently.

# Features
Welcome Screen: A user-friendly entry point that greets users and introduces the app's functionalities.

ID Document Verification: Users are prompted to capture images of their identification documents. The app then uses image processing algorithms to verify the authenticity and validity of the documents.

Photo Capture for Validation: The app guides the user to take a photo of their ID in various positions (front, back) ensuring that all relevant information is captured for processing.

Facial Recognition for Biometric Validation: By accessing the device's camera, users can take a selfie which the app compares to the photo ID for biometric verification. This step employs facial recognition technology to ensure the user's identity matches the document provided.

Verification Process Indicators: The app includes clear indicators of the verification process, informing the user of the current step and what is required next.

Result Confirmation: After processing the ID document and biometric data, the app displays the outcome, confirming whether the verification was successful or not.

ID Information Extraction: The app extracts information from the ID as a string and displays it to customer service representatives, enabling them to review the ID information easily.

# Technology Stack
Backend: Python Flask
AI Tools: Google Cloud Vision API, OpenCV, Pyzbar, Ultralytics YOLO, Facenet PyTorch
Frontend/UI: Java (Android)
