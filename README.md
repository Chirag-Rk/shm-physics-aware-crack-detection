# Physics-Aware Crack Detection for Structural Health Monitoring

## Introduction
Cracks in concrete structures such as bridges, roads, and buildings are commonly inspected through visual surveys conducted by engineers. In recent years, computer vision models have been used to automate crack detection from images. While these models can identify crack-like patterns, they often produce false positives due to shadows, stains, surface textures, or markings on concrete.

This project was developed to reduce such false detections and make crack detection systems more reliable and easier to trust. Instead of relying only on a deep learning model, the system applies simple physics-inspired reasoning to validate whether a detected crack actually makes sense.

---

## Problem Statement
Most CNN-based crack detection systems focus mainly on classification accuracy. In real inspection scenarios, this creates several issues:

- Non-crack patterns are frequently detected as cracks  
- The model provides no explanation for its decisions  
- Engineers have difficulty trusting the output  

In Structural Health Monitoring (SHM), false positives are costly because they can trigger unnecessary inspections and maintenance work, reducing confidence in automated systems.

---

## Approach
The idea behind this project is straightforward: **a real crack should behave like a real crack**.

After the CNN predicts a crack, the system checks whether the detected region satisfies basic physical characteristics of real cracks:

- The crack should have a clear dominant orientation  
- It should be continuous and sufficiently long  
- Very large cracks are accepted even if irregular, since missing them could be risky  

This logic mimics how a human inspector would judge whether a crack is meaningful.

---

## What the System Produces
Instead of only outputting a binary label, the system provides:

- Final decision: **Confirmed** or **Rejected**  
- Clear reasoning explaining why the decision was made  
- A structural **risk level**: LOW / MONITOR / MODERATE / HIGH  

This makes the output suitable for inspection decision support rather than just classification.

---

## System Workflow
Input Image
↓
CNN detects crack-like patterns
↓
Physics-aware checks (orientation and length)
↓
Explainable decision logic
↓
Risk-based inspection output



---

## Key Features
- Crack detection using a CNN (ResNet18)
- Physics-aware false positive reduction
- Explainable decision reasoning
- Structural risk labeling
- Lightweight interactive demo interface

---

## Results
By applying physics-aware validation on top of the CNN, the system reduced false crack detections by approximately **52%** compared to CNN-only detection. More importantly, each decision is explainable, which improves trust in the system.

---

## Demo
A Streamlit-based demo interface allows users to:

- Upload concrete surface images  
- View CNN predictions with confidence  
- See physics-based validation results  
- Understand the final decision and assigned risk level  

The demo is intended to demonstrate the decision logic and is not meant for production deployment.

---

## Limitations
- The system assumes input images are concrete surface images  
- Highly irregular cracks may still require manual inspection  
- Sensor data such as strain or vibration is not included in this version  

These limitations are documented intentionally to keep the system realistic and reliable.

---

## Future Scope
- Crack growth monitoring over time  
- Integration with sensor data (strain, vibration)  
- Crack segmentation for finer geometric analysis  
- Filtering out non-concrete images  
- Edge deployment for real-time inspections  

---

## Tools and Technologies
- Python  
- PyTorch  
- OpenCV  
- Streamlit  
- NumPy  

---

## About the Project
This project focuses on reliability and decision-making rather than only model accuracy. The goal is to support inspectors by reducing false alarms and providing clear, understandable results instead of replacing human judgment.

---

## How to Explain This Project in an Interview
> “Instead of just detecting cracks, my system focuses on deciding whether a detected crack should be trusted. It combines CNN predictions with simple physics-based reasoning and explains each decision with a structural risk level, making it useful for inspection support.”

---

## Author
**Chirag RK**  
Interests: Structural Health Monitoring, Computer Vision, Explainable AI


