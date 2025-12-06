# Project Ideas You Can Complete with Limited Compute

### Neural Networks & Deep Learning -- Lecture Document

#### Project Ideas You Can Complete with Limited Compute

#### Guidelines
```
Remember your approach should be scientific and your code should be clean and readable
with comments and proper documentation,plus you should be able to justify your choices in each step
visualize your data and results, data should be clean and preprocessed.
```

## **Introduction**

Deep learning projects do not always require high-end GPUs or massive
datasets.
With **smart dataset collection**, **transfer learning**, and
**lightweight architectures**, you can complete impactful projects even
with limited computing resources.

Looking to complete a project with limited compute? This document provides:

-   **Feasible project ideas** (Single & Multimodal)
-   **Detailed guidelines** for dataset creation, model design,
    evaluation, and saving
-   Coursework alignment (Proposal → Model → Report → Presentation)
-   **Lightweight models** for low compute
-   **Small datasets** for quick training

## PART 1 --- Multimodal Project Ideas (10 Projects)

*(Use 2 or more modalities: image + text, audio + text, video + text)*

## 1. **Image Caption Generator**

-   **Data:** 40--60 photos + short captions
-   **Model:** MobileNetV2 (encoder) + GRU/LSTM (decoder)
-   **Output:** .pth caption model
-   **Why feasible:** Only decoder trains.

## 2. **OCR + Sentiment Analysis**

-   **Data:** Photos of posters or handwritten notes
-   **Model:** OCR (pretrained) + LSTM/DistilBERT classifier
-   **Output:** Sentiment classifier saved model
-   **Why feasible:** Only text model trained.

## 3. **Multimodal Fake News Detector**

-   **Data:** Student-created fake/real posts (image + caption)
-   **Model:** MobileNet + TF-IDF / DistilBERT
-   **Output:** Fusion classifier
-   **Why feasible:** Small dataset.

## 4. **Product Recommendation from Image + Text**

-   **Data:** Photos + short descriptions
-   **Model:** CNN + text embedding fusion
-   **Output:** Recommendation model
-   **Why feasible:** Simple 2-branch fusion.

## 5. **Low-FPS Video to Text Caption**

-   **Data:** 3--5 sec videos, 1--2 fps extracted frames
-   **Model:** CNN encoder + GRU decoder
-   **Output:** Video captioning model
-   **Why feasible:** Very few frames.

## 6. **Audio Emotion + Text Explanation**

-   **Data:** Voice clips (happy/sad/neutral)
-   **Model:** MFCC + CNN
-   **Output:** Emotion classifier
-   **Why feasible:** Lightweight.

## 7. **Food Image → Ingredient List (Template-based Text)**

-   **Data:** 30--40 food items
-   **Model:** CNN classifier + template text generator
-   **Output:** Food classifier
-   **Why feasible:** Only image model trained.

## 8. **Environment Classifier (Image + Temperature Data)**

-   **Data:** Desk photos + temp/humidity
-   **Model:** CNN + MLP fusion
-   **Output:** Fusion model
-   **Why feasible:** Simple architecture.

## 9. **Meme Classifier (Image + Extracted Text)**

-   **Data:** Meme collection (40--50 items)
-   **Model:** CNN + text classifier
-   **Output:** Meme classifier
-   **Why feasible:** Lightweight fusion.

## 10. **Signboard Reader (Image → Text → Category)**

-   **Data:** Photos of signboards
-   **Model:** OCR → classifier
-   **Output:** Signboard classifier
-   **Why feasible:** Only classifier trained.


## PART 2 --- Single-Modality Project Ideas (10 Projects)

## 11. **Hand Gesture Classifier**

-   **Data:** Photos of 4--6 gestures
-   **Model:** MobileNetV2
-   **Output:** gesture_model.pth
-   **Why feasible:** Small dataset.

## 12. **Student Stress Detection (Text Only)**

-   **Data:** Self-written stress/non-stress statements
-   **Model:** DistilBERT or LSTM
-   **Output:** stress_classifier.pt
-   **Why feasible:** Very small dataset.

## 13. **Lightweight Face Recognition for Attendance**

-   **Data:** 5--10 images per person
-   **Model:** Pretrained face embedding + classifier
-   **Output:** face_classifier.pt
-   **Why feasible:** Only classifier trained.

## 14. **Custom Object Detection (YOLO-Nano)**

-   **Data:** 40--60 labeled images
-   **Model:** YOLOv5n / Nano
-   **Output:** yolo_model.pt
-   **Why feasible:** Nano models train fast.

## 15. **Weather Prediction Using Tabular Data**

-   **Data:** Daily weather logs (student-collected)
-   **Model:** MLP regressor
-   **Output:** weather_mlp.pt
-   **Why feasible:** Very lightweight.

## 16. **Music Genre Classifier**

-   **Data:** Small clips (local genres)
-   **Model:** MFCC + CNN
-   **Output:** genre_cnn.pt
-   **Why feasible:** Fast training.

## 17. **Human Pose Classification**

-   **Data:** Simple poses (sitting, standing, stretching)
-   **Model:** MobileNetV2
-   **Output:** pose_model.pt
-   **Why feasible:** 3-class classifier.

## 18. **Road Surface Quality Classifier**

-   **Data:** Photos of good/damaged roads
-   **Model:** CNN (transfer learning)
-   **Output:** road_quality.pt
-   **Why feasible:** Small binary task.

## 19. **Selfie-Based Emotion Detector**

-   **Data:** Selfies with different expressions
-   **Model:** MobileNetV2
-   **Output:** emotion_model.pt
-   **Why feasible:** Small dataset.

## 20. **Document Type Classifier**

-   **Data:** Photos of ID cards, bills, notebooks
-   **Model:** CNN classifier
-   **Output:** document_classifier.pt
-   **Why feasible:** Low-class count.


# Detailed Project Guidelines

## 1. **Dataset Creation**

Students must: 
- Collect fresh data (images, audio, text, video) 
- Ensure at least **30--60 samples per class** 
- Split into **train/validation/test** (70/15/15)  or as you think is a kind of stanradr (do practise standard splits unless you can justify your splits)
- Perform **EDA**: 
- shape
- class distribution
- sample visualization
- data quality issues


## 2. **Model Design**

-   Use **transfer learning** (MobileNetV2, EfficientNet-B0, DistilBERT)
-   Freeze most layers initially
-   Train **only final layers** for low compute
-   Use regularization:
    -   dropout
    -   data augmentation
    -   early stopping


## 3. **Training Process**

Essential components: 
- Learning rate 
- Optimizer (Adam recommended) 
- Number of epochs (5--10 for transfer learning) 
- Batch size (8--32 depending on dataset size)

Show: 
- Loss curves
- Accuracy curves
- Confusion matrix
- Sample predictions


## 4. **Saving the Model**

Example (PyTorch):

``` python
torch.save(model.state_dict(), "model.pth")
```

Example (TensorFlow):

``` python
model.save("model.h5")
```

## 5. **Inference Script**

Students must write a **separate inference file** that:

-   Loads the saved model
-   Accepts new input
-   Outputs predictions
-   Prints or displays results

## 6. **Frontend (Optional)**

Students may use: 
- Streamlit
- Gradio
- Flask

This is optional but earns more marks in "Application".
Note: Only if your application requires a frontend.


## 7. **Final Report Guidelines**

Must include: 
- Problem statement
- Literature review
- Architecture explanation
- Dataset description
- Model training + results
- Screenshots
- Limitations
- Future improvements


## 8. **Ethical Considerations**

Students should discuss: 
- Data privacy
- Bias
- Responsible use
- Limitations due to small datasets


# Conclusion

These projects are **practical, interesting, and lightweight**,
fitting perfectly:

-   Model training
-   Model saving 
-   Proposal + report
-   Real-world relevance
-   Multimodal/single-modality requirements
-   Low compute constraints