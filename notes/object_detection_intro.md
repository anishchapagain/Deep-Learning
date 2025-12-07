# Object Detection & Segmentation Overview

## Object Detection
- **Goal**: Locate and classify objects in an image with bounding boxes.
- **Typical pipeline**:
  1. Generate region proposals (or dense sliding windows).
  2. Extract features for each proposal.
  3. Classify proposals and refine box coordinates.
- **Two‑stage detectors** achieve higher accuracy by separating proposal generation from classification.

### Two‑Stage Detectors
- **Stage 1 – Region Proposal**: Proposes candidate object locations.
  - Classic: *Selective Search* (used in R‑CNN).
  - Modern: *Region Proposal Network (RPN)* (used in Faster R‑CNN).
- **Stage 2 – Classification & Regression**: Classifies each proposal and adjusts its box.
- **Advantages**: Better accuracy, flexible backbone, easier to incorporate additional tasks (e.g., mask prediction).

## R‑CNN (Region‑Based Convolutional Neural Network)
1. **Selective Search** generates ~2000 region proposals per image.
2. Each proposal is **warped** to a fixed size and fed through a **pre‑trained CNN** (e.g., AlexNet) to extract a 4096‑D feature vector.
3. **Support Vector Machines (SVMs)** classify each feature vector; a separate **linear regressor** refines the bounding box.

**Pros**: Leverages powerful CNN features; clear modular design.
**Cons**: Slow – each proposal requires a forward pass through the CNN; training is complex (multiple models).

![R‑CNN pipeline](https://upload.wikimedia.org/wikipedia/commons/5/5c/R-CNN_architecture.png)

## Fast R‑CNN
- **Key improvement**: Compute CNN features **once per image** using a *shared* convolutional backbone.
- **RoI Pooling** extracts a fixed‑size feature map for each proposal from the shared feature map.
- A single **softmax classifier** and **bbox regressor** are trained jointly.

**Benefits**: ~10× faster training/inference than R‑CNN, end‑to‑end learning.

![Fast R‑CNN architecture](https://upload.wikimedia.org/wikipedia/commons/2/2e/Fast_R-CNN_architecture.png)

## Object Segmentation
- **Semantic Segmentation**: Assign a class label to every pixel (no instance distinction).
- **Instance Segmentation**: Distinguish separate object instances; combines detection with pixel‑level masks.

### Mask R‑CNN (Extension of Faster R‑CNN)
- Adds a **parallel mask branch** to predict a binary mask for each RoI.
- Uses **RoIAlign** for precise pixel‑level alignment.

![Mask R‑CNN results](https://upload.wikimedia.org/wikipedia/commons/1/1c/Mask_R-CNN_example.png)

## Quick Reference Links
- R‑CNN paper: https://arxiv.org/abs/1311.2524
- Fast R‑CNN paper: https://arxiv.org/abs/1504.08083
- Faster R‑CNN paper: https://arxiv.org/abs/1506.01497
- Mask R‑CNN paper: https://arxiv.org/abs/1703.06870