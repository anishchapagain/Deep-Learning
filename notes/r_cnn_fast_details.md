# Deep Dive: R‑CNN and Fast R‑CNN

## 1. R‑CNN (Region‑Based Convolutional Neural Network)

### 1.1 Core Idea
- **Goal**: Detect objects by classifying a set of region proposals.
- **Pipeline**:
  1. **Selective Search** generates ~2000 candidate boxes per image.
  2. Each box is **warped** to a fixed size (e.g., 227×227) and fed through a **pre‑trained CNN** (AlexNet, VGG) to extract a high‑dimensional feature vector.
  3. Two separate models are trained on these features:
     - **SVM** for classification.
     - **Linear regression** for bounding‑box refinement.

### 1.2 Training Details
- **Stage‑wise training** – the CNN, SVM, and regressor are trained **independently**.
- **Fine‑tuning** – after the CNN is pre‑trained on ImageNet, it is fine‑tuned on the detection dataset using the region proposals.
- **Losses**:
  - Classification: hinge loss for SVM (or soft‑max cross‑entropy if you replace SVM with a NN classifier).
  - Bounding‑box regression: smooth L1 loss.

### 1.3 Pros & Cons
| Pros | Cons |
|------|------|
| Leverages powerful CNN features trained on large datasets. | Very **slow** – each proposal requires a full forward pass through the CNN (≈2000 passes per image). |
| Clear modular design – easy to replace components. | Complex training pipeline (multiple models, separate optimizers). |
| Good baseline for later two‑stage methods. | Memory intensive – storing features for all proposals.

### 1.4 Typical Numbers (VOC‑2007)
- **Mean Average Precision (mAP)**: ~53% (AlexNet backbone).
- **Inference time**: ~10 s per image on a single GPU.

### 1.5 Helpful Resources
- **Paper**: [R‑CNN (2014)](https://arxiv.org/abs/1311.2524)
- **Selective Search implementation**: https://github.com/AlpacaDB/selective-search
- **Code snippet (PyTorch‑style)**:

```python
# Pseudo‑code for extracting region features
import torchvision.models as models
import torchvision.transforms as T

cnn = models.alexnet(pretrained=True).features  # feature extractor
transform = T.Compose([T.Resize(227), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]])

for box in proposals:  # proposals from selective search
    region = image.crop(box)               # PIL crop
    region = transform(region).unsqueeze(0)  # (1, C, H, W)
    feat = cnn(region)                     # (1, C', H', W')
    feat = torch.flatten(feat, 1)          # (1, D)
    # store feat for later SVM/regressor training
```

---

## 2. Fast R‑CNN

### 2.1 Core Idea
- **Key innovation**: Compute **CNN convolutional features once per image**, then reuse them for all proposals.
- Introduces **RoI (Region of Interest) Pooling** to extract a fixed‑size feature map for each proposal from the shared feature map.
- Trains a **single network** end‑to‑end with two heads:
  1. **Soft‑max classifier** (instead of SVM).
  2. **Bounding‑box regressor**.

### 2.2 Architecture Overview
```
Image → Conv Backbone → Shared Feature Map
          │
          └─► RoI Pooling (one per proposal) → Fully‑connected layers →
                ├─► Soft‑max (K+1 classes)
                └─► BBox regression (4*K values)
```
- **RoI Pooling**: Divides each proposal into a fixed grid (e.g., 7×7) and applies max‑pooling within each cell.
- **Loss**: Multi‑task loss = classification loss (cross‑entropy) + bbox regression loss (smooth L1).

### 2.3 Training Details
- **Single‑stage optimisation** – the whole network (backbone + heads) is trained jointly.
- **Mini‑batch sampling**: Each mini‑batch contains a mix of foreground (IoU ≥ 0.5) and background (IoU < 0.5) RoIs.
- **Learning rate schedule**: Typical schedule – start at 0.001, decay by 10× after 40k/70k iterations.

### 2.4 Pros & Cons
| Pros | Cons |
|------|------|
| **~10× faster** than R‑CNN (single forward pass). | Still slower than single‑stage detectors (e.g., YOLO, SSD). |
| End‑to‑end training simplifies the pipeline. | RoI Pooling introduces a small quantisation error (addressed later by RoIAlign). |
| Flexible – can swap backbone (ResNet‑101, VGG‑16). | Requires a GPU with enough memory for the shared feature map.

### 2.5 Typical Numbers (VOC‑2007)
- **mAP**: ~66% (VGG‑16 backbone).
- **Inference time**: ~0.2 s per image on a modern GPU.

### 2.6 Helpful Resources
- **Paper**: [Fast R‑CNN (2015)](https://arxiv.org/abs/1504.08083)
- **PyTorch implementation** (torchvision): https://pytorch.org/vision/stable/models.html#fasterrcnn (Fast R‑CNN is the predecessor of Faster R‑CNN; the code can be adapted).
- **RoI Pooling code snippet**:

```python
import torch.nn as nn
import torch.nn.functional as F

class RoIPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, feature_map, rois):
        # rois: (N, 5) [batch_idx, x1, y1, x2, y2]
        return torch.ops.torchvision.roi_pool(feature_map, rois, self.output_size, self.spatial_scale)
```

---

## 3. Quick Comparison Table
| Aspect | R‑CNN | Fast R‑CNN |
|--------|-------|------------|
| Feature extraction | Per‑proposal (≈2000×) | Single image‑wide pass |
| Classifier | SVM (separate) | Soft‑max (joint) |
| Training complexity | Multi‑stage | End‑to‑end |
| Speed (GPU) | ~10 s / img | ~0.2 s / img |
| mAP (VOC‑2007) | ~53% | ~66% |

## 4. Where to Go Next?
- **Faster R‑CNN** adds a **Region Proposal Network (RPN)** to replace Selective Search, further boosting speed.
- **Mask R‑CNN** builds on Faster R‑CNN to add instance‑mask prediction.
- For real‑time needs, explore **single‑stage detectors** (YOLO, SSD, RetinaNet).

---

### References & Image Links (free to embed)
- R‑CNN diagram: https://upload.wikimedia.org/wikipedia/commons/5/5c/R-CNN_architecture.png
- Fast R‑CNN diagram: https://upload.wikimedia.org/wikipedia/commons/2/2e/Fast_R-CNN_architecture.png
- RoI Pooling illustration: https://upload.wikimedia.org/wikipedia/commons/8/86/RoI_Pooling.png