# Code Review: CNN Fine-Tuning (ResNet18 on CIFAR-10)

## 1. Overview
The notebook implements a transfer learning pipeline using a pre-trained ResNet-18 model on the CIFAR-10 dataset.

The overall structure is logical: Setup -> Data Loading -> Model Modification -> Training -> Inference. 

However, there is a fundamental mismatch between the model's expected input resolution (ImageNet-style) and the dataset's actual resolution (CIFAR-10), which will severely hamper performance. 


## 2. Critical Issues (Must Fix)

### 2.1. Input Resolution Mismatch
**Issue:**  
ResNet-18 is designed for **224x224** images. It aggressively downsamples the input by a factor of 32 (5 stages of stride 2).
- **Current Behavior:** We are feeding **32x32** CIFAR-10 images.
- **Result:** After the final convolutional layer, the feature map size will be $32 / 32 = 1 \times 1$. This eliminates all spatial information before the global average pooling layer, limiting the model's ability to learn complex patterns.

**Correction Strategies (Choose one):**
1.  **Upscaling (Easiest)**: Add `transforms.Resize((224, 224))` to `transforms.Compose`.
    *   *Pros:* No model architecture changes needed.
    *   *Cons:* Increases memory usage and training time significantly.
2.  **Modify Architecture (Efficient)**: Replace the first layer to handle small images.
    ```python
    # After loading model
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() # Remove the first maxpool
    ```
    *   *Pros:* training fast.
    *   *Cons:* We lose the pre-trained weights for the first layer (not a huge issue as it learns quickly).

### 2.2. Inference Normalization Missing
**Issue:**  
In the `predict_image_class` function (Inference section), the normalization transform is commented out:
```python
# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
```
**Correction:**  
We **MUST** use the exact same normalization statistics for inference that we used for training. If the model sees un-normalized pixel values (0-1) during inference but learned on normalized values (approx -2 to +2), predictions will be random/garbage.

---

## 3. Best Practices & Improvements

### 3.1. Data Augmentation
**Current:** Only `ToTensor` and `Normalize`.  
**Recommendation:** To prevent overfitting (CIFAR-10 is easy to overfit), add augmentation to the **training** transform:
```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4), # Or Resize
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])
```

### 3.2. Evaluation Function
**Current:** We use `test_loader` inside `training_epoch` or just after it.
**Recommendation:** Ensure we clearly separate "Validation" (checking progress) from "Testing" (final evaluation). 

### 3.3. Layer Replacement
**Current:**
```python
model.fc = nn.Sequential(...)
```
**Observation:** We are adding a hidden layer (Linear -> ReLU -> Linear).  