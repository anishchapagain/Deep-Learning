# Transfer Learning

## 1. Introduction
Transfer Learning is a powerful concept in Deep Learning where a model developed for one task is reused as the starting point for another related task. Instead of training a neural network from scratch, we leverage a pre-trained model that has already learned useful feature representations from a large dataset.

For example, a model trained on ImageNet (which contains millions of labeled images) can be reused to solve a smaller image classification task, such as identifying types of leaves or detecting defects in machinery. This approach helps save time, reduces computation, and often improves performance, especially when the new dataset is small.

### Why Transfer Learning?
- **Faster Training:** The model already knows useful features, reducing the training time.
- **Less Data Required:** Works well even with limited data.
- **Improved Performance:** Leverages learned representations from large datasets.
- **Avoids Overfitting:** Since lower layers are pre-trained, the model generalizes better.

---

## 2. How Transfer Learning Works
Transfer Learning typically follows these steps:

### Step 1: Choose a Pre-trained Model
Select a model that has been pre-trained on a large dataset. Common examples include:
- **Image Models:** ResNet, VGG, Inception, MobileNet, EfficientNet
- **NLP Models:** BERT, RoBERTa, GPT, T5

These models are available in frameworks like PyTorch through `torchvision.models` or `transformers` for NLP.

### Step 2: Freeze Base Layers
The earlier layers of the model contain general feature detectors (e.g., edges, textures in images, or syntactic patterns in text). We typically **freeze** them to retain these learned representations.

```python
for param in model.features.parameters():
    param.requires_grad = False
```

### Step 3: Replace and Train the Classifier Layers
We replace the last layer(s) with new ones suitable for our custom dataset and task.

Example (for image classification):
```python
import torch.nn as nn

model.fc = nn.Linear(model.fc.in_features, num_classes)
```
Now, train only the classifier part using your dataset.

### Step 4: Fine-Tuning (Optional)
After initial training, you can **unfreeze some of the deeper layers** and continue training with a lower learning rate. This allows the model to adapt its learned features more precisely to your dataset.

```python
for param in model.layer4.parameters():
    param.requires_grad = True
```

Then train again with a small learning rate:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

---

## 3. When to Use Transfer Learning
You should consider transfer learning when:

1. **You have limited labeled data** for your specific task.
2. **Your task is similar** to one where large pre-trained models already exist (e.g., object detection, sentiment analysis).
3. **Training from scratch** would be too costly or time-consuming.
4. **You want to achieve strong performance quickly** for a baseline model.

---

## 4. Transfer Learning vs Fine-Tuning
| Aspect | Transfer Learning | Fine-Tuning |
|---------|-------------------|--------------|
| Definition | Reusing pre-trained model as feature extractor | Unfreezing some layers to retrain on new data |
| Layers Trained | Only new classifier layers | Some or all layers retrained |
| Learning Rate | Higher | Lower |
| Training Time | Faster | Slower |
| Best For | Small dataset or similar domain | When domains are somewhat different |

---

## 5. Example: Image Classification with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets

# Load pre-trained model
model = models.resnet18(pretrained=True)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)  # Example: 3 classes

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Train the model
for epoch in range(5):
    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

To fine-tune, unfreeze last few layers and retrain with a smaller learning rate.

---

## 6. Summary
- **Transfer Learning** reuses pre-trained models to save time and improve performance.
- **Fine-tuning** adapts these models more precisely to your data.
- Works well for **vision and NLP** tasks.
- PyTorch makes it easy with `torchvision.models` and `transformers` libraries.

**In essence:** Transfer Learning lets you stand on the shoulders of giants — leveraging years of training data and compute from pre-trained models to achieve results faster and better.

---

**Next Steps for Practice:**
1. Try loading a ResNet18 and train it on a small custom dataset (like flowers or fruits).
2. Experiment with freezing and unfreezing layers.
3. Compare results between feature extraction and fine-tuning.

Further, move to more complex tasks like object detection or NLP fine-tuning using `transformers`.

