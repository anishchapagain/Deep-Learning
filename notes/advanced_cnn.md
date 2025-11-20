# Advanced CNNs in Practice: PyTorch Examples

This document provides practical, hands-on examples of how to use state-of-the-art, pre-trained CNN models for advanced computer vision tasks using PyTorch, `torchvision`, and other popular image libraries like Pillow and OpenCV.

We will cover two major tasks:
1.  **Object Detection** with a pre-trained Faster R-CNN model.
2.  **Image Segmentation** with a pre-trained DeepLabV3 model.

These examples are designed to be runnable and demonstrate the end-to-end process from loading a model to visualizing the results.

---

## 1. Object Detection with Faster R-CNN

In this example, we will use a pre-trained Faster R-CNN model with a ResNet-50 backbone. This model was trained on the COCO dataset, which contains 91 common object categories. We will load the model, process an image, and use OpenCV to draw the predicted bounding boxes and labels on it.

### Explanation of the Process

1.  **Load Model**: We load the `fasterrcnn_resnet50_fpn` model from `torchvision.models.detection` and set `pretrained=True` to get the model with weights learned from the COCO dataset. We call `.eval()` to put the model in inference mode.
2.  **Load Image**: We will fetch an image from a URL using the `requests` library and open it using `Pillow` (PIL).
3.  **Transform Image**: The image needs to be converted from a PIL Image to a PyTorch Tensor. `torchvision.transforms` makes this easy.
4.  **Get Predictions**: We pass the image tensor to the model. The model returns a list of predictions, where each prediction is a dictionary containing `boxes`, `labels`, and `scores`.
5.  **Visualize Results**: We loop through the predictions. For each detected object with a confidence score above a certain threshold (e.g., 0.8), we use `OpenCV` to draw the bounding box (`cv2.rectangle`) and the class label (`cv2.putText`) on the image.

### PyTorch Code Example

```python
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import cv2
import numpy as np
import requests

# 1. Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval() # Set the model to evaluation mode

# 2. Define the class names from the COCO dataset
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img_path, threshold=0.8):
    """
    Gets predictions for an image.
    """
    if img_path.startswith('http'):
        img = Image.open(requests.get(img_path, stream=True).raw)
    else:
        img = Image.open(img_path)
    
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img)
    
    with torch.no_grad():
        pred = model([img_tensor])
        
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    
    # Filter predictions based on the threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    
    return pred_boxes, pred_class

def object_detection_api(img_path, output_path='./output_detection.jpg'):
    """
    Detects objects in an image and saves the output.
    """
    boxes, pred_cls = get_prediction(img_path)
    
    if img_path.startswith('http'):
        img = Image.open(requests.get(img_path, stream=True).raw)
    else:
        img = Image.open(img_path)
        
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    for i in range(len(boxes)):
        # Draw bounding box
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=2)
        # Put class name
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
        
    cv2.imwrite(output_path, img)
    print(f"Object detection result saved to {output_path}")

# Example usage with an image from a URL
# This image contains people, cars, and a bus.
url = 'https://www.wsupercars.com/wallpapers-regular/Mercedes-Benz/1990-Mercedes-Benz-190E-Evolution-II-002-1080.jpg'
object_detection_api(url)
```

---

## 2. Image Segmentation with DeepLabV3

For semantic segmentation, we will use the DeepLabV3 model with a ResNet-101 backbone, also pre-trained on a subset of the COCO dataset. This model can classify pixels into 21 categories (including background).

### Explanation of the Process

1.  **Load Model**: We load the `deeplabv3_resnet101` model from `torchvision.models.segmentation`.
2.  **Load and Transform Image**: The process is similar to the detection example. We load an image and apply a series of transforms: resizing, converting to a tensor, and normalizing. Normalization is important because the model was trained on images normalized in a specific way.
3.  **Get Predictions**: The model returns an output dictionary. The key `'out'` contains the predictions, which is an ordered dictionary where each value is a tensor of shape `(N, C, H, W)`, where `N` is the batch size, `C` is the number of classes, and `H, W` are the height and width.
4.  **Process Output**: We take the `argmax` of the output tensor along the class dimension (`dim=1`). This gives us a 2D tensor where each pixel has a value corresponding to its predicted class index.
5.  **Visualize Results**: To make the result human-readable, we create a color map that assigns a unique color to each class. We then create an RGB image from the 2D segmentation map and overlay it on the original image to clearly see the segmented regions.

### PyTorch Code Example

```python
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import numpy as np
import requests
import cv2

# 1. Load the pre-trained DeepLabV3 model
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval() # Set the model to evaluation mode

def decode_segmap(image, nc=21):
    """
    Decodes the segmentation map into a color image.
    """
    # Define the color map for the 21 classes in the COCO subset
    label_colors = np.array([
        (0, 0, 0),  # 0=background
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
    ])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

def segment(img_path, output_path='./output_segmentation.jpg'):
    """
    Performs segmentation on an image and saves the result.
    """
    if img_path.startswith('http'):
        input_image = Image.open(requests.get(img_path, stream=True).raw)
    else:
        input_image = Image.open(img_path)
        
    input_image = input_image.convert("RGB")

    # 3. Preprocess the image
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    output_predictions = output.argmax(0)

    # 4. Decode the prediction and visualize
    mask = decode_segmap(output_predictions.byte().cpu().numpy())
    
    # Resize mask to be the same size as the original image
    original_np = np.array(input_image)
    mask_resized = cv2.resize(mask, (original_np.shape[1], original_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Overlay the mask on the original image
    overlayed_img = cv2.addWeighted(cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR), 0.6, mask_resized, 0.4, 0)

    cv2.imwrite(output_path, overlayed_img)
    print(f"Segmentation result saved to {output_path}")

# Example usage with image URL
url = 'https://www.wsupercars.com/wallpapers/Mercedes-Benz/1986-Mercedes-Benz-560-SEL-008.jpg'
segment(url)
```
