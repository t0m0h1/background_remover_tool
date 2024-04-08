import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms

# Load pre-trained DeepLabv3+ model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Define image transformations
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your image
image = cv2.imread('images/1.JPG')

# Preprocess the image
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# Perform semantic segmentation
with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# Create mask from predictions
mask = output_predictions.byte().cpu().numpy()

# Apply the mask to the original image
foreground = cv2.bitwise_and(image, image, mask=mask)

# Show original image and foreground
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
plt.title('Foreground')
plt.axis('off')

plt.show()
