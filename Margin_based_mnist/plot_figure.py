import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# load MNIST
transform = transforms.ToTensor()
test_set = datasets.MNIST(root="./mnist", train=False, download=True, transform=transform)

# pick one image
img, label = test_set[0]   # or use any index, or random

# plot
plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()


import random
idx = random.randint(0, len(test_set)-1)
img, label = test_set[idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()
