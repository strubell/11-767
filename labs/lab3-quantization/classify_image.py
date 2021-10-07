import torchvision.models as models
import torch
from torchvision import transforms
from PIL import Image

labels = [line.strip() for line in open("imagenet_labels.txt")]
model = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
model.eval()

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

image = Image.open("output.png").convert('RGB')
image = transform(image)
image = normalize(image)
image = image.unsqueeze(0)

out = model(image)
label = labels[torch.argmax(out[0]).item()]
print(f"Found a {label}")
