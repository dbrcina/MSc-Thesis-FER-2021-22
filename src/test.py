from PIL import Image
from torchvision import transforms

from models import CRNN

x = Image.open(r"C:\Users\dbrcina\Desktop\MSc-Thesis-FER-2021-22\data\generated\train\1\187-ZG2877J.jpg")
x = x.resize((100, 32))
x = transforms.ToTensor()(x)
x = transforms.Grayscale()(x)

model = CRNN(37)
model.forward(x[None, :, :, :])
