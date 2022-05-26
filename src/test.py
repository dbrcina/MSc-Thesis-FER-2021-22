import cv2
import numpy as np
from torchvision import transforms

image = cv2.imread(r"C:\Users\dbrcina\Desktop\MSc-Thesis-FER-2021-22\data\detection\val\1\772-ZG4100AC.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

t = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomPerspective(distortion_scale=0.2, p=1),
    transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
    transforms.Resize((32, 100)),
    transforms.ToPILImage()
])

for _ in range(10):
    i = np.array(t(image))
    cv2.imshow("Test", i)
    cv2.waitKey()
    cv2.destroyWindow("Test")
