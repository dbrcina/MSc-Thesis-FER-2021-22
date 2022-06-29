import cv2
import numpy as np
from torchvision import transforms

image = cv2.imread(r"C:\Users\dbrcina\Desktop\MSc-Thesis-FER-2021-22\data\detection\train\1\1-171NVX75.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

t = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomPerspective(distortion_scale=0.1, p=1),
    transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToPILImage()
])

for i in range(3):
    img = np.array(t(image))
    cv2.imshow("Test", img)
    cv2.imwrite(f"{i + 1}.jpg", img)
    cv2.waitKey()
    cv2.destroyWindow("Test")
