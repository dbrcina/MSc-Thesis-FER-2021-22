from torchvision.transforms import RandomPerspective, ToPILImage

from datasets import ALPROCRDataset

dataset = ALPROCRDataset("data_ocr/train", True)

perspective = RandomPerspective(p=1.0)
to_pil = ToPILImage()

img = dataset[0][0]
img = perspective(img)
to_pil(img).show()
