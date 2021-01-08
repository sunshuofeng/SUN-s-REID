
import math
import random
import torchvision.transforms as T
class RandomErasing(object):

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def build_transforms(cfg, training=True):
    normalize_transform = T.Normalize(mean=cfg['mean'], std=cfg['std'])
    if training:
        transform = T.Compose([
            T.Resize(cfg['train_size']),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(cfg['padding']),
            T.RandomCrop(cfg['train_size']),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5, mean=cfg['mean'])
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg['test_size']),
            T.ToTensor(),
            normalize_transform
        ])

    return transform