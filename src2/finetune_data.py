import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

templates_small = [
    'photo of a {}',
]


templates_small_style = [
    'painting in the style of {}',
]


def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True


class MaskBase(Dataset):
    def __init__(self,
                 datapath,
                 reg_datapath=None,
                 caption=None,
                 reg_caption=None,
                 size=512,
                 interpolation="bicubic",
                 flip_p=0.5,
                 aug=True,
                 style=False,
                 repeat=0.
                 ):

        self.aug = aug
        self.repeat = repeat
        self.style = style
        self.templates_small = templates_small
        if self.style:
            self.templates_small = templates_small_style
        if os.path.isdir(datapath):
            self.image_paths1 = [os.path.join(datapath, file_path) for file_path in os.listdir(datapath) if isimage(file_path)]
        else:
            with open(datapath, "r") as f:
                self.image_paths1 = f.read().splitlines()

        self._length1 = len(self.image_paths1)

        self.image_paths2 = []
        self._length2 = 0
        if reg_datapath is not None:
            if os.path.isdir(reg_datapath):
                self.image_paths2 = [os.path.join(reg_datapath, file_path) for file_path in os.listdir(reg_datapath) if isimage(file_path)]
            else:
                with open(reg_datapath, "r") as f:
                    self.image_paths2 = f.read().splitlines()
            self._length2 = len(self.image_paths2)
        else:
            self.repeat = 10

        self.labels = {
            "relative_file_path1_": [x for x in self.image_paths1],
            "relative_file_path2_": [x for x in self.image_paths2],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.caption = caption

        if os.path.exists(self.caption):
            self.caption = [x.strip() for x in open(caption, 'r').readlines()]

        self.reg_caption = reg_caption
        if os.path.exists(self.reg_caption):
            self.reg_caption = [x.strip() for x in open(reg_caption, 'r').readlines()]

    def __len__(self):
        if self._length2 > 0:
            return 2*self._length2
        elif self.repeat > 0:
            return self._length1*self.repeat
        else:
            return self._length1

    def __getitem__(self, i):
        example = {}

        if i > self._length2 or self._length2 == 0:
            image = Image.open(self.labels["relative_file_path1_"][i % self._length1])
            if isinstance(self.caption, str):
                example["caption"] = np.random.choice(self.templates_small).format(self.caption)
            else:
                example["caption"] = self.caption[i % min(self._length1, len(self.caption)) ]
        else:
            image = Image.open(self.labels["relative_file_path2_"][i % self._length2])
            if isinstance(self.reg_caption, str):
                example["caption"] = np.random.choice(self.templates_small).format(self.reg_caption)
            else:
                example["caption"] = self.reg_caption[i % self._length2]

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]

        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        image = self.flip(image)

        if i > self._length2 or self._length2 == 0:
            if self.aug:
                if np.random.randint(0, 3) < 2:
                    random_scale = np.random.randint(self.size // 3, self.size+1)
                else:
                    random_scale = np.random.randint(int(1.2*self.size), int(1.4*self.size))

                if random_scale % 2 == 1:
                    random_scale += 1
            else:
                random_scale = self.size

            if random_scale < 0.6*self.size:
                add_to_caption = np.random.choice(["a far away ", "very small "])
                example["caption"] = add_to_caption + example["caption"]
                cx = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
                cy = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)

                image = image.resize((random_scale, random_scale), resample=self.interpolation)
                image = np.array(image).astype(np.uint8)
                image = (image / 127.5 - 1.0).astype(np.float32)

                input_image1 = np.zeros((self.size, self.size, 3), dtype=np.float32)
                input_image1[cx - random_scale // 2: cx + random_scale // 2, cy - random_scale // 2: cy + random_scale // 2, :] = image

                mask = np.zeros((self.size // 8, self.size // 8))
                mask[(cx - random_scale // 2) // 8 + 1: (cx + random_scale // 2) // 8 - 1, (cy - random_scale // 2) // 8 + 1: (cy + random_scale // 2) // 8 - 1] = 1.

            elif random_scale > self.size:
                add_to_caption = np.random.choice(["zoomed in ", "close up "])
                example["caption"] = add_to_caption + example["caption"]
                cx = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)
                cy = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)

                image = image.resize((random_scale, random_scale), resample=self.interpolation)
                image = np.array(image).astype(np.uint8)
                image = (image / 127.5 - 1.0).astype(np.float32)
                input_image1 = image[cx - self.size // 2: cx + self.size // 2, cy - self.size // 2: cy + self.size // 2, :]
                mask = np.ones((self.size // 8, self.size // 8))
            else:
                if self.size is not None:
                    image = image.resize((self.size, self.size), resample=self.interpolation)
                input_image1 = np.array(image).astype(np.uint8)
                input_image1 = (input_image1 / 127.5 - 1.0).astype(np.float32)
                mask = np.ones((self.size // 8, self.size // 8))
        else:
            if self.size is not None:
                image = image.resize((self.size, self.size), resample=self.interpolation)
            input_image1 = np.array(image).astype(np.uint8)
            input_image1 = (input_image1 / 127.5 - 1.0).astype(np.float32)
            mask = np.ones((self.size // 8, self.size // 8))

        example["image"] = input_image1
        example["mask"] = mask

        return example
