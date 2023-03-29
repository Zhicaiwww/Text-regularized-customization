import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import cv2

templates_small = [
    'photo of a {}, (photorealistic:1.2), simple background,a weak light and shadow effect on the top right corner'
]


templates_small_style = [
    'painting in the style of {}',
]


def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True


class MaskBasePose(Dataset):
    def __init__(self,
                 datapath,
                 reg_datapath=None,
                 caption=None,
                 reg_caption=None,
                 size=512,
                 interpolation="bicubic",
                 flip_p=0.5,
                 aug=False,
                 style=False,
                 repeat=0.,
                 mask_pose=False,
                 ):

        self.aug = aug
        self.repeat = repeat
        self.style = style
        self.flip_p = flip_p
        self.templates_small = templates_small
        self.mask_pose = mask_pose
        if self.style:
            self.templates_small = templates_small_style
        
        imgpath = os.path.join(datapath,'images')
        posepath = os.path.join(datapath,'pose')
        maskpath = os.path.join(datapath,'mask')
        if reg_datapath is not None:
            reg_imgpath = os.path.join(reg_datapath,'images')
            reg_posepath = os.path.join(reg_datapath,'pose')
            # reg_maskpath = os.path.join(reg_datapath,'mask')
        file_paths = [file_path for file_path in os.listdir(imgpath) if isimage(file_path)]
        self.image_paths1 = [os.path.join(imgpath, file_path) for file_path in file_paths]
        self.pose_paths1 = [os.path.join(posepath, file_path) for file_path in file_paths]
        self.mask_paths1 = [os.path.join(maskpath, file_path) for file_path in file_paths]
        self._length1 = len(self.image_paths1)
        assert self._length1 == len(self.pose_paths1)

        self.image_paths2 = []
        self._length2 = 0
        if reg_datapath is not None:
            file_paths2 = [file_path for file_path in os.listdir(reg_posepath) if isimage(file_path)]
            self.image_paths2 = [os.path.join(reg_imgpath, file_path) for file_path in file_paths2]
            self.pose_paths2 = [os.path.join(reg_posepath, file_path) for file_path in file_paths2]
            # self.mask_paths2 = [os.path.join(reg_maskpath, file_path) for file_path in file_paths]
            self._length2 = len(self.image_paths2)
            assert self._length2 == len(self.pose_paths2)

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
        self.flip = torchvision.transforms.functional.hflip
        
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
            pose = Image.open(self.pose_paths1[i % self._length1])
            human_mask = Image.open(self.mask_paths1[i % self._length1]) if self.mask_pose else None

            if isinstance(self.caption, str):
                example["caption"] = np.random.choice(self.templates_small).format(self.caption)
            else:
                example["caption"] = self.caption[i % min(self._length1, len(self.caption)) ]
        else:
            image = Image.open(self.labels["relative_file_path2_"][i % self._length2])
            pose = Image.open(self.pose_paths2[i % self._length2])
            # human_mask = Image.open(self.mask_paths2[i % self._length2])
            human_mask = None
            if isinstance(self.reg_caption, str):
                example["caption"] = np.random.choice(self.templates_small).format(self.reg_caption)
            else:
                example["caption"] = self.reg_caption[i % self._length2]

        if not image.mode == "RGB":
            image = image.convert("RGB")
        if not pose.mode == "RGB":
            pose = pose.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        ps = np.array(pose).astype(np.uint8)

        
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]

        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
        
        ps = ps[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
        if human_mask is not None:
            hm = np.array(human_mask).astype(np.uint8)
            hm = hm[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
            hm = Image.fromarray(hm)
        else: 
            hm = None
        image = Image.fromarray(img)
        pose = Image.fromarray(ps)


        if np.random.random_sample()<self.flip_p:
            image = self.flip(image)
            pose = self.flip(pose)
            if hm is not None:
                hm = self.flip(hm) 

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
                input_pose1 = np.array(pose).astype(np.uint8)
                input_pose1 = (input_pose1 / 255.).astype(np.float32)
                mask = np.ones((self.size // 8, self.size // 8))
                if hm is not None:
                    human_mask = np.array(hm.resize((self.size // 8, self.size // 8),resample=cv2.INTER_AREA)) /255.
                    mask = mask * human_mask
        else:
            if self.size is not None:
                image = image.resize((self.size, self.size), resample=self.interpolation)
            input_image1 = np.array(image).astype(np.uint8)
            input_image1 = (input_image1 / 127.5 - 1.0).astype(np.float32)
            input_pose1 = np.array(pose).astype(np.uint8)
            input_pose1 = (input_pose1 / 255.).astype(np.float32)
            mask = np.ones((self.size // 8, self.size // 8))
            if hm is not None:
                human_mask = np.array(hm.resize((self.size // 8, self.size // 8),resample=cv2.INTER_AREA)) /255.
                mask = mask * human_mask

        example["image"] = input_image1
        example["mask"] = mask
        example["pose"] = input_pose1



        return example
if __name__ == '__main__':
    import sys
    sys.path.append('../')
    data = MaskBasePose(datapath='data/edn_subject2_train20',caption='<new1> woman',reg_caption = 'man')
    import cv2
    input_pose1 = (data[1]['mask']*255)
    cv2.imwrite("pose.png",input_pose1)