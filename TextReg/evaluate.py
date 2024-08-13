import os
import argparse
import torch
import clip
import PIL

import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class PersonalizedBase(Dataset):
    def __init__(
        self,
        data_root,
        size=None,
        repeats=1,
        interpolation="bicubic",
        flip_p=0.5,
        set="test",
        placeholder_token="*",
        per_image_tokens=False,
        center_crop=False,
        mixing_prob=0.25,
        coarse_class_text=None,
    ):

        self.data_root = data_root

        self.image_paths = sorted(
            [
                os.path.join(self.data_root, file_path)
                for file_path in os.listdir(self.data_root)
            ]
        )

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {
            "bilinear": PIL.Image.Resampling.BILINEAR,
            "bicubic": PIL.Image.Resampling.BICUBIC,
            "lanczos": PIL.Image.Resampling.LANCZOS,
        }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # placeholder_string = self.placeholder_token
        # if self.coarse_class_text:
        #     placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[
                (h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2
            ]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example = (image / 127.5 - 1.0).astype(np.float32)
        return example


class CLIPEvaluator(object):
    def __init__(self, device, clip_model="ViT-B/32") -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess

        self.preprocess = transforms.Compose(
            [
                transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])
            ]  # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
            + clip_preprocess.transforms[:2]  # to match CLIP input scale assumptions
            + clip_preprocess.transforms[4:]
        )  # + skip convert PIL to tensor

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, text, generated_images):
        text_features = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()


class LDMCLIPEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model="ViT-B/32") -> None:
        super().__init__(device, clip_model)

    def evaluate(self, train_images, gen_samples, prompts):

        sim_img = self.img_to_img_similarity(train_images, gen_samples)
        sim_text, imgs_per_prompt = 0.0, int(gen_samples.shape[0] / len(prompts))
        for i, prompt in enumerate(prompts):
            sim_text += self.txt_to_img_similarity(
                prompt.replace("<new1>", ""),
                gen_samples[i * imgs_per_prompt : (i + 1) * imgs_per_prompt, :, :, :],
            )
        return sim_img, sim_text / len(prompts)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen_data_dir",
        type=str,
        default="logs/logs_toTest/2023-04-08T04-49-07_teddybearreg_0_scale0_a_teddybear_ridge_onlyK_noblip/samples",
    )
    parser.add_argument(
        "--gen_caption_txt",
        type=str,
        default="/home/zhicai/poseVideo/custom-diffusion/data/teddybear.txt",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="/home/zhicai/poseVideo/custom-diffusion/data/teddybear",
    )
    parser.add_argument("--cuda", type=str, default="0")
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    device = torch.device("cuda")

    # Load training data
    print(f"Loading training data from {opt.train_data_dir}")
    train_data_loader = PersonalizedBase(
        opt.train_data_dir, size=512, flip_p=0.0
    )  # [-1, 1]
    train_data = [
        torch.from_numpy(train_data_loader[i]).permute(2, 0, 1)
        for i in range(train_data_loader.num_images)
    ]
    train_data = torch.stack(train_data, axis=0)

    # Load generated data
    print(f"Loading generated data from {opt.gen_data_dir}")
    gen_data_loader = PersonalizedBase(
        opt.gen_data_dir, size=512, flip_p=0.0
    )  # [-1, 1]
    gen_data = [
        torch.from_numpy(gen_data_loader[i]).permute(2, 0, 1)
        for i in range(gen_data_loader.num_images)
    ]
    gen_data = torch.stack(gen_data, axis=0)

    # Load prompts for generation
    print(f"Reading prompts from {opt.gen_caption_txt}")
    with open(opt.gen_caption_txt, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
        prompts = [prompt for prompt in data]

    evaluator = LDMCLIPEvaluator(device)
    sim_img, sim_text = evaluator.evaluate(train_data, gen_data, prompts)

    print("Image similarity: ", sim_img.item())
    print("Text similarity: ", sim_text.item())

    # pdb.set_trace()
    os.makedirs(
        os.path.join(
            opt.gen_data_dir.split("/samples")[0],
            f"{sim_img.item():.4f}_{sim_text.item():.4f}",
        )
    )
