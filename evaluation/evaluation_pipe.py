import os, random, pdb
import math
from PIL import Image
from glob import glob
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import (
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    logging,
)
from custom_datasets.utils import parse_templates_from_superclass
from cleanfid.utils import ResizeDataset
from cleanfid.features import build_feature_extractor


EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
              'tif', 'tiff', 'webp', 'npy', 'JPEG', 'JPG', 'PNG'}


def image_grid(_imgs, rows=None, cols=None):

    if rows is None and cols is None:
        rows = cols = math.ceil(len(_imgs) ** 0.5)

    if rows is None:
        rows = math.ceil(len(_imgs) / cols)
    if cols is None:
        cols = math.ceil(len(_imgs) / rows)

    w, h = _imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(_imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def text_img_alignment(img_embeds, text_embeds, target_img_embeds):
    # evaluation inspired from textual inversion paper
    # https://arxiv.org/abs/2208.01618

    log = {}

    # text alignment
    if text_embeds is not None:
        text_img_sim = (img_embeds * text_embeds).sum(dim=-1) / (
            img_embeds.norm(dim=-1) * text_embeds.norm(dim=-1)
        )
        log["text_alignment_avg"] = text_img_sim.mean().item()
        log["text_alignment_all"] = text_img_sim.tolist()

    # image alignment
    if target_img_embeds is not None:
        img_embed_normalized = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        avg_target_img_embed = (
            (target_img_embeds / target_img_embeds.norm(dim=-1, keepdim=True))
            .mean(dim=0)
            .unsqueeze(0)
            .repeat(img_embeds.shape[0], 1)
        )
        img_img_sim = (img_embed_normalized * avg_target_img_embed).sum(dim=-1)
        log["image_alignment_avg"] = img_img_sim.mean().item()
        log["image_alignment_all"] = img_img_sim.tolist()

    return log


def prepare_clip_model_sets(eval_clip_id: str = "models/clip-vit-large-patch14"):
    logging.set_verbosity_error()
    text_model = CLIPTextModelWithProjection.from_pretrained(eval_clip_id)
    tokenizer = CLIPTokenizer.from_pretrained(eval_clip_id)
    vis_model = CLIPVisionModelWithProjection.from_pretrained(eval_clip_id)
    processor = CLIPProcessor.from_pretrained(eval_clip_id)

    return text_model, tokenizer, vis_model, processor


def get_folder_features(fdir, model=None, num_workers=12, num=None,
                        shuffle=False, seed=0, batch_size=128, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None):
    # get all relevant files in the dataset

    files = sorted([file for ext in EXTENSIONS
                for file in glob(os.path.join(fdir, f"**/*.{ext}"), recursive=True)])
    if verbose:
        print(f"Found {len(files)} images in the folder {fdir}")
    # use a subset number of files if needed
    if num is not None:
        if shuffle:
            random.seed(seed)
            random.shuffle(files)
        files = files[:num]
    ts_feats = get_files_features(files, model, num_workers=num_workers,
                                  batch_size=batch_size, device=device, mode=mode,
                                  custom_fn_resize=custom_fn_resize,
                                  custom_image_tranform=custom_image_tranform,
                                  description=description, fdir=fdir, verbose=verbose)
    return ts_feats


def get_files_features(l_files, model=None, num_workers=12,
                       batch_size=128, device=torch.device("cuda"),
                       mode="clean", custom_fn_resize=None,
                       description="", fdir=None, verbose=True,
                       custom_image_tranform=None):
    # wrap the images in a dataloader for parallelizing the resize operation
    dataset = ResizeDataset(l_files, fdir=fdir, mode=mode)
    if custom_image_tranform is not None:
        dataset.custom_image_tranform=custom_image_tranform
    if custom_fn_resize is not None:
        dataset.fn_resize = custom_fn_resize

    dataloader = torch.utils.data.DataLoader(dataset,
                    batch_size=batch_size, shuffle=False,
                    drop_last=False, num_workers=num_workers)

    # collect all inception features
    l_feats = []
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader
    
    for batch in pbar:
        with torch.no_grad():
            feat = model(batch.to(device))
        l_feats.append(feat.detach())
    ts_feats = torch.cat(l_feats)
    return ts_feats


def kernel_distance(feats1, feats2, num_subsets=100, max_subset_size=1000):
    n = feats1.shape[1]
    m = min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)
    t = 0

    for _subset_idx in range(num_subsets):
        indices_x = torch.randperm(feats2.shape[0])[:m]
        indices_y = torch.randperm(feats1.shape[0])[:m]

        x = feats2[indices_x]
        y = feats1[indices_y]

        a = (torch.mm(x, x.t()) / n + 1) ** 3 + (torch.mm(y, y.t()) / n + 1) ** 3
        b = (torch.mm(x, y.t()) / n + 1) ** 3

        t += (a.sum() - torch.diag(a).sum()) / (m - 1) - b.sum() * 2 / m

    kid = t / num_subsets / m
    return float(kid)


def evaluate_pipe1(
    pipe,
    superclass: str,
    target_dir: str,
    learnt_token: str,
    save_path: str,
    guidance_scale: float = 5.0,
    batch_size: int = 5,
    n_test: int = 10,
    n_step: int = 50,
    device: str = "cuda"
    ):

    # Preparation for sampling
    images, prompt_list, img_embeds, text_embeds = [], [], [], []
    placeholder = learnt_token.split(' ')[0] if len(learnt_token.split(' ')) > 1 else None
    templates = parse_templates_from_superclass(superclass, placeholder=placeholder)
    example_templates = (templates * 50)[:n_test]
    prompt_dataset = DataLoader(example_templates, batch_size=batch_size, shuffle=True)

    for prompts in tqdm(prompt_dataset, desc="Generation for CLIP-T"):
        with torch.autocast("cuda"):
            imgs_ = pipe(prompts, num_inference_steps=n_step, guidance_scale=guidance_scale).images
        images.extend(imgs_)
        prompt_list.extend(prompts)

    if save_path is not None:
        os.makedirs(os.path.join(save_path, "samples1"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "grids1"), exist_ok=True)
        for idx, img in enumerate(images):
            img.save(os.path.join(save_path, "samples1", f"{idx:05}_{prompt_list[idx].replace(' ', '_')}.png"))
        for i in range(0, len(images), batch_size**2):
            slice = images[i:i+batch_size**2]
            grid = image_grid(slice, cols=batch_size)
            grid.save(os.path.join(save_path, "grids1", f"{(i):05}-{(i+len(slice)-1):05}.png"))

    # Init for CLIP model 
    text_model, tokenizer, vis_model, processor = prepare_clip_model_sets("models/clip-vit-large-patch14")


    # region [Text Align Evaluation]
    inputs = processor(images=images, return_tensors="pt")
    img_embed = vis_model(**inputs).image_embeds
    img_embeds.append(img_embed)
    img_embeds = torch.cat(img_embeds)

    prompts_wo_placeholder = [prompt.replace(learnt_token, superclass) for prompt in prompt_list]
    inputs = tokenizer(prompts_wo_placeholder, padding=True, return_tensors="pt")
    outputs = text_model(**inputs)
    text_embed = outputs.text_embeds
    text_embeds.append(text_embed)

    text_embeds = torch.cat(text_embeds, dim=0)
    log_text = text_img_alignment(img_embeds, text_embeds, target_img_embeds=None)
    # endregion


    # region [Image Align Evaluation]
    target_images = []
    for file in os.listdir(target_dir):
        if ( file.lower().endswith((".png", ".jpg", ".jpeg"))):
            target_images.append(Image.open(os.path.join(target_dir, file)))
    inputs = processor(images=target_images, return_tensors="pt")
    target_img_embeds = vis_model(**inputs).image_embeds
    log_img = text_img_alignment(img_embeds, None, target_img_embeds)
    # endregion

    return {**log_text, **log_img}


def evaluate_pipe2(
    pipe,
    superclass: str,
    target_dir: str,
    learnt_token: str,
    save_path: str,
    guidance_scale: float = 5.0,
    batch_size: int = 5,
    n_test: int = 10,
    n_step: int = 50,
    device: str = "cuda"
    ):

    # Preparation for sampling
    images, prompt_list, img_embeds, gen_feats = [], [], [], []
    example_templates = [f"photo of a {learnt_token}"] * n_test
    prompt_dataset = DataLoader(example_templates, batch_size=batch_size, shuffle=True)

    for prompts in tqdm(prompt_dataset, desc="Generation for CLIP-I and KID"):
        with torch.autocast("cuda"):
            imgs_ = pipe(prompts, num_inference_steps=n_step, guidance_scale=guidance_scale).images
        images.extend(imgs_)
        prompt_list.extend(prompts)

    if save_path is not None:
        os.makedirs(os.path.join(save_path, "samples2"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "grids2"), exist_ok=True)
        for idx, img in enumerate(images):
            img.save(os.path.join(save_path, "samples2", f"{idx:05}_{prompt_list[idx].replace(' ', '_')}.png"))
        for i in range(0, len(images), batch_size**2):
            slice = images[i:i+batch_size**2]
            grid = image_grid(slice, cols=batch_size)
            grid.save(os.path.join(save_path, "grids2", f"{(i):05}-{(i+len(slice)-1):05}.png"))

    # Clip model 
    text_model, tokenizer, vis_model, processor = prepare_clip_model_sets("models/clip-vit-large-patch14")


    # region [Image Align Evaluation]
    inputs = processor(images=images, return_tensors="pt")
    img_embed = vis_model(**inputs).image_embeds
    img_embeds.append(img_embed)
    img_embeds = torch.cat(img_embeds)

    target_images = []
    for file in os.listdir(target_dir):
        if (file.lower().endswith((".png", ".jpg", ".jpeg"))):
            target_images.append(Image.open(os.path.join(target_dir, file)))
    inputs = processor(images=target_images, return_tensors="pt")
    target_img_embeds = vis_model(**inputs).image_embeds
    log_img = text_img_alignment(img_embeds, None, target_img_embeds)
    # endregion


    # region [KID Evaluation]
    feat_model = build_feature_extractor(mode="clean", use_dataparallel=False, device=device)
    source_feats = get_folder_features(target_dir, feat_model, verbose=False)
    transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    imgs_tensor_299 = torch.stack([transform(img) for img in images])
    with torch.no_grad():
        gen_feats = feat_model((imgs_tensor_299 * 255.).to(device))
    log_img["KID_score"] = 1e3 * kernel_distance(source_feats, gen_feats)
    # endregion

    return log_img