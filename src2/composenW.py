# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.


import sys
import os
import argparse
import random
import torch
import torchvision

import numpy as np
from tqdm import tqdm
from scipy.linalg import lu_factor, lu_solve

sys.path.append('stable-diffusion')
sys.path.append('./')
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


"""_summary_

1. config and model coposition

"""
def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def get_model(path):
    config = OmegaConf.load("configs/custom-diffusion/finetune.yaml")
    model = load_model_from_config(config, path)
    return model, config



def compose(paths,base_configs, category, outpath, pretrained_model_path, prompts, save_path, device='cuda',use_lasso=True):
    model, config = get_model(pretrained_model_path)
    model.to(device)
    model.eval()
    model.requires_grad = False
    pretrain_model_st = model.state_dict()
    layers = []
    layers_modified = []

    def getlayers(model, root_name=''):
        for name, module in model.named_children():
            if module.__class__.__name__ == 'SpatialTransformer':
                layers_modified.append(root_name + '.' + name + '.transformer_blocks.0.attn2.to_k')
                layers_modified.append(root_name + '.' + name + '.transformer_blocks.0.attn2.to_v')
            else:
                if list(module.children()) == []:
                    layers.append(root_name + '.' + name)
                else:
                    getlayers(module, root_name + '.' + name)

    getlayers(model.model.diffusion_model)

    for i in range(len(layers_modified)):
        layers_modified[i] = 'model.diffusion_model' + layers_modified[i] + '.weight'

    def get_text_embedding(prompts):
        with torch.no_grad():
            uc = []
            for text in prompts:
                tokens = tokenizer(text,
                                   truncation=True,
                                   max_length=77,
                                   return_length=True,
                                   return_overflowing_tokens=False,
                                   padding="max_length",
                                   return_tensors="pt")

                tokens = tokens["input_ids"]
                end = torch.nonzero(tokens == 49407)[:, 1].min()
                if 'photo of a' in text[:15]:
                    print(text)
                    uc.append((model.get_learned_conditioning(1 * [text])[:, 4:end+1]).reshape(-1, 768))
                else:
                    uc.append((model.get_learned_conditioning(1 * [text])[:, 1:end+1]).reshape(-1, 768))

        return torch.cat(uc, 0)

    tokenizer = model.cond_stage_model.tokenizer
    embeds = []
    class_biases =[]
    count = 1

    model2_sts = []
    modifier_tokens = []
    categories = []
    config.model.params.cond_stage_config.params = {}
    config.model.params.cond_stage_config.params.modifier_token = None
    for path1, base_config1,cat1 in zip(paths.split('+'), base_configs,category):
        model2_st = torch.load(path1,map_location='cpu')
        if 'embed' in model2_st['state_dict']:
            config.model.params.cond_stage_config.target = 'src2.custom_modules.FrozenCLIPEmbedderWrapper'
            embeds.append(model2_st['state_dict']['embed'][-1:])
            num_added_tokens1 = tokenizer.add_tokens(f'<new{count}>')
            modifier_token_id1 = tokenizer.convert_tokens_to_ids('<new1>')
            for key in model2_st['state_dict'].keys():
                if 'cond_stage_model.class_manager' in key:
                    class_biases.append(model2_st['state_dict'][key])
            modifier_tokens.append(True)
            if config.model.params.cond_stage_config.params.modifier_token is None:
                config.model.params.cond_stage_config.params.modifier_token = f'<new{count}>'
                config.model.params.cond_stage_config.params.concept_classes = [f'<new{count}> {cat1}']
                if hasattr(base_config1.model.params.cond_stage_config.params, 'bias_strengths'):
                    config.model.params.cond_stage_config.params.bias_strengths = base_config1.model.params.cond_stage_config.params.bias_strengths
                else:
                    config.model.params.cond_stage_config.params.bias_strengths = [1.0]
            else:
                config.model.params.cond_stage_config.params.modifier_token += f'+<new{count}>'
                config.model.params.cond_stage_config.params.concept_classes += [f'<new{count}> {cat1}']
                if hasattr(base_config1.model.params.cond_stage_config.params, 'bias_strengths'):
                    config.model.params.cond_stage_config.params.bias_strengths += base_config1.model.params.cond_stage_config.params.bias_strengths
                else:
                    config.model.params.cond_stage_config.params.bias_strengths += [1.0]
        else:
            modifier_tokens.append(False)

        model2_sts.append(model2_st['state_dict'])
        categories.append(cat1)
        count += 1

    embeds = torch.cat(embeds, 0)
    model.cond_stage_model = instantiate_from_config(config.model.params.cond_stage_config)
    token_embeds = model.cond_stage_model.transformer.get_input_embeddings().weight.data
    token_embeds[-embeds.size(0):] = embeds
    for i in range(len(class_biases)):
        getattr(model.cond_stage_model, f'class_manager_{i}').class_bias.data = class_biases[i]

    # f = open(regularization_prompt, 'r')
    # prompt = [x.strip() for x in f.readlines()][:200]
    from collections import defaultdict
    uc_v = defaultdict(list)
    uc_b = defaultdict(list)
    uc = []
    for model2_st, base_config in zip(model2_sts,base_configs):
        prompt = []
        prompt += base_config.model.params.reg_prompt
        prompt += [base_config.data.params.train.params.caption]
        print("constrainted", prompt)
        uc_target = get_text_embedding(prompt)
        uc.append(uc_target)
        for each in layers_modified:
            # [2,1024]
            uc_v[each].append((model2_st[each].to(device)@uc_target.T).T)
    uc = torch.concat(uc, 0)
    
    for each in layers_modified:
        uc_b[each] = (pretrain_model_st[each].clone().to(device)@uc.T).T
        uc_v[each] = torch.cat(uc_v[each], 0)
        # uc_v[each] = torch.stack([uc_v[each][i] for i in range(uc_v[each].size(0))], 0)
        print(uc_v[each].size(), each)


    new_weights = {}
    if use_lasso:
        from sklearn import linear_model
        import numpy as np
        for each in layers_modified:
            Y = np.array(uc_v[each] - uc_b[each]) # 2n x 1024
            X = np.array(uc) # 2n x 768
            clf = linear_model.Lasso(alpha=0.01,tol=1e-3,selection='random',random_state=0)
            clf.fit(X, Y)
            # 1024 x 768
            new_weights[each] = torch.from_numpy(clf.coef_).to(device) + pretrain_model_st[each].clone().to(device)
    else:
        # use the average
        for each in layers_modified:
            new_weights[each] = torch.stack([model_st[each] for model_st in model2_sts]).mean(0).to(device)

    if prompts is not None:
        model.load_state_dict(new_weights, strict=False)
        sampler = DDIMSampler(model)
        sampler.make_schedule(ddim_num_steps=200, ddim_eta=1., verbose=False)

        seed = 68
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        batch_size = 10

        if not os.path.exists(prompts):
            assert prompts is not None
            prompts = [batch_size * [prompts]]

        else:
            print(f"reading prompts from {prompts}")
            with open(prompts, "r") as f:
                prompts = f.read().splitlines()
                prompts = [batch_size * [prompt] for prompt in prompts]
                print(prompts[0])

        sample_path = os.path.join(f'{save_path}/{outpath}/', 'samples')
        os.makedirs(sample_path, exist_ok=True)
        with torch.no_grad():
            for counter, prompt in enumerate(prompts):
                print(prompt)
                uc_try = model.get_learned_conditioning(batch_size * [prompt[0]])

                unconditional_guidance_scale = 6.
                cond = uc_try
                unconditional_conditioning = model.get_learned_conditioning(batch_size * [""])

                img = torch.randn((batch_size, 4, 64, 64)).to(device)
                ddim_use_original_steps = False

                timesteps = sampler.ddpm_num_timesteps if ddim_use_original_steps else sampler.ddim_timesteps
                time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
                total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
                iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

                for i, step in enumerate(iterator):
                    index = total_steps - i - 1
                    ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
                    outs = sampler.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
                    img, _ = outs

                outim = model.decode_first_stage(outs[0])
                outim = torch.clamp((outim + 1.0) / 2.0, min=0.0, max=1.0)
                name = '-'.join(prompt[0].split(' '))
                torchvision.utils.save_image(outim, f'{save_path}/{outpath}/{counter}_{name}.jpg', nrow=batch_size // 2)

    new_weights['embed'] = embeds
    os.makedirs(f'{save_path}/{outpath}', exist_ok=True)
    os.makedirs(f'{save_path}/{outpath}/checkpoints', exist_ok=True)
    os.makedirs(f'{save_path}/{outpath}/configs', exist_ok=True)
    with open(f'{save_path}/{outpath}/configs/config_project.yaml', 'w') as fp:
        OmegaConf.save(config=config, f=fp)
    torch.save({'state_dict': new_weights}, f'{save_path}/{outpath}/checkpoints/delta_epoch=000000.ckpt')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--paths', help='+ separated list of checkpoints', required=True,
                        type=str)
    parser.add_argument('--save_path', help='folder name to save  optimized weights', default='optimized_logs',
                        type=str)
    parser.add_argument('--prompts', help='prompts for composition model (can be a file or string)', default=None,
                        type=str)
    parser.add_argument('--ckpt', required=True,
                        type=str)
    parser.add_argument('--use_lasso', help='use lasso regression to optimize weights', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    # python src2/composenW.py --paths "logs/2023-04-13T08-02-21_barn_classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000022-step=000000199.ckpt+logs/2023-04-13T07-43-35_wooden_pot_reg0.1-0.1_scale0-0/checkpoints/epoch=000039-step=000000199.ckpt" --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt
    args = parse_args()
    paths = args.paths
    base_configs = []
    for path in paths.split('+'):
        base_configs.append(OmegaConf.load(os.path.join(*path.split('/')[:-2], 'configs', [f for f in os.listdir(os.path.join(*path.split('/')[:-2], 'configs')) if 'project' in f][0])))

    categories = [base_config.data.params.train.params.caption.split(' ',1)[-1] for base_config in base_configs]
    outpath = '_'.join(['optimized', '_'.join(categories).replace(' ', '_')])
    outpath +='_lasso' if args.use_lasso else '_average'
    print(outpath)
    compose(paths, base_configs, categories,outpath, args.ckpt, args.prompts, args.save_path,device='cpu',use_lasso=args.use_lasso)
