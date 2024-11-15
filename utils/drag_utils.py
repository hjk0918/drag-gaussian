# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

import copy
import torch
import torch.nn.functional as F

import os
import cv2
import numpy as np
import gradio as gr
from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace

import datetime
import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
import torch.nn.functional as F

from diffusers import DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.models.embeddings import ImageProjection
from drag_pipeline import DragPipeline

from torchvision.utils import save_image
from pytorch_lightning import seed_everything
from tqdm import tqdm

from .lora_utils import train_lora
from .attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl, deregister_attention_editor
from .freeu_utils import register_free_upblock2d, register_free_crossattn_upblock2d


def point_tracking(F0,
                   F1,
                   handle_points,
                   handle_points_init,
                   args):
    with torch.no_grad():
        _, _, max_r, max_c = F0.shape
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]

            r1, r2 = max(0,int(pi[0])-args.r_p), min(max_r,int(pi[0])+args.r_p+1)
            c1, c2 = max(0,int(pi[1])-args.r_p), min(max_c,int(pi[1])+args.r_p+1)
            F1_neighbor = F1[:, :, r1:r2, c1:c2]
            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
            all_dist = all_dist.squeeze(dim=0)
            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            # handle_points[i][0] = pi[0] - args.r_p + row
            # handle_points[i][1] = pi[1] - args.r_p + col
            handle_points[i][0] = r1 + row
            handle_points[i][1] = c1 + col
        return handle_points

def check_handle_reach_target(handle_points,
                              target_points):
    # dist = (torch.cat(handle_points,dim=0) - torch.cat(target_points,dim=0)).norm(dim=-1)
    all_dist = list(map(lambda p,q: (p-q).norm(), handle_points, target_points))
    return (torch.tensor(all_dist) < 2.0).all()

# obtain the bilinear interpolated feature patch centered around (x, y) with radius r
def interpolate_feature_patch(feat,
                              y1,
                              y2,
                              x1,
                              x2):
    x1_floor = torch.floor(x1).long()
    x1_cell = x1_floor + 1
    dx = torch.floor(x2).long() - torch.floor(x1).long()

    y1_floor = torch.floor(y1).long()
    y1_cell = y1_floor + 1
    dy = torch.floor(y2).long() - torch.floor(y1).long()

    wa = (x1_cell.float() - x1) * (y1_cell.float() - y1)
    wb = (x1_cell.float() - x1) * (y1 - y1_floor.float())
    wc = (x1 - x1_floor.float()) * (y1_cell.float() - y1)
    wd = (x1 - x1_floor.float()) * (y1 - y1_floor.float())

    Ia = feat[:, :, y1_floor : y1_floor+dy, x1_floor : x1_floor+dx]
    Ib = feat[:, :, y1_cell : y1_cell+dy, x1_floor : x1_floor+dx]
    Ic = feat[:, :, y1_floor : y1_floor+dy, x1_cell : x1_cell+dx]
    Id = feat[:, :, y1_cell : y1_cell+dy, x1_cell : x1_cell+dx]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd

def drag_diffusion_update(model,
                          init_code,
                          text_embeddings,
                          t,
                          handle_points,
                          target_points,
                          mask,
                          args):

    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"
    if text_embeddings is None:
        text_embeddings = model.get_text_embeddings(args.prompt)

    # the init output feature of unet
    with torch.no_grad():
        unet_output, F0 = model.forward_unet_features(init_code, t,
            encoder_hidden_states=text_embeddings,
            layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
        x_prev_0,_ = model.step(unet_output, t, init_code)
        # init_code_orig = copy.deepcopy(init_code)

    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.latent_lr)

    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (init_code.shape[2],init_code.shape[3]), mode='nearest')
    using_mask = interp_mask.sum() != 0.0

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    for step_idx in range(args.n_pix_step + 1):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            unet_output, F1 = model.forward_unet_features(init_code, t,
                encoder_hidden_states=text_embeddings,
                layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
            x_prev_updated,_ = model.step(unet_output, t, init_code)

            # do point tracking to update handle points before computing motion supervision loss
            if step_idx != 0:
                handle_points = point_tracking(F0, F1, handle_points, handle_points_init, args)
                print('new handle points', handle_points)

            # only do point tracking in the final step
            if step_idx == args.n_pix_step:
                break

            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                break

            loss = 0.0
            _, _, max_r, max_c = F0.shape
            for i in range(len(handle_points)):
                pi, ti = handle_points[i], target_points[i]
                # skip if the distance between target and source is less than 1
                if (ti - pi).norm() < 2.:
                    continue

                di = (ti - pi) / (ti - pi).norm()

                # motion supervision
                # with boundary protection
                r1, r2 = max(0,int(pi[0])-args.r_m), min(max_r,int(pi[0])+args.r_m+1)
                c1, c2 = max(0,int(pi[1])-args.r_m), min(max_c,int(pi[1])+args.r_m+1)
                f0_patch = F1[:,:,r1:r2, c1:c2].detach()
                f1_patch = interpolate_feature_patch(F1,r1+di[0],r2+di[0],c1+di[1],c2+di[1])

                # original code, without boundary protection
                # f0_patch = F1[:,:,int(pi[0])-args.r_m:int(pi[0])+args.r_m+1, int(pi[1])-args.r_m:int(pi[1])+args.r_m+1].detach()
                # f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], args.r_m)
                loss += ((2*args.r_m+1)**2)*F.l1_loss(f0_patch, f1_patch)

            # masked region must stay unchanged
            if using_mask:
                loss += args.lam * ((x_prev_updated-x_prev_0)*(1.0-interp_mask)).abs().sum()
            # loss += args.lam * ((init_code_orig-init_code)*(1.0-interp_mask)).abs().sum()
            print('loss total=%f'%(loss.item()))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return init_code

def drag_diffusion_update_gen(model,
                              init_code,
                              text_embeddings,
                              t,
                              handle_points,
                              target_points,
                              mask,
                              args):

    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"
    if text_embeddings is None:
        text_embeddings = model.get_text_embeddings(args.prompt)

    # positive prompt embedding
    if args.guidance_scale > 1.0:
        unconditional_input = model.tokenizer(
            [args.neg_prompt],
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        unconditional_emb = model.text_encoder(unconditional_input.input_ids.to(text_embeddings.device))[0].detach()
        text_embeddings = torch.cat([unconditional_emb, text_embeddings], dim=0)

    # the init output feature of unet
    with torch.no_grad():
        if args.guidance_scale > 1.:
            model_inputs_0 = copy.deepcopy(torch.cat([init_code] * 2))
        else:
            model_inputs_0 = copy.deepcopy(init_code)
        unet_output, F0 = model.forward_unet_features(model_inputs_0, t, encoder_hidden_states=text_embeddings,
            layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
        if args.guidance_scale > 1.:
            # strategy 1: discard the unconditional branch feature maps
            # F0 = F0[1].unsqueeze(dim=0)
            # strategy 2: concat pos and neg branch feature maps for motion-sup and point tracking
            # F0 = torch.cat([F0[0], F0[1]], dim=0).unsqueeze(dim=0)
            # strategy 3: concat pos and neg branch feature maps with guidance_scale consideration
            coef = args.guidance_scale / (2*args.guidance_scale - 1.0)
            F0 = torch.cat([(1-coef)*F0[0], coef*F0[1]], dim=0).unsqueeze(dim=0)

            unet_output_uncon, unet_output_con = unet_output.chunk(2, dim=0)
            unet_output = unet_output_uncon + args.guidance_scale * (unet_output_con - unet_output_uncon)
        x_prev_0,_ = model.step(unet_output, t, init_code)
        # init_code_orig = copy.deepcopy(init_code)

    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)

    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (init_code.shape[2],init_code.shape[3]), mode='nearest')
    using_mask = interp_mask.sum() != 0.0

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    for step_idx in range(args.n_pix_step):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if args.guidance_scale > 1.:
                model_inputs = init_code.repeat(2,1,1,1)
            else:
                model_inputs = init_code
            unet_output, F1 = model.forward_unet_features(model_inputs, t, encoder_hidden_states=text_embeddings,
                layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
            if args.guidance_scale > 1.:
                # strategy 1: discard the unconditional branch feature maps
                # F1 = F1[1].unsqueeze(dim=0)
                # strategy 2: concat positive and negative branch feature maps for motion-sup and point tracking
                # F1 = torch.cat([F1[0], F1[1]], dim=0).unsqueeze(dim=0)
                # strategy 3: concat pos and neg branch feature maps with guidance_scale consideration
                coef = args.guidance_scale / (2*args.guidance_scale - 1.0)
                F1 = torch.cat([(1-coef)*F1[0], coef*F1[1]], dim=0).unsqueeze(dim=0)

                unet_output_uncon, unet_output_con = unet_output.chunk(2, dim=0)
                unet_output = unet_output_uncon + args.guidance_scale * (unet_output_con - unet_output_uncon)
            x_prev_updated,_ = model.step(unet_output, t, init_code)

            # do point tracking to update handle points before computing motion supervision loss
            if step_idx != 0:
                handle_points = point_tracking(F0, F1, handle_points, handle_points_init, args)
                print('new handle points', handle_points)

            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                break

            loss = 0.0
            _, _, max_r, max_c = F0.shape
            for i in range(len(handle_points)):
                pi, ti = handle_points[i], target_points[i]
                # skip if the distance between target and source is less than 1
                if (ti - pi).norm() < 2.:
                    continue

                di = (ti - pi) / (ti - pi).norm()

                # motion supervision
                # with boundary protection
                r1, r2 = max(0,int(pi[0])-args.r_m), min(max_r,int(pi[0])+args.r_m+1)
                c1, c2 = max(0,int(pi[1])-args.r_m), min(max_c,int(pi[1])+args.r_m+1)
                f0_patch = F1[:,:,r1:r2, c1:c2].detach()
                f1_patch = interpolate_feature_patch(F1,r1+di[0],r2+di[0],c1+di[1],c2+di[1])

                # original code, without boundary protection
                # f0_patch = F1[:,:,int(pi[0])-args.r_m:int(pi[0])+args.r_m+1, int(pi[1])-args.r_m:int(pi[1])+args.r_m+1].detach()
                # f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], args.r_m)

                loss += ((2*args.r_m+1)**2)*F.l1_loss(f0_patch, f1_patch)

            # masked region must stay unchanged
            if using_mask:
                loss += args.lam * ((x_prev_updated-x_prev_0)*(1.0-interp_mask)).abs().sum()
            # loss += args.lam * ((init_code_orig - init_code)*(1.0-interp_mask)).abs().sum()
            print('loss total=%f'%(loss.item()))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return init_code


class DragWrapper:
    def __init__(self, images, points, masks, args) -> None:
        self.args = args
        self.prompt = args.prompt

        # train lora
        self.lora_path = os.path.join(args.output_path, 'lora_weights')
        model_path = args.diffusion_model

        if args.share_lora:
            print('using shared LoRA weights from all views')
            if not os.path.isfile(os.path.join(self.lora_path, "pytorch_lora_weights.safetensors")):
                print(f'training lora: {self.lora_path}')
                # We use all the images to train lora
                train_lora(images, args.prompt, model_path, args.vae_path, self.lora_path,
                        args.lora_step, args.lora_lr, args.lora_batch_size, args.lora_rank)
            else:
                print("Found LoRA weights, skip training!")
        else:
            print('using individual LoRA weights for each view')
            for i in range(len(images)):
                view_path = os.path.join(self.lora_path, f'view_{i:03d}')
                if not os.path.isfile(os.path.join(view_path, "pytorch_lora_weights.safetensors")):
                    print(f'training lora for view {i+1}/{len(images)}')
                    train_lora(images[i:i+1], args.prompt, model_path, args.vae_path, view_path,
                            args.lora_step, args.lora_lr, args.lora_batch_size, args.lora_rank)
                else:
                    print(f"Found LoRA weights for view {i}, skip training!")

                torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        
        # initialize model
        device = args.drag_device
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                            beta_schedule="scaled_linear", clip_sample=False,
                            set_alpha_to_one=False, steps_offset=1)
        model = DragPipeline.from_pretrained(model_path, scheduler=scheduler, 
                                             torch_dtype=torch.float16)
        # call this function to override unet forward function,
        # so that intermediate features are returned after forward
        model.modify_unet_forward()

        # set vae
        if args.vae_path != "default":
            model.vae = AutoencoderKL.from_pretrained(
                args.vae_path
            ).to(model.vae.device, model.vae.dtype)

        # off load model to cpu, which save some memory.
        model.enable_model_cpu_offload(device=args.drag_device)

        # initialize parameters
        # seed = 42 # random seed used by a lot of people for unknown reason
        # seed_everything(seed)

        # print("applying default parameters")
        # model.unet.set_default_attn_processor()
        if args.share_lora:
            print("applying lora: " + self.lora_path)
            model.unet.load_attn_procs(self.lora_path)

        images = self.preprocess_image(images, device, dtype=torch.float16)
        self.images = images

        # preparing editing meta data (handle, target, mask)
        # mask = torch.from_numpy(mask).float() / 255.
        # mask[mask > 0.0] = 1.0
        self.masks = torch.from_numpy(masks).float()
        self.masks = rearrange(self.masks, "b h w -> b 1 h w").to(device)

        handle_points = []
        target_points = []
        # here, the point is in x,y coordinate
        for idx, points_single_image in enumerate(points):
            handle_points_single_image = []
            target_points_single_image = []
            for point in points_single_image:
                handle_point = torch.tensor([point[0, 1], point[0, 0]])
                target_point = torch.tensor([point[1, 1], point[1, 0]])
                handle_points_single_image.append(handle_point)
                target_points_single_image.append(target_point)
                # cur_point = torch.tensor([point[1], point[0]])
                # cur_point = torch.round(cur_point)
                # if idx % 2 == 0:
                #     handle_points.append(cur_point)
                # else:
                #     target_points.append(cur_point)
            handle_points.append(handle_points_single_image)
            target_points.append(target_points_single_image)
        # print('handle points:', handle_points)
        # print('target points:', target_points)

        # obtain text embeddings
        self.text_embeddings = model.get_text_embeddings(args.prompt)

        # invert the source image
        # the latent code resolution is too small, only 64*64
        self.img_init_code = []
        for i in tqdm(range(images.shape[0]), desc='invert images'):
            invert_code = model.invert(images[i:i+1], args.prompt,
                                    encoder_hidden_states=self.text_embeddings,
                                    guidance_scale=args.guidance_scale,
                                    num_inference_steps=args.n_inference_step,
                                    num_actual_inference_steps=args.n_actual_inference_step)
            self.img_init_code.append(invert_code.detach())

        # empty cache to save memory
        torch.cuda.empty_cache()

        # self.init_code = torch.concat(invert_codes, dim=0)
        model.scheduler.set_timesteps(args.n_inference_step)
        self.t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]

        self.model = model

        self.init_handles_points = deepcopy(handle_points)
        self.handle_points = handle_points
        self.target_points = target_points
        self.img_code = [None] * len(images)

    def preprocess_image(self, image, device, dtype=torch.float32):
        image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
        image = rearrange(image, "b h w c -> b c h w")
        image = image.to(device, dtype)
        return image

    def track_points(self, prev_code, cur_code, handles):
        args = self.args
        with torch.no_grad():
            _, F0 = self.model.forward_unet_features(prev_code, self.t,
                encoder_hidden_states=self.text_embeddings,
                layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, 
                interp_res_w=args.sup_res_w)
            _, F1 = self.model.forward_unet_features(cur_code, self.t,
                encoder_hidden_states=self.text_embeddings,
                layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, 
                interp_res_w=args.sup_res_w)

            handles_copy = deepcopy(handles)
            handles = point_tracking(F0, F1, handles, handles_copy, args)
            print('new handle points', handles)

    def update(self, images, image_inds):
        args = self.args
        updated_images = []
        for i, ind in tqdm(enumerate(image_inds), desc='update images'):
            if not args.share_lora:
                lora_path = os.path.join(self.lora_path, f'view_{ind:03d}')
                self.model.unet.load_attn_procs(lora_path)

            init_code = self.model.invert(images[i:i+1], self.prompt,
                                encoder_hidden_states=self.text_embeddings,
                                guidance_scale=args.guidance_scale,
                                num_inference_steps=args.n_inference_step,
                                num_actual_inference_steps=args.n_actual_inference_step)

            if self.img_code[ind] is not None:
                # do point tracking on the rendered image
                self.track_points(self.img_code[ind], init_code, self.handle_points[ind])

            init_code = init_code.float()
            self.text_embeddings = self.text_embeddings.float()
            self.model.unet = self.model.unet.float()

            # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
            updated_init_code = drag_diffusion_update(
                self.model,
                init_code,
                self.text_embeddings,
                self.t,
                self.handle_points[ind],
                self.target_points[ind],
                self.masks[ind:ind+1],
                self.args
            )
            
            updated_init_code = updated_init_code.half()
            self.text_embeddings = self.text_embeddings.half()
            self.model.unet = self.model.unet.half()

            # empty cache to save memory
            torch.cuda.empty_cache()

            update_image = self.render(ind, updated_init_code)
            updated_images.append(update_image)
            self.img_code[ind] = updated_init_code.detach()

        return updated_images
    
    def render(self, image_ind, updated_code):
        args = self.args
        # hijack the attention module
        # inject the reference branch to guide the generation
        editor = MutualSelfAttentionControl(start_step=args.start_step,
                                            start_layer=args.start_layer,
                                            total_steps=args.n_inference_step,
                                            guidance_scale=args.guidance_scale)
        # register_attention_editor_diffusers(model, editor, attn_processor='attn_proc')
        register_attention_editor_diffusers(self.model, editor, attn_processor='lora_attn_proc')

        # inference the synthesized image
        # gen_image = self.model(
        #     prompt=self.prompt,
        #     encoder_hidden_states=self.text_embeddings,
        #     batch_size=1,
        #     latents=updated_code,
        #     guidance_scale=args.guidance_scale,
        #     num_inference_steps=args.n_inference_step,
        #     num_actual_inference_steps=args.n_actual_inference_step
        # )

        gen_image = self.model(
            prompt=self.prompt,
            encoder_hidden_states=torch.cat([self.text_embeddings]*2, dim=0),
            batch_size=2,
            latents=torch.cat([self.img_init_code[image_ind], updated_code], dim=0),
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.n_inference_step,
            num_actual_inference_steps=args.n_actual_inference_step
        )[1].unsqueeze(dim=0)

        deregister_attention_editor(self.model)

        # resize gen_image into the size of source_image
        # we do this because shape of gen_image will be rounded to multipliers of 8
        # gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')

        gen_image = gen_image.float()
        # out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
        # out_image = (out_image * 255).astype(np.uint8)
        return gen_image
