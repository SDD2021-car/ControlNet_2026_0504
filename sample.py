import os
from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
import einops
from PIL import Image
from tqdm import tqdm
#1
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from annotator.util import resize_image

def save_image(img_arr, path):
    Image.fromarray(img_arr).save(path)

def concat_side_by_side(img1, img2):
    h = max(img1.shape[0], img2.shape[0])
    w = img1.shape[1] + img2.shape[1]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:img1.shape[0], :img1.shape[1]] = img1
    out[:img2.shape[0], img1.shape[1]:] = img2
    return out

def concat_three(imgA, imgB, imgC):
    # 三图并排拼接
    h = max(imgA.shape[0], imgB.shape[0], imgC.shape[0])
    w = imgA.shape[1] + imgB.shape[1] + imgC.shape[1]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:imgA.shape[0], :imgA.shape[1]] = imgA
    out[:imgB.shape[0], imgA.shape[1]:imgA.shape[1]+imgB.shape[1]] = imgB
    out[:imgC.shape[0], imgA.shape[1]+imgB.shape[1]:] = imgC
    return out

def process_image(img_path, model, sampler, N, steps, device, resize_to):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to read {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_image(img, resize_to)
    control = torch.from_numpy(img.copy()).float().to(device) / 255.0
    control = torch.stack([control for _ in range(N)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    c_cat = control.to(device)
    # 强制把 conditioning 张量移动到同一 GPU
    c = model.get_unconditional_conditioning(N).to(device)
    uc_cross = model.get_unconditional_conditioning(N).to(device)
    uc_cat = c_cat
    uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
    cond = {"c_concat": [c_cat], "c_crossattn": [c]}
    b, c_ch, h, w = cond["c_concat"][0].shape
    shape = (4, h // 8, w // 8)
    samples, _ = sampler.sample(steps, N, shape, cond, verbose=False, eta=0.0,
                               unconditional_guidance_scale=9.0,
                               unconditional_conditioning=uc_full)
    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    return x_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='模型配置文件路径')
    parser.add_argument('--ckpt', type=str, required=True, help='权重文件路径')
    parser.add_argument('--test_dir', type=str, required=True, help='测试集图片文件夹')
    parser.add_argument('--target_dir', type=str, required=True, help='目标图像文件夹')
    parser.add_argument('--outdir', type=str, default='./results', help='输出根目录')
    parser.add_argument('--device', type=str, default='cuda', help='cuda 或 cpu')
    parser.add_argument('--ddim_steps', type=int, default=50, help='采样步数')
    parser.add_argument('--resize', type=int, default=256, help='图片缩放尺寸')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and device.index is not None:
        torch.cuda.set_device(device.index)  # 保证默认张量创建落在目标卡
    model = create_model(args.config)
    model.load_state_dict(load_state_dict(args.ckpt, location='cpu'))
    model.to(device)
    model.eval()
    # 可选调试：确认 cond_stage_model 在同一设备
    # print("[DEBUG] cond_stage_model device:", next(model.cond_stage_model.parameters()).device)
    sampler = DDIMSampler(model)

    out_samples = Path(args.outdir) / 'samples'
    out_compare = Path(args.outdir) / 'compare'
    out_samples.mkdir(parents=True, exist_ok=True)
    out_compare.mkdir(parents=True, exist_ok=True)

    test_files = sorted([f for f in Path(args.test_dir).iterdir() if f.is_file()])
    print(f"Found {len(test_files)} test images.")

    for img_path in tqdm(test_files):
        gen_imgs = process_image(img_path, model, sampler, N=1, steps=args.ddim_steps, device=device, resize_to=args.resize)
        if gen_imgs is None:
            continue
        gen_img = gen_imgs[0]
        # 保存采样结果
        save_image(gen_img, out_samples / img_path.name)
        # 读取输入图像（testA）
        input_img = cv2.imread(str(img_path))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = resize_image(input_img, args.resize)
        # 读取目标图像（testB）
        target_path = Path(args.target_dir) / img_path.name
        if target_path.exists():
            tgt_img = cv2.imread(str(target_path))
            tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
            tgt_img = resize_image(tgt_img, args.resize)
            compare_img = concat_three(input_img, gen_img, tgt_img)
            save_image(compare_img, out_compare / img_path.name)
        else:
            compare_img = concat_three(input_img, gen_img, gen_img)
            save_image(compare_img, out_compare / img_path.name)

    print(f"采样结果已保存到: {out_samples}")
    print(f"对比结果已保存到: {out_compare}")

if __name__ == '__main__':
    import sys
    # 如果没有命令行参数，则用默认参数
    if len(sys.argv) == 1:
        class Args:
            config = './models/cldm_v15.yaml'
            ckpt = '/NAS_data/hjf/ControlNet-main/checkpoints/GF3/epoch_epoch=399-step_step=351999.ckpt'
            test_dir = '/NAS_data/yjy/GF3_High_Res/testA'
            target_dir = '/NAS_data/yjy/GF3_High_Res/testB'
            prompt = 'an image'
            outdir = '/NAS_data/hjf/ControlNet-main/outputs/GF3'
            device = 'cuda'
            ddim_steps = 50
            resize = 256
        # 构造参数列表
        sys.argv += [
            '--config', Args.config,
            '--ckpt', Args.ckpt,
            '--test_dir', Args.test_dir,
            '--target_dir', Args.target_dir,
            '--outdir', Args.outdir,
            '--device', Args.device,
            '--ddim_steps', str(Args.ddim_steps),
            '--resize', str(Args.resize)
        ]
    main()
