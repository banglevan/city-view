import torch
import torch.nn.functional as F

def extract_patches(img, patch_size=512, stride=512):
    _, _, h, w = img.size()
    num_h = (h - patch_size + stride - 1) // stride + 1
    num_w = (w - patch_size + stride - 1) // stride + 1
    patches = []
    for i in range(num_h):
        for j in range(num_w):
            y_start = min(i * stride, h - patch_size)
            x_start = min(j * stride, w - patch_size)
            patch = img[:, :, y_start:y_start + patch_size, x_start:x_start + patch_size]
            patches.append(patch)
    patches = torch.cat(patches, dim=0)
    return patches, num_h, num_w


def reassemble_patches(patches, num_h, num_w, h, w, patch_size=512, stride=256):
    result = torch.zeros(1, patches.size(1), h, w).to(patches.device)
    norm_map = torch.zeros(1, 1, h, w).to(patches.device)
    patches = F.interpolate(patches, scale_factor=8, mode='bilinear') / 64
    patch_idx = 0
    for i in range(num_h):
        for j in range(num_w):
            y_start = min(i * stride, h - patch_size)
            x_start = min(j * stride, w - patch_size)
            result[:, :, y_start:y_start + patch_size, x_start:x_start + patch_size] += patches[patch_idx]
            norm_map[:, :, y_start:y_start + patch_size, x_start:x_start + patch_size] += 1
            patch_idx += 1

    result /= norm_map
    return result
