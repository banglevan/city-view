import cv2
import numpy as np
import torch
from transformers import CLIPTokenizer
import torchvision.transforms.functional as F
import random
from PIL import Image
from torchvision import transforms
from models.reg_model import Count
from utils.tools import extract_patches, reassemble_patches

class T2ICountInference:
    def __init__(self, model_path='weights/best_model_paper.pth', crop_size=384):
        self.model_path = model_path
        self.crop_size = crop_size
        self.down_ratio = 8

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.concat_size = 224
        self.model = self.load_model()

    def load_model(self):
        unet_config = {'base_size': 384,
                   'max_attn_size': 384 // 8,
                   'attn_selector': 'down_cross+up_cross'}
        model = Count('configs/v1-inference.yaml',
                  'configs/v1-5-pruned-emaonly.ckpt', unet_config=unet_config)
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.device = torch.device('cuda')
        model = model.to(self.device)
        model.set_eval()
        return model

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        
        # Calculate new width to maintain aspect ratio with height = 384
        original_width, original_height = image.size
        target_width = 384
        target_height = int(original_height * target_width / original_width)
        
        # Resize image to target dimensions
        image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)        
        image = self.transform(image)
        inputs = image.to(self.device)
        return inputs

    def preprocess_prompt(self, prompt):
        prompt_attn_mask = torch.zeros(77)
        cls_name_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        cls_name_length = cls_name_tokens['input_ids'].shape[1]
        prompt_attn_mask[1: 1 + cls_name_length] = 1
        return prompt_attn_mask

    def postprocess(self, density_map):
        '''
        Calculate the number of objects in the image
        By using the method of count the peak of the density map
        Args:
            density_map: numpy array of shape (height, width)
        Returns:
            number of objects in the image
        '''
        pass

    def inference(self, image_path, prompt):
        batch_size = 16
        inputs = self.preprocess_image(image_path).unsqueeze(0)
        prompt_attn_mask = self.preprocess_prompt(prompt).unsqueeze(0)
        prompts = (prompt,)
        gt_prompt_attn_mask = prompt_attn_mask.to(self.device).unsqueeze(2).unsqueeze(3)
        cropped_imgs, num_h, num_w = extract_patches(inputs, patch_size=self.crop_size, stride=self.crop_size)
        outputs = []
        with torch.set_grad_enabled(False):
            num_chunks = (cropped_imgs.size(0) + batch_size - 1) // batch_size
            for i in range(num_chunks):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, cropped_imgs.size(0))
                outputs_partial = self.model(cropped_imgs[start_idx:end_idx], 
                                             prompts * (end_idx - start_idx), 
                                             gt_prompt_attn_mask.repeat((end_idx - start_idx), 1, 1, 1))[0]
                outputs.append(outputs_partial)
            results = reassemble_patches(torch.cat(outputs, dim=0), num_h, num_w, inputs.size(2), inputs.size(3),
                                         patch_size=self.crop_size, stride=self.crop_size) / 60
            results = results.squeeze(0).squeeze(0).detach().cpu().numpy()
            return results

if __name__ == "__main__":
    t2i_inference = T2ICountInference()
    t2i_inference.inference("D:/city-view/yolo_crowd/data/ex1.jpg", "person")