import torch
import os
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from models.models import DGModel_final
from utils.misc import denormalize, divide_img_into_patches, get_padding
from configs.get_config import get_inference_config

class MPCountInference:
    def __init__(self):
        self.model_path = get_inference_config()["model_path"]
        self.device = get_inference_config()["device"]
        self.save_path = get_inference_config()["save_path"]
        self.vis_dir = get_inference_config()["vis_dir"]
        self.unit_size = get_inference_config()["unit_size"]
        self.patch_size = get_inference_config()["patch_size"]
        self.log_para = get_inference_config()["log_para"]
        self.unit_size = get_inference_config()["unit_size"]
        self.vis_dir = get_inference_config()["vis_dir"]

        os.makedirs(self.vis_dir, exist_ok=True)

        self.model = self.load_model()

    def load_model(self):
        model = DGModel_final().to(self.device)
        model.load_state_dict(torch.load(self.model_path, 
                                         map_location=self.device), 
                              strict=False)
        model.eval()

        return model

    def preprocess(self, image_path):
        img = Image.open(image_path).convert('RGB')
        
        if self.unit_size > 0:
            w, h = img.size
            new_w = (w // self.unit_size + 1) * self.unit_size if w % self.unit_size != 0 else w
            new_h = (h // self.unit_size + 1) * self.unit_size if h % self.unit_size != 0 else h

            padding, h, w = get_padding(h, w, new_h, new_w)

            img = F.pad(img, padding)
        img = F.to_tensor(img)
        img = F.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        tensor = img.unsqueeze(0).to(self.device)
        
        return tensor

    def visualize(self, pred_dmap, tensor, img_name, pred_count):
        denormed_img = denormalize(tensor)[0].cpu().permute(1, 2, 0).numpy()
        fig = plt.figure(figsize=(10, 5))
        ax_img = fig.add_subplot(121)
        ax_img.imshow(denormed_img)
        ax_img.set_title(img_name)
        ax_dmap = fig.add_subplot(122)
        ax_dmap.imshow(pred_dmap)
        ax_dmap.set_title(f'Predicted count: {pred_count}')
        plt.savefig(os.path.join(self.vis_dir, img_name.split('.')[0] + '.png'))
        plt.show()
        plt.close(fig)
    
    @torch.no_grad()
    def inference(self, image_path):
        tensor = self.preprocess(image_path)
        h, w = tensor.shape[2:]
        ps = self.patch_size
        if h >= ps or w >= ps:
            pred_dmap = torch.zeros(1, 1, h, w)
            pred_count = 0
            img_patches, nh, nw = divide_img_into_patches(tensor, ps)
            for i in range(nh):
                for j in range(nw):
                    patch = img_patches[i*nw+j]
                    pred_dpatch = self.model(patch)[0]
                    pred_dmap[:, :, i*ps:(i+1)*ps, j*ps:(j+1)*ps] = pred_dpatch
        else:
            pred_dmap = self.model(tensor)[0]
        pred_map_np = pred_dmap.squeeze().cpu().numpy()
        pred_count = pred_dmap.sum().cpu().item() / self.log_para
        self.visualize(pred_map_np, 
                       tensor, image_path.split('/')[-1], 
                       pred_count)
        return pred_map_np, pred_count

if __name__ == "__main__":
    inference = MPCountInference()
    pred_dmap, pred_count = inference.inference("D:\\city-view\\yolo_crowd\\data\\Crowd_in_street.jpg")
    print(pred_count)
    