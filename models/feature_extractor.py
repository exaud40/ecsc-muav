import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


# CLIP Feature extractor for LandUse
class CLIPFeatureExtractor_LU(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", 
            use_safetensors=True).to(device)

        for p in self.model.parameters():
            p.requires_grad = False

        self.input_resolution = 224
        self.output_dim = self.model.config.projection_dim
        self.device = device

    def forward(self, x):

        x = F.interpolate(x,
                          size=(self.input_resolution, self.input_resolution),
                          mode='bilinear',
                          align_corners=False)

        x = x * 2 - 1

        with torch.no_grad():
            vision_outputs = self.model.vision_model(x)
            features = self.model.visual_projection(vision_outputs[1])

        return features
    
    
    
# CLIP Feature extractor for RSSCN7
class CLIPFeatureExtractor_RS(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_safetensors=True).to(device)

        for p in self.model.parameters():
            p.requires_grad = False
            
            self.input_resolution = 224

    def forward(self, x):
        x = F.interpolate(x, 
                          size=(self.input_resolution, self.input_resolution), 
                          mode='bilinear', 
                          align_corners=False)
        x = x * 2 - 1

        with torch.no_grad():
            v = self.model.vision_model(x)
            return self.model.visual_projection(v[1])   
    