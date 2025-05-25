import torch
import torch.nn as nn
import torchvision.transforms as transforms
from . import clip
# import clip
from .loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from .loralib import layers as lora_layers


def build_lora(encoder='both', 
               position='all', 
               params=['q', 'k', 'v'], 
               r=2, 
               alpha=1, 
               dropout_rate=0, 
               backbone = 'ViT-L/14'
        
):
    clip_model, preprocess = clip.load(backbone, device='cpu')
    clip_model.eval()
    list_lora_layers = apply_lora(encoder, position, params, r, alpha, dropout_rate, backbone, clip_model)
    mark_only_lora_as_trainable(clip_model)
    return clip_model


class ClipLora(torch.nn.Module):
    def __init__(self, clip_model):
        super(ClipLora, self).__init__()
        self.image_encoder = clip_model.visual
        self.fc1 = nn.Linear(768, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1)


    def forward(self, x):
        image_features = self.image_encoder(x)
        image_features = self.fc1(image_features)
        image_features = self.relu(image_features)
        scores = self.fc2(image_features)
        return scores
    

def build_clip_lora(encoder='both', 
               position='all', 
               params=['q', 'k', 'v'], 
               r=2, 
               alpha=1, 
               dropout_rate=0, 
               backbone = 'ViT-B/16'
        
):
    clip_model = build_lora(encoder, position, params, r, alpha, dropout_rate, backbone)
    model = ClipLora(clip_model)
    return model

if __name__ == "__main__":
    # Example usage
    model = build_clip_lora(backbone = 'ViT-L/14')
    # Now you can use clip_model for inference or training
    # model = ClipLora(clip_model)
    model.cuda()
    x = torch.rand(4, 3, 224, 224).cuda()
    
    text = torch.randint(0, 100, (4, 77), dtype=torch.long).cuda()
    # y1 = model(x)
    # print(y1.shape)
    have_gread = []
    for name, param in model.named_parameters():
        print(name)
        # if param.requires_grad==True:
        #     print(name)
        #     have_gread.append(name)
    print(len(have_gread))
    # # print(y2.shape)
    # from torchinfo import summary

    # summary(model, input_data=(x))