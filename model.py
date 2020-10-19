import torch
from torch import nn
from efficientnet_pytorch import EfficientNet

def create_model():
    model = EfficientNet.from_name("efficientnet-b0")
    model._fc = nn.Linear(1280, len(targets.target))

    model.load_state_dict(torch.load('utils/model.pt', map_location=torch.device('cpu')))
    model.eval()

    return model
