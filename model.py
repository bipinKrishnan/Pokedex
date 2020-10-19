import torch
from torch import nn
from efficientnet_pytorch import EfficientNet

from targets import target

def create_model():
    model = EfficientNet.from_pretrained("efficientnet-b2")
    model._fc = nn.Linear(1408, len(target))

    model.load_state_dict(torch.load('utils/model-91.0374.pt', map_location=torch.device('cpu')))
    model.eval()

    return model

