import numpy as np

import torch
from torch import nn
from torchvision.transforms import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image

from utils import targets
import streamlit as st

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                          (0.5, 0.5, 0.5))
])

model = EfficientNet.from_name("efficientnet-b0")
model._fc = nn.Linear(1280, len(targets.target))

model.load_state_dict(torch.load('utils/model.pt', map_location=torch.device('cpu')))
model.eval()

img = st.file_uploader("Upload Image", type=['jpeg', 'jpg', 'png'])
img = Image.open(img).convert('RGB')

img = transform(img)

out = model(img.unsqueeze(0))
pred = targets.target[torch.max(out, 1)[1].item()]

st.image(img, caption=pred)
st.text(pred)

