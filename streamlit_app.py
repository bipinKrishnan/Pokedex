import numpy as np
import pandas as pd

import torch
from torch import nn
from torchvision.transforms import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image

from utils import targets
import streamlit as st
from gtts import gTTS
import os

df = pd.read_csv('utils/df.csv')

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

img = st.file_uploader("Upload Image", type=['jpeg', 'jpg', 'png', 'webp'])
pil_img = Image.open(img).convert('RGB')

img = transform(pil_img)

out = model(img.unsqueeze(0))
pred = targets.target[torch.max(out, 1)[1].item()]

st.image(pil_img.resize((224, 224)), caption=pred)

if pred in df[' Name'].values:
    values = df[df[' Name']==pred]
    #st.text(f"This is {pred}, a {values[' Type1'][values[' Type1'].index[0]]} type pokemon")
    text = f"This is {pred}, a {values[' Type1'][values[' Type1'].index[0]]} type pokemon"
            
    tts = gTTS(text, lang='en-au')
    tts.save('hello.ogg')
    audio = open('hello.ogg', 'rb')
    audio_bytes = audio.read()
    st.audio(audio_bytes, format='audio/ogg', start_time=0)
    #display(Audio('hello.mp3', autoplay=True))
    
    os.remove('hello.ogg')
            
else:
    st.text(f"This is {pred}")
#st.text(pred)

