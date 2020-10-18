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
import random

st.title("Pokédex")
st.markdown("for generation one **Pokémon**")

df = pd.read_csv('utils/pokemon.csv')
df_ = pd.read_csv('utils/moves.csv')

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

if img:
    pil_img = Image.open(img).convert('RGB')

    img = transform(pil_img)

    out = model(img.unsqueeze(0))
    pred = targets.target[torch.max(out, 1)[1].item()]

    st.image(pil_img.resize((224, 224)))

    if pred in df['species'].values:
        cat = df[df['species']==pred]
        species = cat['species'][cat.index[0]]
        type1 = cat['type1'][cat.index[0]]
        class1 = cat['class'][cat.index[0]]
        
        moves = df_[df_['type']==type1]
        
        rand = random.randrange(0, len(moves))
        move = moves.iloc[rand]['move']
        move_desc = moves.iloc[rand]['description']

        text = f"-- This is {pred}, a {type1} type {class1}\n\n-- Since {pred} is a {type1} pokemon, it has a special move called {move}\n\n-- {move_desc}\n\n___"
        st.text(text)
          
        #text_audio = f"This is {pred}, a {type1} type {class1}\nSince {pred} is a {type1} pokemon, it has a special move called {move}\n{move_desc}"
        tts = gTTS(text, lang='en-gb')
        tts.save('hello.ogg')
        audio = open('hello.ogg', 'rb')
        audio_bytes = audio.read()
        st.audio(audio_bytes, format='audio/ogg', start_time=0)
        #display(Audio('hello.mp3', autoplay=True))
    
        os.remove('hello.ogg')
            
    else:
        st.text(f"This is {pred}\nCurrently, there is not much details about {pred} in my database!!!")
