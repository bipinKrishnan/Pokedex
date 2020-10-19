import torch
from PIL import Image
from torchvision.transforms import transforms
import pandas as pd
from gtts import gTTS
import random
from targets import target

from bs4 import BeautifulSoup
import requests
import re
import streamlit as st

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                          (0.5, 0.5, 0.5))
])

def create_image(img):
    pil_img = Image.open(img).convert('RGB')   
    img = transform(pil_img)    

    return pil_img, img

def load_df(path1, path2):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    return df1, df2

def make_pred(model, img):
    out = model(img.unsqueeze(0))
    pred = target[torch.max(out, 1)[1].item()]
    
    return pred
    
def get_pokemon_details(pred, df, df_):
    cat = df[df['species']==pred]
    species = cat['species'][cat.index[0]]
    type1 = cat['type1'][cat.index[0]]
    class1 = cat['class'][cat.index[0]]
        
    moves = df_[df_['type']==type1]
        
    rand = random.randrange(0, len(moves))
    move = moves.iloc[rand]['move']
    move_desc = moves.iloc[rand]['description']

    return type1, class1, move, move_desc

def get_audio(text):
    tts = gTTS(text[:-3], lang='en-gb')
    tts.save('hello.ogg')
    audio = open('hello.ogg', 'rb')
    audio_bytes = audio.read()

    return audio_bytes

def get_image(name):
    name = name.replace(' ', '')
    u = f'https://www.google.co.in/search?q={name}+gif&source=lnms&tbm=isch'
        #u = ''.join(f'https://tenor.com/search/{name}-gifs')
        #html = urllib.request.urlopen(u)
        #soup = BeautifulSoup(html, 'html.parser')
        #url = soup.find_all('img')[2].get('src')
    #url = f'https://img.pokemondb.net/artwork/{name}.jpg'
    st.text(u)
    html = requests.get(u).content
    soup = BeautifulSoup(html, 'html.parser')
    st.text(soup.find_all('img', class_='n3VNCb'))
    url = soup.find_all('img', class_='n3VNCb').get('src')
    
        
    return url
