import streamlit as st
import os
import torch

from model import create_model
from helper_func import *


st.title("Pokédex")
st.markdown("for generation one Pokémon")

df, df_ = load_df('utils/pokemon.csv', 'utils/moves.csv')

model = create_model()

img = st.file_uploader("Upload Image", type=['jpeg', 'jpg', 'png', 'webp'])

if img:
    images = create_image(img)
    pred = make_pred(model, images[1])

    st.image(images[0].resize((224, 224)))

    if pred in df['species'].values:
        details = get_pokemon_details(pred, df, df_)

        text = f"-- This is {pred}, a {details[0]} type {details[1]}\n\n-- Since {pred} is a {details[0]} pokemon, it has a special move called {details[2]}\n\n-- {details[3]}\n\n"
        st.text(text)

        audio_bytes = get_audio(text)
        st.audio(audio_bytes, format='audio/ogg', start_time=0)
        st.text('___')
    
        os.remove('hello.ogg')
            
    else:
        text = f"This is {pred}\nCurrently, there is not much details about {pred} in my database!!!"
        st.text(text)

        audio_bytes = get_audio(text)
        st.audio(audio_bytes, format='audio/ogg', start_time=0)

        os.remove('hello.ogg')
