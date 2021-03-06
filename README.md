# Pokédex  

<img src="media/pikachu.gif"/>

This is a web app version similar to Pokédex(a device for getting details about a pokemon, found in the Pokémon cartoon series), built and deployed with [streamlit](https://www.streamlit.io/).

You can upload pokemon images by launching the app or interactively play with code for training the model by lauching the notebook.

[![app](https://img.shields.io/badge/launch-app-brightgreen)](https://share.streamlit.io/bipinkrishnan/pokedex/main)  [![notebook](https://img.shields.io/badge/launch-notebook-blue)](https://www.kaggle.com/bipinkrishnan/pokemon-image-classification-with-efficientnet)

#### I. Running the app locally
1. Clone the repo by running the below command and change directory to the cloned repo

       git clone https://github.com/bipinKrishnan/Pokedex.git
  
1. Make sure that you have installed all the libraries specified in the [requirements.txt](https://github.com/bipinKrishnan/Pokedex/blob/main/requirements.txt) file by running the following command

       pip install -r requirements.txt

2. Now run the following command to launch the app

       streamlit run streamlit_app.py 
       
#### II. Code structure
* [model.py](https://github.com/bipinKrishnan/Pokedex/blob/main/model.py) - This file contains the code to change the EfficientNet model architecture to suit our problem set.

* [helper_func.py](https://github.com/bipinKrishnan/Pokedex/blob/main/helper_func.py) - This file contains helper functions to load and preprocess the image, to make predictions, get details about the predicted pokemon and so on.

* [streamlit_app.py](https://github.com/bipinKrishnan/Pokedex/blob/main/streamlit_app.py) - This file contains all the details on how to display the front-end of the streamlit app.

* [targets.py](https://github.com/bipinKrishnan/Pokedex/blob/main/targets.py) - This file contains all the 166 pokemon names stored in a variable called target in the form of a python list.

#### III. Pretrained weights
* You can find the pretrained weights for the model [here](https://github.com/bipinKrishnan/Pokedex/releases) or directly download the weights.

  [![app](https://img.shields.io/badge/download-weights-orange)](https://github.com/bipinKrishnan/Pokedex/releases/download/0.0.1/model.pt)
