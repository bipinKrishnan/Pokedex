# Pokédex  

#![Alt Text](media/pikachu.gif)
<img src="media/pikachu.gif" width="40" height="40" />

This is a web app version similar to Pokédex(a device for getting details about a pokemon found in the Pokémon cartoon series), built and deployed with [streamlit](https://www.streamlit.io/).

You can upload pokemon images and play with the app or interactively play with code by lauching the notebook.

[![app](https://img.shields.io/badge/launch-app-brightgreen)](https://share.streamlit.io/bipinkrishnan/pokemon/main)  [![notebook](https://img.shields.io/badge/launch-notebook-blue)](https://www.kaggle.com/bipinkrishnan/pokemon-image-classification-with-efficientnet)

#### I. Running the app locally
1. Clone the repo by running the below command and change directory to the cloned repo

       git clone https://github.com/bipinKrishnan/pokemon.git
  
1. Make sure that you have installed all the libraries specified in the [requirements.txt](https://github.com/bipinKrishnan/pokemon/blob/main/requirements.txt) file by running the following command

       pip install -r requirements.txt

2. Now run the following command to launch the app

       streamlit run streamlit_app.py 
       
#### II. Code structure
* [model.py](https://github.com/bipinKrishnan/pokemon/blob/main/model.py) - This file contains the code to make changes to the EfficientNet model to suit our problem set.

* [helper_func.py](https://github.com/bipinKrishnan/pokemon/blob/main/helper_func.py) - This file contains helper functions to load and preprocess the image, to make predictions, get details about the predicted pokemon and so on.

* [streamlit_app.py](https://github.com/bipinKrishnan/pokemon/blob/main/streamlit_app.py) - This file contains all the details on how to display the front-end of the streamlit app.

* [targets.py](https://github.com/bipinKrishnan/pokemon/blob/main/targets.py) - This file contains all the 166 pokemon names stored in a variable called target in the form of a python list.

#### III. Pretrained weights
* You can find the pretrained weights for the model [here](https://github.com/bipinKrishnan/pokemon/tree/main/utils) or directly download the weights.

  [![app](https://img.shields.io/badge/download-weights-orange)](https://github.com/bipinKrishnan/pokemon/raw/main/utils/model.pt)




