import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
import yaml
import pickle

from os import listdir
from PIL import Image


def predict_image(path, model):
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction
    
    Returns
    -------
    Predicted class
    """
    images = np.array([np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))])
    print(images.shape)
    prediction_vector = model.predict(images)
    predicted_classes = np.argmax(prediction_vector, axis=1)
    return predicted_classes[0]
   
    
with open('parameters.yaml') as file:
    P = yaml.load(file, Loader=yaml.FullLoader)

DATA_DIR = pathlib.Path(P['DIR']['DATA_DIR'])
MODELS_DIR = pathlib.Path(P['DIR']['MODELS_DIR'])
IMAGE_WIDTH = P['IMAGE_DIM']['IMAGE_WIDTH']
IMAGE_HEIGHT = P['IMAGE_DIM']['IMAGE_HEIGHT']
IMAGE_DEPTH = P['IMAGE_DIM']['IMAGE_DEPTH']


#list_files contains all possible model (either .h5 or .pkl format)
list_files = [i for i in listdir("models") if (i.endswith('.h5') | i.endswith('.pkl'))]

# First widget : choose the target variable
list_names = list(set([i.split('_')[1] for i in list_files]))
name = st.sidebar.radio(
    "Souhaitez-vous prédire le constructeur ou la famille ?",
    list_names
)

# Second widget : select which model to use
list_models = [x.split('_')[0] for x in list_files if name in x]
model = st.sidebar.radio(
    "Quel algorithme souhaitez-vous utiliser ?",
    list_models
)


# Import labels
with open('labels/labels_' + name + '.yaml') as file:
    labels = yaml.load(file, Loader=yaml.FullLoader)

# Import model
if model == "SVM":
    m = pickle.load(open(f'{MODELS_DIR}/{model}_{name}_.pkl', 'rb'))
else:
    m = tf.keras.models.load_model(f'{MODELS_DIR}/{model}_{name}_.h5')

    
st.title("Identification d'avion")

uploaded_file = st.file_uploader("Chargez une image d'avion") #, accept_multiple_files=True)#

if uploaded_file:
    loaded_image = plt.imread(uploaded_file)
    st.image(loaded_image)
  
predict_btn = st.checkbox("Identifier", disabled=(uploaded_file is None))

if predict_btn:
    st.balloons()
    images = np.array([np.array(Image.open(uploaded_file).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))])
    if model == "SVM":
        images = images.reshape(len(images), -1)
        probas = m.predict_proba(images).reshape(len(labels),)
    else:
        probas = m.predict(images).reshape(len(labels),) 
    classe = np.argmax(probas)
    proba = probas[classe]
    label = labels[classe]
    st.write(f"Numéro de classe prédite : {classe}, il s'agit d'un {label} ! La probabilité est de {round(100*proba, 2)}%")
    m1, m2, m3 = st.columns(3)
    m1.metric("Classe", classe)
    m2.metric("Label", label)
    m3.metric("Probabilité", str(round(100*proba, 2)) + "%")
    display_proba_graph = st.button('Afficher le graphique des probabilités')
    if predict_btn & display_proba_graph:
        fig = plt.figure(figsize=(10, round(len(labels)/3)))
        ax = sns.barplot(y=labels, x=probas, palette="Blues_d")
        ax.bar_label(ax.containers[0], np.char.add(np.round(probas*100, 2).astype('str'), "%"))
        ax.set(xlim=(0, 1))
        ax.set_title("Distribution des probabilités")
        st.pyplot(fig)
        
    
    