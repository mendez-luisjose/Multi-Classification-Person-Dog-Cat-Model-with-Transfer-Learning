import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from skimage.transform import resize
import tensorflow_hub as hub

#Put here the Model Path
MODEL_PATH = ''

width_shape = 224
height_shape = 224

header = st.container()
body = st.container()

def model_prediction(img, model):

    img_resize = resize(img, (width_shape, height_shape))
    X=tf.keras.applications.imagenet_utils.preprocess_input(img_resize*255)
    X = np.expand_dims(X,axis=0)

    X = np.vstack([X])
    
    preds = model.predict(X)

    arg_max_result = np.argmax(preds)
    percentage = preds[0][arg_max_result] * 100

    if arg_max_result == 0 :
        return "Cat " + "%.2f" % percentage + "%"
    elif arg_max_result == 1 :
        return "Dog " + "%.2f" % percentage + "%"
    elif arg_max_result == 2 :
        return "Person " + "%.2f" % percentage + "%"
    
with header :
    st.title("Person 🧑🏻‍🦱 - Dog 🐶 - Cat Model 🐈")
    st.header("Multi Classification Person-Dog-Cat Model with Transfer Learning")
    st.image("./img.jpg")

with body :
    st.subheader("Check It-out!")

    model=''

    if model=='':
        model = tf.keras.models.load_model(
            ("./human_dog_cat_model.h5"),
            custom_objects={'KerasLayer':hub.KerasLayer}
        )

    img = st.file_uploader("Upload an Image: ", type=["png", "jpg", "jpeg"])

    col1, col2, col3 = st.columns([0.3,1,0.2])

    col4, col5, col6 = st.columns([0.9,1,0.2])

    
    if img is not None:
        image = np.array(Image.open(img))    
        col2.image(image, width=400)
    

    if col5.button("Predict"):
         prediction = model_prediction(image, model)
         st.success("The Given Image is a "  +  prediction + "!")    




