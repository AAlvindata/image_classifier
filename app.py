import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import time

# --- PATH SETTING ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"

# --- GENERAL SETTINGS ---
PAGE_TITLE = "Imange Classfier (VGG19)"
st.set_page_config(page_title=PAGE_TITLE)

# --- LOAD CSS & SAMPLE DATASET FOR DOWNLOAD ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

# load model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('vgg_19.h5')
    return model
with st.spinner('Model is being loaded..'):
    model = load_model()

# Main page
st.title("Image Classification")

file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)


# In[ ]:


def upload_predict(upload_image, model):
    
    size = (180,180)    
    image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
        
    img_reshape = img_resize[np.newaxis,...]
    
    prediction = model.predict(img_reshape)
    pred_class=decode_predictions(prediction,top=1)
        
    return pred_class


# In[ ]:


if file is None:
    st.info("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = upload_predict(image, model)
    image_class = str(predictions[0][0][1])
    score=np.round(predictions[0][0][2])
    st.info("please wait...")
    time.sleep(0.5)
    st.info(f"Done! I guess this is {image_class}!")

