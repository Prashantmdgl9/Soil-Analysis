import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import base64

classes = {0:"Alluvial",1:"Black",2:"Clay",3:"Red"}

suggestions = {"Alluvial": "Tomatoes, Sage, Roses, Butterfly bush, Ferns, Daffodils, Lavender", "Black" : "Citrus fruits, Sunflower, Legumes, Microgreens, Peppers",
"Clay" : "Kale, Lettuce, Broccoli, Cabbage, Aster, Daylily, Magnolia, Juniper, Pine, Geranium, Ivy", "Red" : "Peanuts, Grams, Potatoes, Sweet potato, Banana, Papaya"}



healthType = ['Scab',
 'Rot',
 'Rust',
 'Healthy',
 'Healthy',
 'Powdery mildew',
 'Healthy',
 'Leaf spot',
 'Common_rust',
 'Northern Leaf Blight',
 'Healthy',
 'Black rot',
 'Black Measles',
 'Leaf blight',
 'Healthy',
 'Citrus greening',
 'Bacterial spot',
 'Healthy',
 'Bacterial spot',
 'Healthy',
 'Early blight',
 'Late blight',
 'Healthy',
 'Healthy',
 'Healthy',
 'Powdery mildew',
 'Leaf_scorch',
 'healthy',
 'Bacterial spot',
 'Early blight',
 'Late blight',
 'Leaf Mold',
 'Leaf spot',
 'Spider mite',
 'Target Spot',
 'Yellow Leaf',
 'Mosaic virus',
 'Healthy']


datapath = "snapshot/"


def main():

    page = st.sidebar.selectbox("App Selections", ["Homepage", "About", "Identify", "Plant_Health"])
    if page == "Identify":
        st.title("Soil Identifier")
        identify()
    elif page == "Homepage":
        homepage()
    elif page == "About":
        about()
    elif page == "Plant_Health":
        health()



def health():
    st.title("Check the health of your plant")
    set_png_as_page_bg(datapath+'identify3.jpg')
    leaf_model = load_model('models/leaf-model.h5')
    st.set_option('deprecation.showfileUploaderEncoding', True)
    st.subheader("Choose an image of a leaf that you want to check, please take photograph of only single leaf")
    uploaded_file = st.file_uploader("Upload an image", type = "jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        st.write("")
        name = "temp1.jpg"

        image.save(datapath+name)

        result = model_predict(datapath+name, leaf_model)
        pred = healthType[result]
        st.header("The state of your leaf is - "+ pred )


def homepage():
    html_temp = """
    <html>
    <head>
    <style>
    body {
      background-color: #fe2631;
    }
    </style>
    </head>
    <body>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    image = Image.open(datapath+'home6.png')
    st.image(image, use_column_width = True)


def about():
    set_png_as_page_bg(datapath+'mud4.jpg')
    st.title("A fistful of soil")
    st.header("“And somewhere there are engineers"
    " Helping others fly faster than sound."
    " But, where are the engineers"
    " helping those who must live on the ground?“")
    st.header("      "+ "                   - A Young Oxfam Poster")

    st.subheader("This is a preliminary work to classify soils based on the images that are uploaded by the user. A Convolutional Neural Network has been trained on sample images to identify"
     " the types of soil. Such work can find application in remote sensing and automatic classification of the land areas based on soil type.")
    st.subheader("Presently, the soils are classified into 4 categories viz. Alluvial, Black, Red, or Clay. Based on the classification of the soil, suggestions are made on the type of crops"
    " that can be grown there. ")

    st.subheader("This is version 1 of the product, there will be further improvements.")

#@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: 2200px;
    background-repeat: no-repeat;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)


def identify():
    set_png_as_page_bg(datapath+'identify2.jpg')
    soil_model = load_model('models/soil_model2.h5')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.subheader("Choose a soil image file that you extracted from the work site or field")
    uploaded_file = st.file_uploader("Upload an image", type = "jpg")
    #temp_file = NamedTemporaryFile(delete = False)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        st.write("")
        name = "temp.jpg"

        image.save(datapath+name)

        result = model_predict(datapath+name ,soil_model)
        pred = classes[result]
        st.header("The soil is of "+ pred + " type")
        st.subheader("The types of crops suggested for "+ pred + " soil are: "+ suggestions[pred])



def model_predict(image_path,model):

    image = load_img(image_path,target_size=(224,224))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)

    result = np.argmax(model.predict(image))
    return result


if __name__ == '__main__':
    main()
