import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

classes = {0:"Alluvial",1:"Black",2:"Clay",3:"Red"}

suggestions = {"Alluvial": "Rice, Wheat, Sugarcane, Maize, Cotton, Soyabean, Jute", "Black" : "Wheat,Jowar, Millets, Linseed, Castor, Sunflower",
"Clay" : "Rice, Lettuce, Chard, Broccoli, Cabbage, Beans", "Red" : "Cotton, Wheat, Pulses, Millets, OilSeeds, Potatoes"}


def main():

    page = st.sidebar.selectbox("App Selections", ["Homepage", "About", "Identify", "Contact Us"])
    if page == "Identify":
        st.title("Soil Identifier")
        identify()

def identify():

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
        datapath = "snapshot/"
        image.save(datapath+name)
        #st.write("Just a second...")
        #temp_file.write(uploaded_file.getvalue())
        #data_x = uploaded_file.read()
        #data_x
        #st.write(load_img(temp_file.name))

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
