import streamlit  as st
from PIL import Image
import os
from ultralytics import YOLO
import numpy as np

# *load the model
# @st.cache_resources
def load_model():
    return YOLO('best.pt')

model = load_model()

st.title("Furniture Detection App")
st.write("Upload image to detect furniture (coffee table, sofa, drawer)")

uploaded_file = st.file_uploader("Upload image" , type = ['jpg','jpeg','png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting..."):
        results = model.predict(source=np.array(image), conf=0.25)

    # Draw results
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Detected Objects", use_column_width=True)



