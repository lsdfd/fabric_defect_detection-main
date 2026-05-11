import streamlit as st

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT))

from utils import bytes_to_np, classify_image, get_defect_segment

st.set_page_config(layout="wide")

# image ingestion
image_input = st.file_uploader("Upload a fabric image to initiate defect detection:", type="png")
if image_input is not None:
    img = bytes_to_np(image_input)

    st.image(img)

    with st.spinner('Processing image...'):
        result, certainty = classify_image(img)
    if sum(result):
        st.write("Defect detected with {:.4f}% certainty. Obtaining segmentation mask.".format(certainty * 100))
        with st.spinner('Obtaining segmentation...'):
            segmented = get_defect_segment(img, result)
            st.image(segmented, channels="RGB")
    else:
        st.write("No defects detected!")
