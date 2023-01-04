import pandas as pd
import numpy as np
from preprocessing.preprocessing import utils
from preprocessing.preprocessing.embeddings import embed
import os
import unittest
from preprocessing.preprocessing.utils import _SimpleSequence
from predict.predict.run import TextPredictionModel
import streamlit as st

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.realpath(__file__))


artefacts_path = os.path.abspath(os.path.join(current_dir, 'train', 'data', 'artefacts'))
print(artefacts_path)


def main():
    st.title("Language Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"></h2>
    </div>
    """
    model = TextPredictionModel.from_artefacts(artefacts_path)

    st.markdown(html_temp, unsafe_allow_html=True)
    text = st.text_input("Text to Predict", "Type Here")
    result = ""
    st.text(text)
    if st.button("Predict"):
        result = model.predict([text])
    st.success('The label of the given text  {}'.format(result))
    if st.button("About"):
        st.text("Predicting the Language of a given stackoverflow request using text classifier")
        st.text("API built with Streamlit")


#
if __name__ == '__main__':

    main()




# if __name__ == "__main__":
#     # Get the absolute path of the current directory
#     current_dir = os.path.dirname(os.path.realpath(__file__))
#
#
#     artefacts_path = os.path.abspath(os.path.join(current_dir, 'train', 'data', 'artefacts'))
#     print(artefacts_path)
#     model = TextPredictionModel.from_artefacts(artefacts_path)
#
#     text = "Is it possible to execute the procedure of a function in the scope of the caller?"
#     result = model.predict([text])
#     print(result)
#


