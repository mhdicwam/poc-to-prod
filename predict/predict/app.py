import streamlit as st
from run import *
import os

import os

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.realpath(__file__))
print(current_dir)
# Get the absolute path of the parent directory
parent_dir = os.path.abspath( os.pardir)
print(parent_dir)

artefacts_path = os.path.abspath(os.path.join('../..', 'train', 'data','artefacts'))
print(artefacts_path)
def main():
    st.title("Language Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"></h2>
    </div>
    """
    model = TextPredictionModel.from_artefacts(artefacts_path)

    st.markdown(html_temp,unsafe_allow_html=True)
    text=st.text_input("Text to Predict","Type Here")
    result=""
    if st.button("Predict"):
        result=model.predict([text])
    st.success('The label of the given text  {}'.format(result))
    if st.button("About"):
        st.text("Predicting the Language of a given stackover flow request using classifier")
        st.text("API built with Streamlit")

#
# if __name__== '__main__':
#     main()