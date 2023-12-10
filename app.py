import streamlit as st
import joblib
import pandas as pd
from src.data_pre_processing import preprocess_data


# Load the trained model (using Streamlit's cache mechanism)
def load_data():
    model = joblib.load('model.joblib')
    selector = joblib.load('feature_selector.joblib')
    processor = joblib.load('preprocessor.joblib')
    return model, selector, processor

model, selector, processor = load_data()

label_list = [
    'valid-question',
    'Duplicate',       
    'Needs more focus',              
    'Not suitable for this site',    
    'Needs details or clarity',      
    'Opinion-based'                 
]

def main():
    st.title("StackOverflow Question Classifier")

    # Labels input (assuming multiple labels can be entered)
    labels_input = st.text_input("Enter labels (comma-separated):")
    labels = [label.strip() for label in labels_input.split(',')] if labels_input else []

    # Title input
    title = st.text_input("Enter the title of your StackOverflow question:")

    # Title input
    body = st.text_input("Enter the body of your StackOverflow question:")


    # Submit button
    if st.button("Classify"):
        # Check if the title and labels are not empty
        if title and labels and body:
            # Pre-process the input
            df = preprocess_data(pd.DataFrame([{"title": title, "body": body, "tags": labels, 'closed_reason': 'valid-question'}]))
            df = processor.transform(df)
            df = selector.transform(df)
            predicted_class = model.predict(df)

            # Display the prediction
            st.write(f"The predicted class for the entered question is: {label_list[predicted_class[0]]}")
        else:
            st.error("Please enter both a title and labels.")

if __name__ == "__main__":
    main()
