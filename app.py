import streamlit as st

# Assuming you have a function to predict the class based on text and label
# Import your model and prediction function here

labels_set = set()  # Set to store unique labels

import streamlit as st
import joblib

# Load the trained model (using Streamlit's cache mechanism)
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('model_filename.joblib')
    return model

model = load_model()


def main():
    st.title("Custom Label Text Classification")

    # Text input
    text = st.text_area("Enter your text here:")

    # Label input
    label = st.text_input("Enter your label:")

    # Display a warning if there are already 5 unique labels
    if len(labels_set) >= 5 and label not in labels_set:
        st.warning("Only 5 unique labels are allowed. Please enter one of the existing labels.")
    else:
        labels_set.add(label)

    # Submit button
    if st.button("Submit"):
        # Check if the text and label are not empty
        if text and label:
            # Call your model's prediction function here
            # For example: predicted_class = model.predict(text, label)
            predicted_class = "predicted_class"  # Placeholder for model prediction
            st.write(f"The predicted class for the entered text is: {predicted_class}")
        else:
            st.error("Please enter both text and a label.")

if __name__ == "__main__":
    main()
