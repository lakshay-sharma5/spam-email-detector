import streamlit as st
import joblib

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧"
)

st.title("📧 Spam Email Detector")
st.write("Enter an email below to check whether it is Spam or Not Spam.")

# Load model and vectorizer
try:
    model = joblib.load("spam_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    st.error("Could not load the model files.")
    st.code(str(e))
    st.stop()

# Email input
email = st.text_area(
    "Enter Email Text",
    height=200,
    placeholder="Type or paste your email here..."
)

# Prediction
if st.button("🔍 Check Email", use_container_width=True):

    if not email.strip():
        st.warning("Please enter an email first.")

    else:
        try:
            email_tfidf = tfidf.transform([email])
            prediction = model.predict(email_tfidf)[0]

            if str(prediction) == "1":
                st.error("🚨 This email is SPAM")
            else:
                st.success("✅ This email is NOT SPAM")

        except Exception as e:
            st.error("Prediction error:")
            st.code(str(e))