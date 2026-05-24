import streamlit as st
import joblib
import re
import string

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.title {
    font-size: 42px;
    font-weight: bold;
    color: white;
    text-align: center;
    margin-top: 20px;
}

.subtitle {
    font-size: 18px;
    color: #cbd5e1;
    text-align: center;
    margin-bottom: 30px;
}

.stTextArea textarea {
    background-color: #1e293b;
    color: white;
    border-radius: 12px;
    border: 2px solid #334155;
    font-size: 18px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
    margin-top: 25px;
}

.spam {
    background-color: #7f1d1d;
    color: #fecaca;
}

.ham {
    background-color: #14532d;
    color: #bbf7d0;
}

.confidence {
    text-align: center;
    font-size: 18px;
    color: white;
    margin-top: 10px;
}

.footer {
    text-align: center;
    margin-top: 40px;
    color: #94a3b8;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = joblib.load(
    r'D:\AIML\Machine Learning\ML Project\Supervised ML Project\Intermediate\Email Spam Detector\models\spam_model.pkl'
)

vectorizer = joblib.load(
    r'D:\AIML\Machine Learning\ML Project\Supervised ML Project\Intermediate\Email Spam Detector\models\vectorizer.pkl'
)

# =========================
# CLEANING FUNCTION
# =========================
def clean_text(text):

    text = text.lower()

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\s+', ' ', text).strip()

    return text

# =========================
# UI
# =========================
st.markdown('<div class="title">📧 Spam Email Classifier</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Detect whether a message is Spam or Not Spam using NLP + Machine Learning</div>',
    unsafe_allow_html=True
)

message = st.text_area(
    "Enter Your Message",
    height=180,
    placeholder="Type your email or message here..."
)

# =========================
# PREDICT BUTTON
# =========================
if st.button("Predict Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        cleaned_message = clean_text(message)

        vector_input = vectorizer.transform([cleaned_message])

        prediction = model.predict(vector_input)[0]

        confidence = model.predict_proba(vector_input).max() * 100

        # SPAM
        if prediction == "spam":

            st.markdown(
                f"""
                <div class="result-box spam">
                    🚨 Spam Message
                </div>
                """,
                unsafe_allow_html=True
            )

        # NOT SPAM
        else:

            st.markdown(
                f"""
                <div class="result-box ham">
                    ✅ Not Spam
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f'<div class="confidence">Confidence: {confidence:.2f}%</div>',
            unsafe_allow_html=True
        )

# =========================
# FOOTER
# =========================
st.markdown(
    '<div class="footer">Built using Streamlit, TF-IDF and Naive Bayes</div>',
    unsafe_allow_html=True
)