import re
import string
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# download nltk data
nltk.download('stopwords')
nltk.download('punkt')

# stopwords
stop_words = set(stopwords.words('english'))

# text cleaning function
def clean_text(text):

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # remove html tags
    text = re.sub(r'<.*?>', '', text)

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove emojis
    text = text.encode('ascii', 'ignore').decode('ascii')

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # tokenize
    words = word_tokenize(text)

    # remove stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Load model
model = joblib.load(
    r'D:\AIML\Machine Learning\ML Project\Supervised ML Project\Intermediate\Email Spam Detector\models\spam_model.pkl'
)

# Load vectorizer
vectorizer = joblib.load(
    r'D:\AIML\Machine Learning\ML Project\Supervised ML Project\Intermediate\Email Spam Detector\models\vectorizer.pkl'
)

# User input
message = input("Enter message: ")

# CLEAN INPUT MESSAGE
cleaned_message = clean_text(message)

# Transform cleaned text
message_vector = vectorizer.transform([cleaned_message])

# Prediction
prediction = model.predict(message_vector)[0]

# Calculate confidence
confidence     = model.predict_proba(message_vector).max() * 100

# Output
if prediction == "spam":
    print("Spam Message")
else:
    print("Not Spam")

print(f"Confidence: {confidence:.2f}%")