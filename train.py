import pandas as pd
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')   



# step 1: Load dataset
df = pd.read_csv(r"Machine Learning\ML Project\Supervised ML Project\Intermediate\Email Spam Detector\Dataset\spam.csv", encoding='latin-1')



# step 2: Remove unnecessary columns from the dataset. and hold only the 'text' and 'label' columns.
df = df[['v1', 'v2']]          # we are keeping only the 'v1' and 'v2' columns which contain the label and text data respectively.
df.columns = ['label', 'text']   # renaming the columns to 'label' and 'text' for better understanding.
# print(df.head())


stop_words = set(stopwords.words('english')) 

# step 3: clean the text data by removing special characters, numbers, punctuations etc.
def clean_text(text):

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove emojis and special characters
    text = ''.join(char for char in text if char.isascii())

    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # tokenize
    words = word_tokenize(text)

    # remove stopwords
    words = [word for word in words if word not in stop_words]

    # remove numbers.
    # it is not necessary to remove numbers for email spam detection because sometimes numbers can be used in spam emails to attract attention.

    return ' '.join(words)

# Apply cleaning  ( apply this clean_text function to the 'text' column of the dataframe and create a new column 'clean_text' to store the cleaned text data.)
df['clean_text'] = df['text'].apply(clean_text)




# step 4: split the dataset into training and testing sets.
X = df['clean_text']  # features (cleaned text data)
y = df['label']       # target variable (labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)




# step 5: convert the text data into numerical features using TF-IDF vectorization.
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)



# step 6: train a machine learning model (Naive Bayes) on the training data.
model = MultinomialNB()

#  Train
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")

# step 7: import os is use to create a directory named "models" if it does not already exist, and then save the trained model and vectorizer using joblib.dump() function. This allows us to reuse the model and vectorizer later for making predictions without having to retrain the model every time.
import os
os.makedirs("models", exist_ok=True)


# step 8: save the trained model and the vectorizer for future use.
joblib.dump(model, r'D:\AIML\Machine Learning\ML Project\Supervised ML Project\Intermediate\Email Spam Detector\models\spam_model.pkl')
joblib.dump(vectorizer, r'D:\AIML\Machine Learning\ML Project\Supervised ML Project\Intermediate\Email Spam Detector\models\vectorizer.pkl')

print("Model saved successfully.")