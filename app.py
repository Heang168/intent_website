from flask import Flask, request, render_template
import fasttext
from joblib import load
import numpy as np
from sklearn.preprocessing import LabelEncoder
from khnormal import correctingSpelling
from bilstm_tokenizer import tokenize_sentences_bilstm
import string
import re


model = fasttext.load_model("CrawlMergeFasttextEmbeddingv3.bin")
encoder = LabelEncoder()

encoder.classes_ = np.load("labelEncoder_classes_.npy", allow_pickle=True)
estimator = load("MLP_best_parameter.pkl")


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def removeStopWordAndPunct(question):
    stopWords = ['អោយ', 'ដែរ', 'ណា', 'លើ', 'ដោយ', 'មើល', 'នាយ', 'អ្វី', 'បន្ដិច', 'ថ្មី', 'ដល់', 'ដឹង', 'អាច', 'តើ',
                 'មួយ', 'ឯ', 'និង', 'ត្រូវ', 'ល្អ', 'ថា', 'ម៉េច', 'នៅ', 'នេះ', 'អ្នក', 'និយម', 'មាន', 'ឬ', 'ពួក',
                 'ប្រាប់', 'មែន', 'អត់', 'នោះ', 'ក្នុង', 'អស់', 'ប៉ុន្មាន', 'ខាង', 'យើង', 'ដែល', 'វា', 'ពី', 'គួរ',
                 'បាន', 'វិញ', 'ចំនួន', 'សួរ', 'ខ្លះ', 'ធ្វើ', 'ការ', 'ពេល', 'អី', 'ទៅ', 'តែ', 'គេ', 'រឺ', 'ទេ', 'ទទួល',
                 'ខ្ញុំ', 'សរុប', 'របស់', 'ចង់', 'បើ', 'ក៏', 'នូវ', 'នឹង', 'លែង', 'កំពុង', 'អំពី', 'យ៉ាង', 'គាត់',
                 'ខ្លួន', 'គឺ', 'នាង', 'មក', 'លោក', 'អតីត', 'យក', 'នៃ', 'ហើយ']

    newSentence = remove_punctuation(question)
    for word in newSentence.split(" "):
        if word in stopWords:
            newSentence = newSentence.replace(word, "")
            newSentence = re.sub(r"{}".format(word), "", newSentence)
            newSentence = re.sub(r"\s{2,}", " ", newSentence)
            newSentence = re.sub(r"\s$|^\s", "", newSentence)

    return newSentence


# Create an app object using the Flask class.
app = Flask(__name__)


# Load the trained model. (Pickle file)
# model = pickle.load(open('models/model.pkl', 'rb'))

# Define the route to be home.
# The decorator below links the relative route of the URL to the function it is decorating.
# Here, home function is with '/', our root directory.
# Running the app sends us to index.html.
# Note that render_template means it looks for the file in the templates folder.

# use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')


# You can use the methods argument of the route() decorator to handle different HTTP methods.
# GET: A GET message is send, and the server returns data
# POST: Used to send HTML form data to the server.
# Add Post method to the decorator to allow for form submission.
# Redirect to /predict page with the output
@app.route('/predict', methods=['POST'])
def predict():
    tmp_features = [x.strip() for x in request.form.values()]  # Convert string inputs to float.
    segmented = tokenize_sentences_bilstm(tmp_features)
    corrected = correctingSpelling(segmented)

    afterStopWord = removeStopWordAndPunct(corrected[0])
    vectorized = model.get_sentence_vector(afterStopWord)
    rose = estimator.predict([vectorized])
    label = encoder.inverse_transform(rose)

    # prediction = model.predict([features])  # features Must be in the form [[a, b]]

    return render_template('index.html',additional={"segmented":corrected, "afterStopWord": afterStopWord}, prediction_text='{}   =>   {}'.format(tmp_features[0], label[0]))


# When the Python interpreter reads a source file, it first defines a few special variables.
# For now, we care about the __name__ variable.
# If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__).  
# So if we want to run our code right here, we can check if __name__ == __main__
# if so, execute it here.
# If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
