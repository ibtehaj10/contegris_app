from flask import Flask, request, jsonify
import json
# import jsonpickle
import spacy_udpipe
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import threading
import fasttext
from huggingface_hub import hf_hub_download
app = Flask(__name__)

spacy_udpipe.download("ur")
spacy_udpipe.download("en")


VALID_TOKEN = '940809d05631dea13edae629a420fdd9'
############################################
def token_required(f):
    def decorator(*args, **kwargs):
        token = None
        # Check if an Authorization header is part of the request
        if 'Authorization' in request.headers:
            # Attempt to extract the token from the header
            token = request.headers['Authorization'].split(" ")[1]
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        if token != VALID_TOKEN:
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(*args, **kwargs)
    decorator.__name__ = f.__name__
    return decorator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########################## MODEL LOADING #######################
# lang_tokenizer = AutoTokenizer.from_pretrained("mwz/LanguageDetection")
# lang_model = AutoModelForSequenceClassification.from_pretrained("mwz/LanguageDetection").to(device)
model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)
# model  = model.to(device)

ur_sent_token = AutoTokenizer.from_pretrained("mwz/UrduClassification")
# ur_sent_model = AutoModelForSequenceClassification.from_pretrained("mwz/UrduClassification")

ru_sent_token = AutoTokenizer.from_pretrained("mwz/RomanUrduClassification")
# ru_sent_model = AutoModelForSequenceClassification.from_pretrained("mwz/RomanUrduClassification")

en_sent_token = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
# en_sent_model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
# For the language detection model (already done in your snippet)
# lang_model = AutoModelForSequenceClassification.from_pretrained("mwz/LanguageDetection").to(device)

# For sentiment models (do this right after loading each model)
ur_sent_model = AutoModelForSequenceClassification.from_pretrained("mwz/UrduClassification").to(device)
ru_sent_model = AutoModelForSequenceClassification.from_pretrained("mwz/RomanUrduClassification").to(device)
en_sent_model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb").to(device)

lang_model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
lang_model = fasttext.load_model(lang_model_path)
####################################### Language detector
def detect_language(text):


    predictions = lang_model.predict(text, k=5)

    for prediction in predictions:
        label = prediction[0]
        if label in ("__label__eng_Latn", "__label__urd_Arab"):
            return label.replace("__label__", "")
    return "Roman Urdu"

############### CLEAN TEXT ##############
def clean_text(input_text):
    # Define regex patterns for email, phone, URL, and website
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'((?:\+\d{2}[-\.\s]??|\d{4}[-\.\s]??)?(?:\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}))'
    url_pattern = r"\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s!()[]{};:'\".,<>?«»“”‘’]))"
    website_pattern = r"(http|https)://[\w-]+(.[\w-]+)+\S*"

    cleaned_text = re.sub(email_pattern, '', input_text)
    cleaned_text = re.sub(phone_pattern, '', cleaned_text)
    cleaned_text = re.sub(url_pattern, '', cleaned_text)
    cleaned_text = re.sub(website_pattern, '', cleaned_text)

    return cleaned_text

################ MAIN FUNCTION #################################
def analyze_text_combined(text):
    global lang_model, ur_sent_model, ru_sent_model, en_sent_model
    # Clean the input text
    text = str(text)


    if torch.cuda.is_available():
        # print("CUDA is available. GPU:", torch.cuda.get_device_name(0))
        pass
    else:
        print("CUDA is not available.")

    # print(text)
    cleaned_text = clean_text(text)
    
    

    detected_language = detect_language(cleaned_text)

    # Sentiment Analysis
    sentiment_tokenizer = None
    sentiment_model = None

    if detected_language == "urd_Arab":
        sentiment_tokenizer = ur_sent_token
        sentiment_model = ur_sent_model 
    elif detected_language == "Roman Urdu":
        sentiment_tokenizer = ru_sent_token
        sentiment_model = ru_sent_model
    else:
        sentiment_tokenizer = en_sent_token 
        sentiment_model = en_sent_model

# Assuming sentiment_tokenizer and sentiment_model are already defined

# Prepare the input and move it to the device (GPU or CPU)
    sentiment_model = sentiment_model.to(device)

    sentiment_input = sentiment_tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True)
    sentiment_input = {k: v.to(device) for k, v in sentiment_input.items()}

    sentiment_logits = sentiment_model(**sentiment_input).logits
    sentiment_logits = sentiment_logits.cpu()
    sentiment_pred = sentiment_logits.argmax().item()


    sentiment_label = ""
    if detected_language == "urd_Arab":
        sentiment_label = "Positive" if sentiment_pred == 0 else "Negative"
    elif detected_language == "Roman Urdu":
        sentiment_label = "Negative" if sentiment_pred == 0 else ("Neutral" if sentiment_pred == 1 else "Positive")
    else:
        sentiment_label = "Negative" if sentiment_pred == 0 else "Positive"


    # print('detected_lang  : ',detected_lang)
    # Verb Classification
    spacy_models = {
        "eng_Latn": spacy_udpipe.load("en"),
        "Roman Urdu": spacy_udpipe.load("en"),
        "urd_Arab": spacy_udpipe.load("ur"),
    }

    nlp = spacy_models.get(detected_language)
    if nlp is None:
        print("Language not supported")
        return

    doc = nlp(text)

    verbs = []
    for token in doc:
        if token.pos_ == "VERB":
            verbs.append(token.text)
    roman_actionable=[]
    # Actionable Words for Roman Urdu
    actionable_file_path = "actionable.txt"  # Update this with the correct path
    with open(actionable_file_path, "r", encoding="utf-8") as actionable_file:
        actionable_words_roman = [line.strip().lower() for line in actionable_file]
        roman_actionable = actionable_words_roman
    # print(roman_actionable)
    #actionable_words = []
    if detected_language == "Roman Urdu":
        for i in text.split(' '):
                # print(i)
                if i in roman_actionable:
                    # print(i)
                    verbs.append(i)
                # print(j)  
                # if i == j:

        
                    # print("detect action : ",i)
                    #verbs.append(i)
        # verbs = actionable_words_roman

    # Profanity Classification
    profanity_file_path = "All_Profane.txt"  # Update this with the correct path
    with open(profanity_file_path, "r", encoding="utf-8", errors="ignore") as profanity_file:
        profanity_words = [line.strip() for line in profanity_file]

    masked_text = text
    masked_words = []


    for word in profanity_words:
        regex = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
        masked_word = get_replacement_for_swear_word(word)

        if regex.search(masked_text):
            masked_text = regex.sub(masked_word, masked_text)
            masked_words.append({"word": word, "index": masked_text.index(masked_word)})
        if masked_words == []:
            pass
        else:
            sentiment_label = "Negative"

            
    if detected_language == "urd_Arab":
        detected_language = "Urdu"

    return {
        "Detected Language": detected_language,
        "Sentiment": sentiment_label,
        "Actionable words": verbs,
        "Profanities": masked_words,
    }

def get_replacement_for_swear_word(word):
    return word[:1] + "*" * (len(word) - 2) + word[-1:]




def analyze_text_combined_wrapper(text, result_container):
    # Call the original function and store its result
    result = analyze_text_combined(text)
    result_container['result'] = result


@app.route('/predict', methods=['POST'])
@token_required
def predict():
    data = request.json
    text = data['text']
    
    # Container for the result from the thread
    result_container = {}
    
    # Start the thread, passing the container for the result
    prediction_thread = threading.Thread(target=analyze_text_combined_wrapper, args=(text, result_container))
    prediction_thread.start()
    prediction_thread.join()  # Wait for the thread to complete

    # Retrieve the result from the container
    result = result_container.get('result', 'Processing error')
    return jsonify({'result': result})

if __name__ == '__main__':
    # Create a thread for running the Flask app
    app.run(threaded=True,host="0.0.0.0")