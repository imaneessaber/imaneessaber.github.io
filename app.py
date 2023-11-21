from flask import Flask, render_template, request, render_template_string
import random
import string
import nltk
from nltk.stem import WordNetLemmatizer
import joblib
from pymessenger.bot import Bot

app = Flask(__name__)
PAGE_ACCESS_TOKEN = 'YOUR_PAGE_ACCESS_TOKEN'
bot = Bot(PAGE_ACCESS_TOKEN)
# Charger le modèle du chatbot
model = joblib.load('model.pkl')

# Reading in the corpus
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenization
sent_tokens = nltk.sent_tokenize(raw)  # converts to a list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to a list of words

# Preprocessing
lemmer = WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    """If the user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


disease_categories = {
    'category_1': 'Acne',
    'category_2': 'Back pain',
    'category_3': 'Blurry vision',
    'category_4': 'Body feels weak',
    'category_5': 'Cough',
    'category_6': 'Earache',
    'category_7': 'Emotional pain',
    'category_8': 'Feeling cold',
    'category_9': 'Feeling dizzy',
    'category_10': 'Foot ache',
    'category_11': 'Hair falling out',
    'category_12': 'Hard to breathe',
    'category_13': 'Head ache',
    'category_14': 'Heart hurts',
    'category_15': 'Infected wound',
    'category_16': 'Injury from sports',
    'category_17': 'Internal pain',
    'category_18': 'Joint pain',
    'category_19': 'Knee pain',
    'category_20': 'Muscle pain',
    'category_21': 'Neck pain',
    'category_22': 'Open wound',
    'category_23': 'Shoulder pain',
    'category_24': 'Skin issue',
    'category_25': 'Stomach ache',
}

category_advice = {
    'Acne': [
        "Wash your face regularly with a gentle cleanser.",
        "Avoid picking or squeezing acne lesions to prevent scarring.",
        "Consider using over-the-counter acne products with benzoyl peroxide or salicylic acid."
    ],
    'Back pain': [
        "Practice good posture to relieve back pain.",
        "Stretch your back and do gentle exercises to strengthen your core muscles.",
        "Apply hot or cold packs to the affected area for pain relief."
    ],
    'Blurry vision': [
        "If you experience sudden blurry vision, seek immediate medical attention.",
        "Regular eye exams are important to detect vision problems early.",
        "Protect your eyes from UV rays and computer screen strain."
    ],
    'Body feels weak': [
        "Ensure you are getting enough rest and sleep.",
        "Maintain a balanced diet rich in nutrients and vitamins.",
        "Consult a healthcare professional if weakness persists."
    ],
    'Cough': [
        "Stay hydrated and drink warm fluids to soothe your throat.",
        "Use a humidifier or take a steamy shower to ease coughing.",
        "If your cough is severe or prolonged, consult a doctor."
    ],
    'Earache': [
        "Avoid inserting objects into your ear and seek medical advice.",
        "Apply a warm cloth to the ear for comfort.",
        "Pain relievers can help with earache, but consult a healthcare professional for persistent pain."
    ],
    'Emotional pain': [
        "Talk to a therapist or counselor to address emotional pain.",
        "Reach out to friends and family for emotional support.",
        "Consider mindfulness and relaxation techniques to manage emotional distress."
    ],
    'Feeling cold': [
        "Wear warm clothing and layer up in cold weather.",
        "Keep your home warm and use blankets to stay cozy.",
        "If you have persistent cold sensations, consult a doctor."
    ],
    'Feeling dizzy': [
        "Sit down if you feel dizzy to prevent falling.",
        "Stay hydrated and eat regular, balanced meals.",
        "If dizziness is frequent or severe, consult a healthcare professional."
    ],
    'Foot ache': [
        "Wear comfortable and supportive footwear.",
        "Consider foot massages and gentle stretches to relieve foot pain.",
        "If the pain persists, consult a podiatrist or orthopedic specialist."
    ],
    'Hair falling out': [
        "Maintain a healthy diet and address any nutritional deficiencies.",
        "Avoid tight hairstyles and excessive heat styling.",
        "Consult a dermatologist if hair loss is significant or ongoing."
    ],
    'Hard to breathe': [
        "If you have difficulty breathing, seek immediate medical attention.",
        "Avoid allergens and irritants that may exacerbate breathing problems.",
        "Consider breathing exercises and pulmonary rehabilitation if needed."
    ],
    'Headache': [
        "Stay hydrated and rest in a quiet, dark room.",
        "Over-the-counter pain relievers may help with headaches.",
        "If headaches are chronic or severe, consult a doctor."
    ],
    'Heart hurts': [
        "If you experience chest pain, seek immediate medical attention.",
        "Follow heart-healthy habits, such as a balanced diet and regular exercise.",
        "Consult a cardiologist for heart-related concerns."
    ],
    'Infected wound': [
        "Clean the wound with mild soap and water and keep it covered.",
        "Apply an over-the-counter antibiotic ointment if recommended.",
        "Consult a healthcare professional if signs of infection worsen or persist."
    ],
    'Injury from sports': [
        "Rest and avoid strenuous activities to allow the injury to heal.",
        "Apply ice to reduce swelling and pain.",
        "Consider physical therapy or rehabilitation exercises for recovery."
    ],
    'Internal pain': [
        "If you have severe or persistent internal pain, consult a doctor.",
        "Pay attention to your diet and digestive habits for gastrointestinal comfort.",
        "Investigate the cause of internal pain through medical examination."
    ],
    'Joint pain': [
        "Maintain a healthy weight and exercise to strengthen the joints.",
        "Consider over-the-counter joint supplements like glucosamine and chondroitin.",
        "Consult a rheumatologist for chronic or severe joint pain."
    ],
    'Knee pain': [
        "Apply ice to reduce knee pain and inflammation.",
        "Wear a knee brace or use crutches if necessary.",
        "Consult an orthopedic specialist if knee pain persists or worsens."
    ],
    'Muscle pain': [
        "Rest the affected muscles and avoid strenuous activity.",
        "Apply heat or cold packs to relieve muscle pain.",
        "Consider physical therapy or massage for muscle pain relief."
    ],
    'Neck pain': [
        "Maintain good posture and avoid excessive strain on your neck.",
        "Gentle neck exercises and stretches may help alleviate neck pain.",
        "Consult a physiotherapist for persistent neck pain or injuries."
    ],
    'Open wound': [
        "Clean the wound with mild soap and water, and apply an antiseptic.",
        "Keep the wound covered with a sterile bandage to prevent infection.",
        "Seek medical attention for deep or contaminated wounds."
    ],
    'Shoulder pain': [
        "Rest the shoulder and avoid activities that worsen the pain.",
        "Apply ice to reduce shoulder pain and inflammation.",
        "Consult an orthopedic specialist for persistent shoulder pain."
    ],
    'Skin issue': [
        "Keep your skin clean and moisturized to prevent skin issues.",
        "Avoid harsh skincare products and excessive sun exposure.",
        "Consult a dermatologist for persistent or severe skin problems."
    ],
    'Stomach ache': [
        "Avoid overeating and maintain a balanced diet.",
        "Use over-the-counter antacids or digestive aids for temporary relief.",
        "Consult a gastroenterologist for chronic or severe stomach issues."
    ]
}

# Generating response
def response(user_response):
    robo_response = ''

    # Check first if it's a greeting
    if greeting(user_response) is not None:
        robo_response = robo_response + greeting(user_response)
    else:
        predicted_category = model.predict([user_response])[0]

        # Suppose you have already predicted the disease category and stored it in the 'predicted_category' variable
        if predicted_category in category_advice:
            advice_list = category_advice[predicted_category]
            random_advice = random.choice(advice_list)
            # Display the randomly generated advice
            robo_response = f"Imy: {random_advice}"
        else:
            # If the predicted category is not in the dictionary, display a default message
            robo_response = "Sorry, we don't have advice for this disease category."

    return robo_response


# This function will handle the chatbot logic
def chat_logic(user_message):
    robo_response = ''
    category_response = ''
    advice_response = ''
    greeting_response = ''  # Initialisation en dehors du bloc conditionnel
    sent_tokens.append(user_message)

    # Check first if it's a greeting
    greeting_response = greeting(user_message)
    if greeting_response:
        robo_response = robo_response + greeting_response
    else:
        predicted_category = model.predict([user_message])[0]

        # Suppose you have already predicted the disease category and stored it in the 'predicted_category' variable
        if predicted_category in category_advice:
            advice_list = category_advice[predicted_category]
            random_advice = random.choice(advice_list)
            # Display the randomly generated advice
            category_response = f" {predicted_category}"
            advice_response = f" {random_advice}"
            sent_tokens.remove(user_message)

    return greeting_response or '', category_response, advice_response


# Route to render the HTML template
@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-16') as file:
        content = file.read()
    return render_template_string(content)

# Route to handle chat logic
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_message']

    # Call the chatbot logic function
    greeting_response, category_response, advice_response = chat_logic(user_message)

    # Pass the updated data to the template
    with open('index.html', 'r', encoding='utf-16') as file:
        content = file.read()
    return render_template_string(content, greeting_response=greeting_response, category_response=category_response, advice_response=advice_response)


# Route pour le point de terminaison du webhook
@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        # La vérification du webhook lors de la configuration de l'application Messenger
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
    else:
        # Recevoir les messages et les gérer
        output = request.get_json()
        for event in output['entry']:
            messaging = event['messaging']
            for message in messaging:
                if message.get('message'):
                    recipient_id = message['sender']['id']
                    if message['message'].get('text'):
                        user_message = message['message']['text']
                        greeting_response, category_response, advice_response = chat_logic(user_message)

                        # Envoyer la réponse au bot
                        send_message(recipient_id, greeting_response + '\n' + category_response + '\n' + advice_response)

    return "Message Processed"

# Fonction pour vérifier le token lors de la configuration du webhook
def verify_fb_token(token_sent):
    if token_sent == 'YOUR_VERIFICATION_TOKEN':
        return request.args.get("hub.challenge")
    return 'Token de vérification non valide'

# Fonction pour envoyer un message via PyMessenger
def send_message(recipient_id, response):
    bot.send_text_message(recipient_id, response)


# Run the application
if __name__ == '__main__':
    app.run(debug=True)
