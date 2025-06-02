
# IvoireDelices_Chatbot.py - Chatbot vocal pour restaurant ivoirien avec interface moderne

import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import tempfile

# Configuration de la page
st.set_page_config(page_title="🍽️ Ivoire Délices Chatbot", page_icon="🍛", layout="wide")

# --- CSS personnalisé ---
st.markdown("""
<style>
body {
    background-color: #fff8f0;
    font-family: 'Segoe UI', sans-serif;
}
.chat-container {
    background-color: #ffffff;
    border-radius: 15px;
    padding: 20px;
    margin-top: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
.user-bubble {
    background-color: #ff9800;
    color: white;
    padding: 10px 15px;
    border-radius: 18px 18px 0 18px;
    margin-bottom: 10px;
    max-width: 70%;
    text-align: right;
    margin-left: auto;
}
.bot-bubble {
    background-color: #e8f5e9;
    color: #2e7d32;
    padding: 10px 15px;
    border-radius: 18px 18px 18px 0;
    margin-bottom: 10px;
    max-width: 70%;
    text-align: left;
}
.title {
    text-align: center;
    color: #ff6f00;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #4caf50;
    margin-bottom: 30px;
}
.audio-zone {
    background-color: #fff3e0;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
    border-left: 6px solid #ff9800;
}
</style>
""", unsafe_allow_html=True)

# Données du restaurant
questions = [
    "Qu'est-ce que Ivoire Délices?", "Quels plats proposez-vous?",
    "Quels sont vos horaires d'ouverture?", "Où êtes-vous situé?",
    "Est-ce que vous livrez à domicile?", "Comment puis-je commander?",
    "Quels sont vos plats les plus populaires?", "Quels sont vos prix?",
    "Acceptez-vous les paiements mobiles?", "Y a-t-il des plats végétariens?",
    "Faites-vous des promotions?", "Puis-je réserver une table?",
    "Avez-vous des boissons typiques?", "Y a-t-il des desserts ivoiriens disponibles?",
    "Puis-je organiser un événement chez vous?", "Quel est votre numéro de téléphone?",
    "Puis-je commander pour un groupe?", "Quels sont vos moyens de contact?",
    "Quels sont les délais de livraison?", "Quels sont les jours d'ouverture?"
]

answers = [
    "Ivoire Délices est un restaurant ivoirien traditionnel situé à Abidjan, proposant des plats authentiques faits maison.",
    "Nous servons garba, attiéké poisson, foutou, sauce graine, kedjenou, alloco, riz gras, placali, etc.",
    "Ouvert tous les jours de 10h à 22h.", "Situé à Abidjan, Cocody Angré, près du carrefour les Oscars.",
    "Oui, livraison disponible via Glovo, Yango ou notre service interne.", 
    "Commandez par téléphone, WhatsApp ou en ligne.", 
    "Garba spécial, attiéké poisson braisé, foutou sauce graine et kedjenou de poulet.", 
    "Prix entre 1500 et 6000 FCFA.", 
    "Oui, Mobile Money (Orange, MTN, Moov) accepté.", 
    "Oui, options végétariennes disponibles.", 
    "Promos régulières sur nos réseaux sociaux.", 
    "Oui, réservation possible par appel ou WhatsApp.", 
    "Boissons : jus de bissap, gingembre, bouye, etc.", 
    "Desserts : dégué, gâteau manioc, bananes flambées.", 
    "Oui, anniversaires et événements sur réservation.", 
    "📞 +225 07 07 07 07 07", 
    "Oui, commandes de groupe acceptées (prévenez 24h à l'avance).", 
    "Contact : WhatsApp, email ou Instagram.", 
    "Livraison en 30 à 60 minutes.", 
    "Ouvert tous les jours, même fériés."
]

# Fonctions
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9À-ſ\s]", "", text)
    return text

def get_response(user_input):
    data = questions + [user_input]
    data = [preprocess(q) for q in data]
    tfidf = TfidfVectorizer().fit_transform(data)
    cosine_sim = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = cosine_sim.argmax()
    return answers[idx] if cosine_sim[0, idx] > 0.2 else "Désolé, je n'ai pas compris votre question."

def recognize_speech_from_audio(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    try:
        text = r.recognize_sphinx(audio, language='fr-FR')
        return text
    except sr.UnknownValueError:
        return "Désolé, je n'ai pas compris ce que vous avez dit."
    except sr.RequestError as e:
        return f"Erreur de reconnaissance vocale: {e}"

# --- Interface ---
st.markdown('<h1 class="title">🍽️ Bienvenue chez Ivoire Délices</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">Posez vos questions à notre chatbot vocal !</h3>', unsafe_allow_html=True)

with st.container():
    user_input = st.text_input("💬 Posez votre question ici...")
    if user_input:
        st.markdown(f'<div class="user-bubble">{user_input}</div>', unsafe_allow_html=True)
        bot_reply = get_response(user_input)
        st.markdown(f'<div class="bot-bubble">{bot_reply}</div>', unsafe_allow_html=True)

st.markdown('<div class="audio-zone"><strong>📢 Envoyez un audio (WAV, FR)</strong></div>', unsafe_allow_html=True)
audio_file = st.file_uploader("", type=["wav"])
if audio_file:
    st.audio(audio_file, format="audio/wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    transcription = recognize_speech_from_audio(tmp_path)
    st.markdown(f'<div class="user-bubble">🗣️ {transcription}</div>', unsafe_allow_html=True)
    bot_reply = get_response(transcription)
    st.markdown(f'<div class="bot-bubble">{bot_reply}</div>', unsafe_allow_html=True)
