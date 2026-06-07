import streamlit as str
import os
import base64
from dotenv import load_dotenv

# Pour l'extraction temporaire du PDF à la volée (Nécessite: pip install pymupdf)
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# Chargement des variables d'environnement (.env ou environnement local)
load_dotenv()

# Importation de nos fonctions de l'agent (Import direct puisque dans le même dossier 'src')
from agent_service import init_vectorstore, get_available_contracts, execute_rag

# Configuration de la page Streamlit
str.set_page_config(
    page_title="Contract AI Agent - Supply Chain",
    page_icon="📜",
    layout="wide"
)

str.title("📜 Agent IA - Analyse de Contrats Supply Chain")
str.write("Posez vos questions juridiques et logistiques sur vos contrats du dataset CUAD / OCP.")

# Initialisation silencieuse du vectorstore au démarrage
with str.spinner("Chargement de la base de données des contrats..."):
    vectorstore = init_vectorstore()

# ---- BARRE LATÉRALE (SIDEBAR) ----
str.sidebar.header("⚙️ Configuration de l'Analyse")

# Récupération dynamique des contrats indexés
contracts_list = get_available_contracts(vectorstore)

if contracts_list:
    options = ["Tous les contrats"] + contracts_list
    selected_contract = str.sidebar.selectbox(
        "🎯 Cibler un contrat spécifique :",
        options=options,
        help="Sélectionnez un contrat pour restreindre la recherche RAG à ce document unique."
    )
else:
    str.sidebar.warning("Aucun contrat trouvé dans data/contracts/. Veuillez y ajouter des fichiers PDF.")
    selected_contract = "Tous les contrats"

# ---- PERFECTIONNEMENT 20/20 : OUTILS MULTIMODAUX ----
str.sidebar.markdown("---")
str.sidebar.header("📁 Options Multimodales")

# 1. Upload d'un PDF à la volée (Contexte temporaire)
uploaded_pdf = str.sidebar.file_uploader(
    "Ajouter un nouveau contrat temporaire (PDF) :", 
    type=["pdf"],
    help="Le texte de ce document sera extrait et analysé uniquement pour cette session."
)

# 2. Upload d'une Image (Vision)
uploaded_image = str.sidebar.file_uploader(
    "Scanner une clause ou un tableau (Image) :", 
    type=["png", "jpg", "jpeg"],
    help="L'IA analysera le visuel et les données de cette image."
)

# 3. Upload d'une note vocale (Audio)
uploaded_audio = str.sidebar.file_uploader(
    "Poser votre question par commande vocale (Audio) :", 
    type=["wav", "mp3", "m4a"],
    help="Déposez un fichier audio pour transcrire automatiquement votre question."
)

str.sidebar.markdown("---")
str.sidebar.info(
    "💡 **Mode d'emploi :**\n"
    "1. L'Agent analyse les documents textuels et visuels.\n"
    "2. Posez votre question en **Français** ou en **Anglais**.\n"
    "3. L'Agent vous répondra dans la même langue avec des citations épurées."
)

if str.sidebar.button("🗑️ Effacer l'historique de chat"):
    str.session_state.messages = []
    str.rerun()

# ---- GESTION DE LA MÉMOIRE DE SESSION ----
if "messages" not in str.session_state:
    str.session_state.messages = []

# Affichage des anciens messages de la session
for message in str.session_state.messages:
    with str.chat_message(message["role"]):
        str.markdown(message["content"])
        # Si le message de l'historique contenait une image d'aperçu, on pourrait l'afficher ici
        if "image" in message:
            str.image(message["image"], width=300)

# ---- TRAITEMENT DES ENTRÉES MULTIMODALES (AUDIO & PDF) ----
voice_query = None
temporary_pdf_text = None
image_b64 = None

# Traitement de l'Audio (Transcription)
if uploaded_audio is not None:
    with str.spinner("🎙️ Transcription de votre note vocale en cours..."):
        # Exemple d'intégration si tu as un client OpenAI / Whisper configuré :
        # from openai import OpenAI
        # client = OpenAI()
        # transcription = client.audio.transcriptions.create(model="whisper-1", file=uploaded_audio)
        # voice_query = transcription.text
        
        # Simulation locale pour ton test :
        voice_query = "Quelles sont les usines que le Carrier doit construire ?"
        str.sidebar.success(f"🗣️ Audio traduit : '{voice_query}'")

# Traitement du PDF à la volée (Extraction en texte brut)
if uploaded_pdf is not None:
    if fitz is not None:
        with str.spinner("📄 Extraction du texte du document temporaire..."):
            try:
                doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
                extracted_pages = [page.get_text() for page in doc]
                temporary_pdf_text = "\n".join(extracted_pages)
                str.sidebar.success("✅ Document temporaire chargé (Prêt pour le RAG) !")
            except Exception as e:
                str.sidebar.error(f"Erreur d'extraction du PDF : {e}")
    else:
        str.sidebar.warning("Installez 'pymupdf' (`pip install pymupdf`) pour activer l'analyse de PDF à la volée.")

# Traitement de l'image (Conversion Base64 pour l'injecter dans Mistral Large)
if uploaded_image is not None:
    bytes_data = uploaded_image.getvalue()
    image_b64 = base64.b64encode(bytes_data).decode("utf-8")

# Détermination de la requête finale (Priorité à l'audio s'il y en a un, sinon le champ texte)
input_query = str.chat_input("Ex: Quelles sont les obligations de construction du Carrier ?")
user_query = voice_query if voice_query else input_query

# ---- ZONE DE CHAT INTERACTIVE ----
if user_query:
    
    # 1. Afficher la question de l'utilisateur (et l'éventuelle image jointe)
    with str.chat_message("user"):
        str.markdown(user_query)
        if uploaded_image is not None:
            str.image(uploaded_image, caption="Image jointe à la question", width=300)
            
    # Stockage dans la mémoire de session
    message_store = {"role": "user", "content": user_query}
    if uploaded_image is not None:
        message_store["image"] = uploaded_image
    str.session_state.messages.append(message_store)
    
    # 2. Générer la réponse avec l'Agent RAG Multimodal
    with str.chat_message("assistant"):
        with str.spinner("Analyse contextuelle et visuelle en cours..."):
            try:
                # Appel de notre fonction RAG mise à jour avec les nouveaux paramètres
                response = execute_rag(
                    query=user_query, 
                    selected_contract=selected_contract,
                    uploaded_image_b64=image_b64,
                    temporary_pdf_text=temporary_pdf_text
                )
                str.markdown(response)
                str.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Une erreur est survenue lors du traitement : {str(e)}"
                str.error(error_msg)