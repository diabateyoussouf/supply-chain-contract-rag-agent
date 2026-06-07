import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage

# Configuration des chemins
PDF_DIR = "data/contracts"
VECTORSTORE_DIR = "data/vectorstore_contracts_V2"

def init_vectorstore():
    """Initialise ou charge la base de données vectorielle Chroma."""
    model_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(VECTORSTORE_DIR) and len(os.listdir(VECTORSTORE_DIR)) > 0:
        vectorstore = Chroma(
            collection_name="contracts",
            embedding_function=model_embeddings,
            persist_directory=VECTORSTORE_DIR
        )
    else:
        if not os.path.exists(PDF_DIR):
            os.makedirs(PDF_DIR)
            
        loader = PyPDFDirectoryLoader(PDF_DIR)
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
            encoding_name="cl100k_base"
        )
        chunks = loader.load_and_split(splitter)
        
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=model_embeddings,
            collection_name="contracts",
            persist_directory=VECTORSTORE_DIR
        )
    return vectorstore

def get_available_contracts(vectorstore):
    """Récupère la liste unique des fichiers PDF indexés dans Chroma."""
    try:
        data = vectorstore.get()
        metadatas = data.get('metadatas', [])
        sources = set(m.get('source', '').split('/')[-1] for m in metadatas if m.get('source'))
        return sorted(list(sources))
    except Exception:
        return []

def execute_rag(query, selected_contract=None, uploaded_image_b64=None, temporary_pdf_text=None):
    """
    Exécute la chaîne RAG avec support multimodal (Image, Audio converti en texte, PDF à la volée).
    """
    context_for_query = ""
    
    # 1. Gestion du PDF importé "à la volée" par l'utilisateur (Priorité Contexte Local)
    if temporary_pdf_text:
        context_for_query += f"[Source: Document importé par l'utilisateur]\n{temporary_pdf_text}\n\n---\n\n"
    
    # 2. RAG classique sur la base de données vectorielle
    vectorstore = init_vectorstore()
    search_kwargs = {"k": 15}
    
    if selected_contract and selected_contract != "Tous les contrats":
        search_kwargs["filter"] = {"source": {"$regex": f".*{selected_contract}"}}
        
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    
    relevant_document_chunks = retriever.invoke(query)
    
    context_list = []
    for chunk in relevant_document_chunks:
        source_path = chunk.metadata.get("source", "Contrat Inconnu")
        source_name = source_path.split("/")[-1] 
        page_num = chunk.metadata.get("page", 0) + 1
        
        formatted_chunk = f"[Source: {source_name} - Page {page_num}]\n{chunk.page_content}"
        context_list.append(formatted_chunk)
        
    context_for_query += "\n\n---\n\n".join(context_list)
    
    # Configuration du LLM - Clé d'API
    cle_extraite = os.getenv("CUAD_KEY")
    if not cle_extraite:
        return "Erreur : La clé API 'CUAD_KEY' n'est pas configurée dans les variables d'environnement."
        
    # Passage au modèle Large pour assurer le traitement d'images (Vision) et le multilingue parfait
    llm = ChatMistralAI(
        model="mistral-large-latest", 
        temperature=0.0,
        api_key=cle_extraite 
    )
    
    # Construction du système d'instructions
    system_instruction = """You are an expert AI assistant specializing in legal and supply chain contracts review.
Your task is to answer the user's question accurately, directly, and concisely based ONLY on the provided contract excerpts (Context) and any attached visual layout.

CRITICAL INSTRUCTIONS:
1. Language Mirroring: Detect the language of the User's Question (e.g., French, English). You MUST formulate your entire response in that EXACT same language. Translate the insights from the context accurately if necessary.
# Dans src/agent_service.py, affine l'instruction 2 :
2. Grounding & UX: Be direct, structured, and use clear bullet points. Do not invent or assume anything. 
   When reading numbers, decimals, or percentages from an image or table, extract them with absolute precision (e.g., do not confuse 3.1% with 31%).
3. Clean Citations (No raw filenames): You MUST cite where you found the information, but make it very elegant, short, and adapted to the target language.
   - NEVER include raw file extensions (like .pdf) or system codes (like _20150417_8-K_EX-10.5_Transportation Agreement).
   - Clean the contract name to make it user-friendly (e.g., write "Range Resources" or "Document importé").
   - Format example in French: "(Page 6)" or "(Range Resources, Page 15)".
4. Strictness: If the context does not contain the specific answer to the question, state clearly in the target language that you cannot find the information in the current documents. Do not attempt to generalize."""

    # 3. Préparation du contenu du message utilisateur (Multimodal structuré)
    user_content = [
        {
            "type": "text", 
            "text": f"Voici le contexte des contrats disponibles :\n<context>\n{context_for_query}\n</context>\n\nQuestion de l'utilisateur : {query}"
        }
    ]
    
    # Si l'utilisateur a injecté une image dans l'interface, on la pousse dans la mémoire visuelle de Mistral
    if uploaded_image_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{uploaded_image_b64}"
            }
        })
        
    # Packaging final des messages sous format LangChain standardisé
    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_content)
    ]
    
    response = llm.invoke(messages)
    return response.content