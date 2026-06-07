# 📜 Contract AI Agent - Supply Chain Multimodal RAG

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MistralAI](https://img.shields.io/badge/Mistral_AI-Large-000000?style=for-the-badge&logo=mistral&logoColor=white)](https://mistral.ai/)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-1C3C3C?style=for-the-badge)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-007ACC?style=for-the-badge)](https://idna.com/)

Ce projet est un Agent d'intelligence artificielle avancé basé sur une architecture **RAG (Retrieval-Augmented Generation) Multimodale**. Il permet d'analyser, de requêter et d'extraire des informations complexes (clauses juridiques, tableaux de tarification, graphiques logistiques) à partir de contrats Supply Chain (Dataset CUAD / OCP).

---

## 🚀 Fonctionnalités

- **RAG Textuel Strict** : Analyse approfondie des PDF avec indexation vectorielle (`ChromaDB` & `HuggingFace Embeddings`). Zero hallucination grâce à un alignement strict sur le contexte.
- **Multilingue Dynamique (Language Mirroring)** : Posez vos questions en Français, en Anglais ou dans n'importe quelle langue. L'agent détecte la langue de la question et formule sa réponse dans cette même langue.
- **Citations Épurées** : L'IA extrait automatiquement les numéros de pages et les noms de contrats nettoyés de leurs codes système bruts (ex: `Range Resources, Page 15`).
- **Analyse Multimodale (Vision)** : Capacité de vision native via `Mistral Large` permettant d'analyser et d'extraire des métriques depuis des captures d'écran de tableaux financiers ou d'annexes tarifaires.
- **Flexibilité des Entrées** : Prise en charge des questions textuelles classiques, des commandes vocales (fichiers Audio) ainsi que de l'analyse de nouveaux contrats PDF à la volée.

---

## 📂 Structure du Projet

```text
supply-chain-contract-rag-agent/
│
├── data/
│   ├── contracts/                    # Dossier où déposer vos fichiers PDF à indexer
│   └── vectorstore_contracts_V2/     # Base de données vectorielle locale ChromaDB (Générée automatiquement)
│
├── src/
│   ├── app.py                        # Interface utilisateur de chat (Streamlit)
│   └── agent_service.py              # Moteur logique de l'agent (ChromaDB, Retriever, Pipeline Mistral Large)
│
├── .env                              # Clés secrètes (Exclu du dépôt distant par le .gitignore)
├── .gitignore                        # Fichiers et dossiers à exclure de Git
├── README.md                         # Documentation du projet
└── requirements.txt                  # Liste des dépendances Python indispensables
```
