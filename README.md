RAG avec SPLADE et SmolLM2

Ce projet implémente et évalue une pipeline de Retrieval-Augmented Generation (RAG) sur le domaine Technology du jeu de données LoTTE. La recherche de documents repose sur SPLADE et la génération de réponses sur SmolLM2.

L'objectif est de comparer plusieurs stratégies d'entraînement et d'amélioration de la recherche :

entraînement contrastif avec InfoNCE ;
entraînement par distillation ;
augmentation de données à partir de questions synthétiques ;
réécriture de requêtes par un LLM avant la recherche.

Les pipelines évaluées sont :

InfoNCE ;
InfoNCE + Data Augmentation ;
InfoNCE + Data Augmentation + Rewriting ;
Distillation.

La qualité de la recherche est mesurée avec les métriques MRR et Recall sur 20 requêtes de test. La qualité de génération est évaluée avec BERTScore, en comparant les réponses de SmolLM2 à celles générées par Gemini à partir du même contexte récupéré.

Structure du projet
src/
├── Splade.py          # Modèle SPLADE
├── data.py            # Chargement et préparation des données
├── rag.py             # Pipeline RAG
├── prompting.py       # Prompts et réécriture de requêtes
├── score.py           # Métriques d'évaluation
├── loss.py            # Fonctions de perte
└── evaluation/        # Scripts d'entraînement et d'évaluation


On installe le projet comme ca : 

