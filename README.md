# RAG avec SPLADE et SmolLM2

> Projet expérimental de **Retrieval-Augmented Generation (RAG)** : recherche documentaire avec des représentations SPLADE, puis génération de réponses avec **SmolLM2-1.7B-Instruct**. Évaluation sur **LoTTE / Technology**.

## Objectif

Étudier l'impact de différentes stratégies d'amélioration de la recherche documentaire sur un système RAG :
- **InfoNCE** : apprentissage contrastif du retriever SPLADE ;
- **Hard negatives** : exemples négatifs extraits avec BM25 ;
- **Augmentation de données** : ajout facultatif de questions synthétiques ;
- **Query rewriting** : reformulation des requêtes avant retrieval ;
- **Distillation** : supervision à partir d'un cross-encoder.

Ce dépôt est un **prototype de recherche / expérimentation**, pas une application déployée en production. Les scripts réalisent entraînement et évaluation, et non un service web.

## Architecture

```text
Questions (LoTTE Technology)
       │
       ├── [optionnel] Réécriture de requêtes (SmolLM2)
       ▼
    SPLADE (encodeur sparse)
       │ similarité dot-product
       ▼
    Top-k documents (corpus indexé avec PyTerrier)
       │
       ▼
    Contexte + question
       │
       ▼
    SmolLM2-1.7B-Instruct
       │
       ▼
    Réponse générée
```

**Stack :** Python, PyTorch, Hugging Face Transformers, PyTerrier, pandas, tqdm et Hugging Face Evaluate (BERTScore).

## Installation

Prérequis : **Python 3.10+**, accès internet pour télécharger LoTTE et les modèles Hugging Face. Prévoir plusieurs Go de mémoire (GPU conseillé ; exécution CPU possible mais lente).

```bash
git clone https://github.com/Eliott04/RAG.git
cd RAG
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

> Si vous souhaitez tester les modifications de préparation portfolio avant leur fusion, utilisez `git checkout portfolio/rag-cleanup` après le clone.

## Exécution

Depuis la **racine** du dépôt :

```bash
python -m src.evaluation.eval_Splade
```

Ce script télécharge le dataset, construit l'index PyTerrier, entraîne SPLADE, exécute la chaîne RAG et calcule les métriques. **Attention :** il peut télécharger plusieurs modèles et effectuer un entraînement coûteux (3 époques par défaut dans la branche de nettoyage).

Autres expériences :

```bash
python -m src.evaluation.eval_Splade_augemented
python -m src.evaluation.Distillation
```

L'expérience augmentée **fonctionne sans fichier synthétique** (elle continue avec les qrels). Pour activer cette variante, fournir des paires `{ "query": "...", "doc_id": "..." }` dans `src/evaluation/outputs/practical-05/training_data_for_splade.json`.

Les scripts ne sont pas paramétrés en ligne de commande : ajuster `num_docs`, `num_queries` et `num_epochs` dans le script concerné si nécessaire. Les index temporaires sont placés dans `outputs/` (ignoré par Git).

## Évaluation

- **Recherche :** MRR et Recall@1, @5 et @10 à partir des jugements de pertinence LoTTE.
- **Génération :** BERTScore (precision, recall, F1), sur un ensemble de **20 réponses de référence intégrées aux scripts InfoNCE**.
- **Échantillonnage :** jusqu'à 150 requêtes, dont les 20 dernières sont réservées à l'évaluation dans les scripts principaux.

**Important :** aucune valeur chiffrée de performance n'est revendiquée ici sans exécution reproductible ni journaux de résultats. Les références de génération sont codées en dur et doivent être vérifiées par rapport aux identifiants exacts du jeu de test avant d'interpréter le BERTScore. La distillation utilise une procédure expérimentale distincte et ne calcule actuellement que les métriques de retrieval.

## Organisation du code

```text
src/
├── Splade.py                        # SPLADE, InfoNCE et distillation
├── data.py                          # Préparation LoTTE et hard-negative mining
├── rag.py                           # Retrieval + génération
├── prompting.py                     # Prompts / query rewriting
├── score.py                         # MRR, Recall@k, BERTScore
├── loss.py                          # Fonctions de perte
├── utils.py                         # Gestion des modèles et devices
└── evaluation/
    ├── eval_Splade.py               # Expérience InfoNCE
    ├── eval_Splade_augemented.py     # Augmentation + rewriting
    └── Distillation.py              # Expérience distillation
```

## Limites connues et pistes d'amélioration

- Les essais nécessitent le téléchargement des modèles et du dataset, sans checkpoint ni résultats versionnés.
- La totalité des représentations des documents est stockée en mémoire : convenable pour un prototype, pas pour un index de grande taille.
- Le modèle de génération et l'évaluation BERTScore demandent des ressources matérielles importantes.
- La variante distillation demande une validation fonctionnelle complémentaire ; le teacher est entraîné dans le script.
- Une CI avec tests unitaires, un petit jeu de données de démonstration et une configuration commune des expériences amélioreraient la reproductibilité.

## Vérifications rapides

```bash
python -m compileall -q src
python -m unittest discover -s tests -v
```

La compilation et les tests unitaires ne remplacent **pas** un run complet sur LoTTE.

---

**Statut :** prototype expérimental — projet RAG / Information Retrieval / NLP.
