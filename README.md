# Experimental RAG — SPLADE × SmolLM2

**Étude expérimentale de Retrieval-Augmented Generation (RAG)** sur le domaine **Technology** du dataset **LoTTE**. **Eliott Rakotondramanitra et Louis-Antoine Pham** (février 2026).

## Objectif

Comparer des approches d'amélioration d'un pipeline RAG combinant **SPLADE** (recherche documentaire) et **SmolLM2** (génération de réponses) :
- entraînement de SPLADE par **InfoNCE** ou **distillation** ;
- **data augmentation** par génération de questions synthétiques ;
- **query rewriting** avant la recherche de documents.

L'étude mesure la qualité de la recherche avec **MRR / Recall@k** et celle de la génération avec **BERTScore**, sur **20 questions de test**. Le corpus expérimental comprend environ **15 000 documents** et **150 questions**.

## Résultats du rapport

| Méthode | MRR | Recall@10 |
| --- | ---: | ---: |
| InfoNCE | 0,2836 | 0,4500 |
| InfoNCE + augmentation | **0,5083** | **0,6458** |
| InfoNCE + augmentation + rewriting | 0,2419 | 0,5000 |
| Distillation | 0,0167 | 0,0167 |
| BM25 (référence) | 0,7408 | 0,8417 |

**Conclusion :** l'augmentation par requêtes synthétiques améliore nettement SPLADE par rapport à InfoNCE seul. L'expansion des requêtes réduit les performances dans cette expérience ; la distillation n'a pas donné de résultats exploitables. **BM25 reste la meilleure référence** sur ce protocole.

Pour la génération, le rapport indique un **BERTScore F1 de 0,5796** avec augmentation, contre **0,5619** après rewriting. Ces chiffres sont **tirés du rapport**, et non d'une nouvelle reproduction des expériences.

## Technologies et code

**Python · PyTorch · Transformers · PyTerrier · SPLADE · SmolLM2 · LoTTE · BERTScore**

- `src/Splade.py`, `src/rag.py` : entraînement du retriever et pipeline retrieval → génération.
- `src/data.py`, `src/score.py` : données, BM25, métriques.
- `src/evaluation/` : expériences InfoNCE, augmentation / rewriting et distillation.
