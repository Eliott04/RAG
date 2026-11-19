# Experimental RAG — SPLADE × SmolLM2

**Experimental study of Retrieval-Augmented Generation (RAG)** on the **Technology** domain of the **LoTTE** dataset. **Eliott Rakotondramanitra and Louis-Antoine Pham** (February 2026).

## Objective

Compare approaches to improving a RAG pipeline combining **SPLADE** (document retrieval) and **SmolLM2** (answer generation):
- training SPLADE using **InfoNCE** or **distillation**;
- **data augmentation** through synthetic question generation;
- **query rewriting** before document retrieval.

The study measures retrieval quality using **MRR / Recall@k** and generation quality using **BERTScore**, on **20 test questions**. The experimental corpus contains approximately **15,000 documents** and **150 questions**.

## Results from the Report

| Method | MRR | Recall@10 |
| --- | ---: | ---: |
| InfoNCE | 0.2836 | 0.4500 |
| InfoNCE + augmentation | **0.5083** | **0.6458** |
| InfoNCE + augmentation + rewriting | 0.2419 | 0.5000 |
| Distillation | 0.0167 | 0.0167 |
| BM25 (baseline) | 0.7408 | 0.8417 |

**Conclusion:** augmentation with synthetic queries significantly improves SPLADE compared with InfoNCE alone. Query expansion reduces performance in this experiment; distillation did not produce usable results. **BM25 remains the best baseline** under this protocol.

For generation, the report indicates a **BERTScore F1 of 0.5796** with augmentation, compared with **0.5619** after rewriting. These figures are **taken from the report**, not from a new reproduction of the experiments.

## Technologies and Code

**Python · PyTorch · Transformers · PyTerrier · SPLADE · SmolLM2 · LoTTE · BERTScore**

- `src/Splade.py`, `src/rag.py`: retriever training and retrieval → generation pipeline.
- `src/data.py`, `src/score.py`: data, BM25, metrics.
- `src/evaluation/`: InfoNCE, augmentation / rewriting, and distillation experiments.
