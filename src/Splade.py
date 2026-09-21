from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from tqdm.auto import tqdm
import pandas as pd

def compute_splade_representation(
    text: str,
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the SPLADE representation of a text.

    Args:
        text: Text to encode
        model: MLM model
        tokenizer: Tokenizer

    Returns:
        Sparse vector of size vocab_size
    """
    # Implement SPLADE representation

    # 2. Get MLM logits
    # 3. Apply ReLU then log(1 + x)
    # 4. Max-pooling over positions
    # 1. Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
       
        outputs = model(**inputs)
        logits = outputs.logits
        activations = torch.log1p(torch.relu(logits))
        splade_vec, _ = torch.max(activations, dim=1)


    return splade_vec.squeeze(0)

class SPLADEEncoder(nn.Module):
    """
    SPLADE encoder based on an MLM model.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        sparsity_weight: float = 0.0001,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.sparsity_weight = sparsity_weight
        self.vocab_size = self.tokenizer.vocab_size

    def encode(
        self,
        texts: List[str],
        max_length: int = 256,
    ) -> torch.Tensor:
        """
        Encode a list of texts into SPLADE representations.

        Returns:
            Tensor of shape (batch_size, vocab_size)
        """

        device = next(self.model.parameters()).device

        # 1. Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        # 2. Forward pass through MLM
        outputs = self.model(**inputs)
        logits = outputs.logits

        # 3. Sparsification: ReLU + log(1 + x)
        activations = torch.log1p(torch.relu(logits))

         # 4. Max-pooling over positions (attention to mask!)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        activations = activations * attention_mask
        splade_rep, _ = torch.max(activations, dim=1)
    

        return splade_rep

    def compute_flops_loss(self, sparse_rep: torch.Tensor) -> torch.Tensor:
        """
        Compute FLOPS penalty to encourage sparsity.

        FLOPS = sum_j (sum_i w_ij)^2
        """
        flops = (sparse_rep.sum(dim=0) ** 2).sum()
        return self.sparsity_weight * flops

    def forward(
        self,
        query_texts: List[str],
        doc_texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        """
        query_rep = self.encode(query_texts)
        doc_rep = self.encode(doc_texts)

        flops_loss = (
            self.compute_flops_loss(query_rep)
            + self.compute_flops_loss(doc_rep)
        )

        return query_rep, doc_rep, flops_loss
    

# Visualize activated terms
def visualize_splade_terms(
    rep: torch.Tensor,
    tokenizer: AutoTokenizer,
    top_k: int = 20,
):
    """Display most activated terms."""
    top_indices = torch.topk(rep, k=top_k).indices
    top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
    top_values = rep[top_indices].tolist()

    print("Activated terms:")
    for token, value in zip(top_tokens, top_values):
        bar = "█" * int(value * 5)
        print(f"  {token:20s} {value:6.3f} {bar}")

def contrastive_loss(
    query_rep: torch.Tensor,
    doc_rep: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Compute InfoNCE contrastive loss.

    Args:
        query_rep: (batch_size, vocab_size)
        doc_rep: (batch_size, vocab_size)
        temperature: Temperature for softmax

    Returns:
        Average InfoNCE loss
    """

    # 1. Compute query-doc similarity scores (dot product)
    scores = torch.matmul(query_rep, doc_rep.T)

    # 2. Scale by temperature
    scores = scores / temperature

    # 3. Targets: positives are on the diagonal
    batch_size = scores.size(0)
    targets = torch.arange(batch_size, device=scores.device)

    # 4. InfoNCE loss (cross-entropy over rows)
    loss = F.cross_entropy(scores, targets)

    return loss

def train_splade(
    encoder: SPLADEEncoder,
    train_data: List[Dict],
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    temperature: float = 0.05,
    use_hard_negatives: bool = False,
):
    """
    Train the SPLADE encoder.

    Args:
        encoder: SPLADE model
        train_data: List of triplets {query, positive, negatives}
        num_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        temperature: Temperature for InfoNCE
        use_hard_negatives: If True, expects triplets with 'negatives' field
    """
    from torch.optim import AdamW

    optimizer = AdamW(encoder.parameters(), lr=learning_rate)
    encoder.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_contrastive = 0
        total_flops = 0
        num_batches = 0

        # Shuffle data
        import random

        shuffled = train_data.copy()
        random.shuffle(shuffled)

        # Create batches
        for i in tqdm(range(0, len(shuffled), batch_size), desc=f"Epoch {epoch + 1}"):
            batch = shuffled[i : i + batch_size]
            if len(batch) < 2:  # Need at least 2 for contrastive
                continue

            if use_hard_negatives:
                # Use triplets with hard negatives
                queries = [ex["query"] for ex in batch]
                positives = [ex["positive"][:500] for ex in batch]
                # Collect hard negatives from all examples in batch
                all_negatives = []
                for ex in batch:
                    all_negatives.extend([n[:500] for n in ex.get("negatives", [])[:2]])

                # Encode queries
                query_rep = encoder.encode(queries)
                # Encode positives + negatives together
                all_docs = positives + all_negatives
                doc_reps = encoder.encode(all_docs)
                positive_rep = doc_reps[: len(positives)]
                negative_rep = doc_reps[len(positives) :]

                # Contrastive loss with in-batch + hard negatives
                cont_loss = contrastive_loss(query_rep, positive_rep, temperature)

                # Add hard negative loss if we have negatives
                if len(negative_rep) > 0:
                    # Compute scores with negatives and ensure they're lower
                    neg_scores = torch.matmul(
                        F.normalize(query_rep, p=2, dim=-1),
                        F.normalize(negative_rep, p=2, dim=-1).T,
                    )
                    # Margin loss: positive scores should be higher than negative
                    hard_neg_loss = F.relu(neg_scores.mean() + 0.2).mean()
                    cont_loss = cont_loss + 0.3 * hard_neg_loss

                flops_loss = encoder.compute_flops_loss(
                    query_rep
                ) + encoder.compute_flops_loss(doc_reps)
            else:
                # Standard in-batch negatives only
                queries = [ex["query"] for ex in batch]
                docs = [ex["positive"][:500] for ex in batch]

                # Forward
                query_rep, doc_rep, flops_loss = encoder(queries, docs)

                # Contrastive loss
                cont_loss = contrastive_loss(query_rep, doc_rep, temperature)

            # Total loss
            loss = cont_loss + flops_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_contrastive += cont_loss.item()
            total_flops += flops_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_cont = total_contrastive / num_batches
        avg_flops = total_flops / num_batches

        print(
            f"Epoch {epoch + 1}: Loss={avg_loss:.4f} (Contrastive={avg_cont:.4f}, FLOPS={avg_flops:.4f})"
        )

    encoder.eval()
    print("Training complete!")



class CrossEncoderTeacher(nn.Module):
    """
    Cross-encoder for scoring query-document pairs.

    Unlike the bi-encoder, the cross-encoder sees query and doc together,
    allowing richer interactions but preventing pre-computation.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)

    def forward(
        self,
        queries: List[str],
        documents: List[str],
    ) -> torch.Tensor:
        """
        Score query-document pairs.

        Args:
            queries: List of queries
            documents: List of documents (same length as queries)

        Returns:
            Relevance scores (batch_size,)
        """
        # Implement cross-encoder forward
        pairs = [q + " [SEP] " + d for q, d in zip(queries, documents)]
        inputs = self.tokenizer(pairs,padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]
        scores = self.classifier(cls_output)

        return scores

def distillation_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute distillation loss (MSE or KL divergence).

    Args:
        student_scores: Student (SPLADE) scores
        teacher_scores: Teacher (cross-encoder) scores
        temperature: Temperature for softening distributions (KL only)

    Returns:
        Distillation loss
    """
    # Simple MSE between scores
    return F.mse_loss(student_scores, teacher_scores.detach())

def retriever_to_results(
    retriever_fn,
    queries_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert a retriever function to a results DataFrame for ir-measures.

    Args:
        retriever_fn: Function that takes a query and returns [(doc_id, score)]
        queries_df: DataFrame with queries

    Returns:
        DataFrame with columns [qid, docno, score]
    """
    all_results = []

    for _, query_row in tqdm(
        queries_df.iterrows(), desc="Retrieving", total=len(queries_df)
    ):
        qid = query_row["qid"]
        query_text = query_row["query"]

        results = retriever_fn(query_text)
        for doc_id, score in results:
            all_results.append({"qid": qid, "docno": doc_id, "score": score})

    return pd.DataFrame(all_results)


def trainSplade_Distillation(
    encoder_student: SPLADEEncoder,
    cross_encoder: CrossEncoderTeacher,
    train_data: List[Dict],
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5
):
    from torch.optim import AdamW

    optimizer = AdamW(encoder_student.parameters(), lr=learning_rate)
    encoder_student.train()
    cross_encoder.eval()
    for param in cross_encoder.parameters():
            param.requires_grad = False
            
    for epoch in range(num_epochs):
        total_loss = 0
        total_flops = 0
        num_batches = 0

        # Shuffle data
        import random

        shuffled = train_data.copy()
        random.shuffle(shuffled)
    
        for i in tqdm(range(0, len(shuffled), batch_size), desc=f"Epoch {epoch+1}"):
            
            
            batch = shuffled[i : i + batch_size]
            if len(batch) < 2:  # Need at least 2 for contrastive
                continue
                
            queries = [ex["query"] for ex in batch]
            docs = [ex["doc"] for ex in batch]

        
            query_rep, doc_rep, flops_loss = encoder_student(queries, docs)

            student_scores = torch.sum(query_rep * doc_rep, dim=1).squeeze(-1)
            with torch.no_grad():
                cross_encoder_scores = cross_encoder(queries,docs).squeeze(-1)
            
            loss = distillation_loss(student_scores,cross_encoder_scores) + flops_loss
            

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_flops += flops_loss.item()
            num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_flops = total_flops / num_batches


        print(
                f"Epoch {epoch+1}: Loss={avg_loss:.4f},FLOPS={avg_flops:.4f})"
            )
        encoder_student.eval()
        print("Training complete!")
