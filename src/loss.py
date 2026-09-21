import torch
import torch.nn.functional as F

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
        Average loss
    """
    # Implement contrastive loss
    scores = torch.matmul(query_rep, doc_rep.T) / temperature
    labels = torch.arange(scores.size(0)).to(scores.device)
    loss = F.cross_entropy(scores, labels)

    return loss

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
        temperature: Temperature for softening distributions

    Returns:
        Distillation loss
    """
    # Simple MSE between scores
    return F.mse_loss(student_scores, teacher_scores.detach())