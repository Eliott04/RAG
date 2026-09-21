import torch
from tqdm.auto import tqdm
from src.Splade import SPLADEEncoder
from src.data import get_doc_text,get_text_from_index
from typing import List, Dict, Tuple


class RAGPipeline:
    """
    Complete RAG pipeline: Retrieval + Generation.

    Uses PyTerrier index for document storage (scalable to large collections).
    """

    def __init__(
        self,
        retriever: SPLADEEncoder,
        generator,
        device,
        generator_tokenizer,
        pt_index_ref,
        doc_ids: List[str],
        top_k: int = 3,
        
    ):
        self.retriever = retriever
        self.generator = generator
        self.gen_tokenizer = generator_tokenizer
        self.pt_index_ref = pt_index_ref
        self.doc_ids = doc_ids
        self.top_k = top_k
        self.device = device
        
        # Pre-compute document representations
        self._index_documents()

    def _get_doc_text(self, doc_id: str) -> str:
        """Get document text from PyTerrier index."""
        return get_doc_text(self.pt_index_ref, doc_id)

    def _get_texts(self, doc_ids: List[str]) -> Dict[str, str]:
        """Get multiple document texts from PyTerrier index."""
        return get_text_from_index(self.pt_index_ref, doc_ids)

    def _index_documents(self, batch_size: int = 16):
        """Index all documents with SPLADE."""
        print("Indexing documents with SPLADE...")

        # Get texts from PyTerrier index in batches
        all_reps = []

        with torch.no_grad():
            for i in tqdm(range(0, len(self.doc_ids), batch_size)):
                batch_ids = self.doc_ids[i : i + batch_size]
                batch_texts = self._get_texts(batch_ids)
                # Truncate texts and maintain order
                texts = [batch_texts.get(doc_id, "")[:500] for doc_id in batch_ids]
                reps = self.retriever.encode(texts)
                all_reps.append(reps.cpu())

        self.doc_reps = torch.cat(all_reps, dim=0)
        print(f"Documents indexed: {len(self.doc_ids)}")

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Retrieve most relevant documents.

        Args:
            query: User question
            top_k: Number of documents to retrieve

        Returns:
            List of (doc_id, score)
        """
        if top_k is None:
            top_k = self.top_k

        # Encode query
        with torch.no_grad():
            query_rep = self.retriever.encode([query]).cpu()

        # Compute scores
        scores = torch.matmul(query_rep, self.doc_reps.T).squeeze(0)

        # Top-k
        top_indices = torch.topk(scores, k=min(top_k, len(scores))).indices

        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = scores[idx].item()
            results.append((doc_id, score))

        return results

    def generate(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 200,
    ) -> str:
        """Generate a response based on context."""
        system_msg = "You are a helpful assistant that answers technical questions using the provided context."
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        prompt = self.gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.gen_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.gen_tokenizer.eos_token_id,
            )

        response = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()

        return response

    def __call__(self, query: str) -> Dict:
        """
        Complete pipeline: retrieval + generation.

        Args:
            query: User question

        Returns:
            Dict with retrieved_docs and generated_answer
        """
        # Implement RAG pipeline
        retrieved = self.retrieve(query)
        doc_texts = self._get_texts([doc_id for doc_id, _ in retrieved])
        context = "\n\n".join([doc_texts[doc_id][:300] for doc_id, _ in retrieved])
        answer = self.generate(query, context)


        return {
            "query": query,
            "retrieved_docs": retrieved,
            "generated_answer": answer,
            "context": context
        }
