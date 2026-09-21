import pyterrier  as pt
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from torch import optim,nn
from src.utils import get_best_device
from src.Splade import SPLADEEncoder,CrossEncoderTeacher,trainSplade_Distillation
from src.score import evaluate_retriever
from src.rag import RAGPipeline
import shutil
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from src.data import create_distillation_dataset
device = get_best_device()




"====================================DATASET=============================================================="
# Load the LoTTE technology dataset via PyTerrier's ir-datasets integration
dataset = pt.get_dataset("irds:lotte/technology/dev/search")
print(f"Dataset loaded: lotte/technology/dev/search")
# Configuration
num_docs = 15000  # Limit documents for practical
num_queries = 150  # Limit queries

# Get queries and qrels first to know which documents are relevant
queries_df = dataset.get_topics().head(num_queries)
queries_df['query'] = queries_df['query'].astype(str).str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
qrels_df = dataset.get_qrels()

# Filter qrels for our queries first
query_ids = set(queries_df["qid"].tolist())
qrels_df = qrels_df[qrels_df["qid"].isin(query_ids)].copy()

# Get the set of relevant document IDs we need
relevant_doc_ids = set(qrels_df["docno"].tolist())

# Load documents as list of dicts (format expected by IterDictIndexer)
# Prioritize relevant ones, then fill with others
corpus = []  # List of {"docno": ..., "text": ...}
seen_ids = set()

for doc in dataset.get_corpus_iter():
    doc_id = doc["docno"]
    if doc_id in relevant_doc_ids:
        corpus.append(doc)
        seen_ids.add(doc_id)
    elif len(corpus) < num_docs:
        corpus.append(doc)
        seen_ids.add(doc_id)

    if len(corpus) >= num_docs and relevant_doc_ids.issubset(seen_ids):
        break

# Filter qrels for documents we actually loaded
loaded_doc_ids = {doc["docno"] for doc in corpus}
qrels_df = qrels_df[qrels_df["docno"].isin(loaded_doc_ids)].copy()

print("Dataset loaded:")
print(f"  - Documents: {len(corpus)}")
print(f"  - Queries: {len(queries_df)}")
print(f"  - Qrels: {len(qrels_df)}")


"===========================INDEXATION BM25======================================================================="

# Create an index with metadata storage for document text
output_dir = Path("./outputs/practical-05")
output_dir.mkdir(parents=True, exist_ok=True)
index_path = (output_dir / "index_lotte").absolute()
if index_path.is_dir():
    shutil.rmtree(index_path)
index_path.mkdir(parents=True, exist_ok=True)

# meta: field_name -> max_length (characters to store)
# meta_reverse: fields to build reverse lookup (docno -> docid)
indexer = pt.IterDictIndexer(
    str(index_path),
    overwrite=True,
    meta={"docno": 50, "text": 4096},
    meta_reverse=["docno"],
    properties={"index.meta.data-source": "fileinmem"},  # Load metadata into memory
)
index_ref = indexer.index(corpus)

print(f"Index created with {len(corpus)} documents (text stored in metadata)")
bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")




"===========================training dataset======================================================================="


# Create training data with hard negatives
num_train_queries = 200

# Create triplets from qrels (original data)
training_distill = create_distillation_dataset(
    queries_df,
    qrels_df,
    index_ref,
    bm25,
    num_hard_negatives=3
)



class RelevanceDataset(Dataset):
    def __init__(self, triplets):
        """
        triplets: Liste de dictionnaires {'query': str, 'doc': str, 'label': float}
        """
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

# Exemple de construction de données (mélange de positifs et négatifs)
train_data = RelevanceDataset(training_distill)
dataloader = DataLoader(train_data, batch_size=16, shuffle=True)







"===========================training======================================================================="
num_epochs = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize cross-encoder
cross_encoder = CrossEncoderTeacher("cross-encoder/ms-marco-MiniLM-L12-v2")
cross_encoder = cross_encoder.to(device)
cross_encoder.train() # Mode entraînement

# Fine_tune cross_encoder
optimizer = optim.AdamW(cross_encoder.parameters(), lr=2e-5) 
criterion = nn.BCEWithLogitsLoss()



for epoch in range(10):
    total_loss = 0
    
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        
        queries = batch['query']
        docs = batch['doc']
        labels = batch['label'].float().to(device) #
        
        # Forward pass
        scores = cross_encoder(queries, docs)
        
        # Calcul de l'erreur
        loss = criterion(scores.squeeze(-1), labels)
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1} | Loss Moyenne: {total_loss / len(dataloader):.4f}")



splade_encoder = SPLADEEncoder("distilbert-base-uncased")
splade_encoder = splade_encoder.to(device)
print(f"SPLADE encoder loaded on {device}")

# Train SPLADE with hard negatives (reduced version for the practical)
num_epochs = 20

# Use training triplets created in Part 1b (with hard negatives)
if training_distill:
    trainSplade_Distillation(
        splade_encoder,
        cross_encoder,
        training_distill,
        num_epochs=num_epochs,
        batch_size=4,
    )
else:
    print("No training triplets available. Skipping SPLADE training.")

    
"===========================RAG======================================================================="   

# Load generation model
#
# Options (uncomment ONE gen_model_name):
#
# 1. SmolLM2-1.7B float16 (~3.4GB) - default, works on all platforms
gen_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)

# Detect if model is pre-quantized (AWQ/GPTQ) by name
is_quantized = "AWQ" in gen_model_name or "GPTQ" in gen_model_name

if is_quantized:
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        device_map="auto",
    )
    print(f"Generation model loaded: {gen_model_name} (pre-quantized)")
else:
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        torch_dtype=torch.float16,
    )
    gen_model = gen_model.to(device)
    print(f"Generation model loaded: {gen_model_name} on {device}")

gen_model.eval()

doc_id_list = [doc["docno"] for doc in corpus]

rag = RAGPipeline(
    retriever=splade_encoder,
    generator=gen_model,
    device = device,
    generator_tokenizer=gen_tokenizer,
    pt_index_ref=index_ref,
    doc_ids=doc_id_list,
    top_k=3,
)

# Evaluate SPLADE
test_queries = queries_df.tail(20)  # Use last queries as test

def splade_retriever(query: str) -> List[Tuple[str, float]]:
    return rag.retrieve(query, top_k=20)


"=======================================EVALUATION================================================================"

splade_metrics = evaluate_retriever(splade_retriever, test_queries, qrels_df)

print("\n=== SPLADE Results ===")
for metric, value in splade_metrics.items():
    print(f"  {metric}: {value:.4f}")