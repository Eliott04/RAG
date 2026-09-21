import pyterrier  as pt
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from src.utils import get_best_device
from src.Splade import SPLADEEncoder,train_splade
from src.score import evaluate_retriever,evaluate_generation
from src.rag import RAGPipeline
import shutil
from tqdm.auto import tqdm
from pathlib import Path
from typing import List, Dict, Tuple
from src.data import create_training_triplets

device = get_best_device()

"==============================================================================================================================================="
generated_gold = [
"Yes, placing the pagefile on an SSD is recommended for the best performance as SSDs handle random reads and sequential writes well. While it may shorten the SSD's lifespan due to frequent writes, it increases stability compared to disabling it, even with large RAM.",
"iGPU Multi-Monitor is a feature that keeps the integrated graphics enabled even when a dedicated graphics card is installed. This allows the use of motherboard video ports for additional monitors while the dedicated GPU handles rendering.",
"In layman terms, they are essentially the same concept for parallel processing, with the naming difference being commercial branding for NVIDIA and AMD respectively. However, they differ architecturally: NVIDIA CUDA cores are generally bigger, more complex, and run at higher frequencies, while AMD stream processors are smaller and simpler.",
"They are conceptually the same, representing the parallel processing units for NVIDIA and AMD respectively. However, they cannot be directly equated due to architectural differences, with CUDA cores being larger and more complex compared to the smaller AMD stream processors.",
"eCryptfs works by storing encrypted files in a private directory (usually `~/.Private`) and using a configuration directory to manage mounting. Upon login, a PAM module uses the login passphrase to unwrap a mount passphrase, add it to the kernel keyring, and automatically mount the decrypted home directory.",
"This is often a software issue where the OS fails to detect a key release, potentially caused by driver conflicts. Solutions include pressing both Ctrl keys simultaneously, using Ctrl+Alt+Del, pressing the Fn key, or using software like SharpKeys to disable the stuck key.",
"The default time zone for IIS logs is UTC. However, IIS Manager allows configuration to save logs in the local time format instead.",
"Yes, the default time for IIS logs is in UTC.",
"Functionally, they are 100% the same thing. \"httpd\" is simply the program name often used on RedHat/CentOS for the Apache Web server, whereas Ubuntu/Debian use \"apache2\".",
"Yes, they are functionally the same; \"httpd\" is the binary name often used (e.g., on RedHat) for the Apache Web server.",
"VGA connectors have screws to prevent the plug from falling out of the socket, as the cables are relatively heavy and easily knocked loose.",
 "Yes, RJ45 connectors work with Cat6 cable because the standard is backward compatible. However, using Cat6 specific connectors is recommended to ensure the best speed and avoid limiting the connection to the slowest link.",
"A private network is typically a physical network within a single location controlled by one entity. A VPN (Virtual Private Network) connects devices or networks across a larger network (like the internet) via a tunnel, making them appear as if they are on the same private network.",
"In a segmented memory model, memory is divided into sections, requiring a segment value and an offset to access a location. In a flat memory model, memory is treated as a single continuous space where a program sees a linear address space (from zero to the top), which is the standard for modern operating systems.",
"`who` displays basic information about users currently logged onto the system. `finger` provides detailed personal information (real name, phone, etc.) and can query users on remote networks.",
"\"Size\" is the actual byte count of the file, while \"Size on disk\" is the space occupied based on allocation units (clusters); if a file isn't a multiple of the cluster size, unused space fills the rest. Differences also arise from compression, hardlinks, or offline files that haven't been downloaded yet.",
 "VOB files are large because they contain the actual video and audio data. Their sizes vary due to variable bit length encoding used by video codecs.",
"MemFree indicates the amount of physical RAM, in kilobytes, that is currently unused by the system.",
"Yes, PGP can generally open GPG files because both are implementations of the same OpenPGP standard.",
"Yes, they are compatible as they both implement the OpenPGP standard, though slight differences in supported algorithms or legacy formats may exist."
]

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

splade_encoder = SPLADEEncoder("distilbert-base-uncased")
splade_encoder = splade_encoder.to(device)
print(f"SPLADE encoder loaded on {device}")

# Create triplets from qrels (original data)
training_triplets = create_training_triplets(
    queries_df,
    qrels_df,
    index_ref,
    bm25,
    num_hard_negatives=1,
    max_queries=num_train_queries,
)

print(f"\nTraining triplets from qrels: {len(training_triplets)}")
# Use training triplets created in Part 1b (with hard negatives)


"===========================training======================================================================="
num_epochs = 20

if training_triplets:
    print(f"Training on {len(training_triplets)} triplets with hard negatives...")
    train_splade(
        splade_encoder,
        training_triplets,
        num_epochs=num_epochs,
        batch_size=4,
        use_hard_negatives=True,
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
    generator_tokenizer=gen_tokenizer,
    pt_index_ref=index_ref,
    doc_ids=doc_id_list,
    top_k=3,
    device = device
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
    
answer = []
for qid,query in zip(test_queries["qid"],test_queries["query"]):
   answer.append(rag(query)['generated_answer'])
    
generation_metrics = evaluate_generation(generated_gold,answer)

print("\n===Generation Results ===")
for metric, value in generation_metrics.items():
    print(f"  {metric}: {value:.4f}")
