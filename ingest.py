import json
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_PATH = "train_conversational_25percent.jsonl"
INDEX_PATH = "medical_db.index"
METADATA_PATH = "medical_metadata.json"
# Using a high-quality medical-friendly embedder
EMBEDDER_ID = "all-mpnet-base-v2" 

def ingest_medical_data():
    # 1. Initialize Embedder
    print(f"--- Loading Embedder: {EMBEDDER_ID} ---")
    embedder = SentenceTransformer(EMBEDDER_ID)
    
    rag_corpus = [] 
    rag_queries = []

    # 2. Process Dataset
    print(f"--- Processing Dataset: {DATASET_PATH} ---")
    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: File {DATASET_PATH} not found!")
        return

    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Processing Metadata"):
            try:
                item = json.loads(line)
                messages = item.get('messages', [])
                
                # --- EXTRACT QUESTION (The Search Query) ---
                user_content = messages[0]['content']
                q_text = next((x['text'] for x in user_content if x.get('type') == 'text'), "")
                
                # --- EXTRACT ASSISTANT CONTENT (The Clinical Evidence) ---
                # This pulls the full block containing <think>Reasoning</think>Answer
                gt_content = messages[1]['content'][0]['text']
                
                if not q_text or not gt_content:
                    continue

                # --- IMPLEMENTATION ---
                formatted_entry = gt_content 
                
                rag_queries.append(q_text)
                rag_corpus.append(formatted_entry)
                
            except Exception as e:
                continue

    # 3. Create Embeddings
    print(f"--- Encoding {len(rag_queries)} clinical questions... ---")
    corpus_embeddings = embedder.encode(rag_queries, convert_to_numpy=True, show_progress_bar=True)

    # 4. Build FAISS Index (Using Cosine Similarity)
    print("--- Building FAISS Index ---")
    faiss.normalize_L2(corpus_embeddings)
    
    d = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(d) 
    index.add(corpus_embeddings.astype('float32'))

    # 5. Save Everything
    print(f"--- Saving Files ---")
    faiss.write_index(index, INDEX_PATH)
    
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        # Saving the list of <think>Reasoning</think>Answer strings
        json.dump(rag_corpus, f, ensure_ascii=False, indent=2)

    print(f" RAG Index Built with {index.ntotal} clinical cases.")
    print(f"Format stored: Captured Reasoning Tags from Dataset")

if __name__ == "__main__":
    ingest_medical_data()