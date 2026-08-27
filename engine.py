import torch
import json
import faiss
import os
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_ID = "sarnsrun/qwen_2.5_VL-7B-Instruct_PMC-VQA_cp800"
EMBEDDER_ID = "all-mpnet-base-v2"

# === 1. ORIGINAL DB (PMC-VQA) ===
INDEX_FILE = "medical_db.index"
METADATA_FILE = "medical_metadata.json"

# === 2. NEW DB (MedQuAD) ===
TEXT_INDEX_FILE = "medquad.index"
TEXT_METADATA_FILE = "medquad_metadata.json"

print("--- 🔄 LOADING HYBRID RAG ENGINE ---")

# 1. Load Embedder
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   (Embedder running on: {device})")
embedder = SentenceTransformer(EMBEDDER_ID, device=device)

# 2. Load ORIGINAL Visual Database (PMC-VQA)

print("   Loading Visual DB (PMC-VQA)...")
if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
    index = faiss.read_index(INDEX_FILE)
    # Move to GPU if available for speed
    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
else:
    print("   ⚠️ Warning: Visual DB files not found.")
    index, metadata = None, []

# 3. Load NEW Text Database (MedQuAD)
print("   Loading Text DB (MedQuAD)...")
if os.path.exists(TEXT_INDEX_FILE) and os.path.exists(TEXT_METADATA_FILE):
    text_index = faiss.read_index(TEXT_INDEX_FILE)
    if device == "cuda":
        res_text = faiss.StandardGpuResources()
        text_index = faiss.index_cpu_to_gpu(res_text, 0, text_index)
        
    with open(TEXT_METADATA_FILE, 'r', encoding='utf-8') as f:
        text_metadata = json.load(f)
else:
    print("   ⚠️ Warning: MedQuAD Text DB files not found. (Did you run build_text_index.py?)")
    text_index, text_metadata = None, []

# 4. Load Model (No Changes Here)
print(f"   Loading Qwen Model...")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    device_map="cuda", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model = model.merge_and_unload()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

print("✅ AI Engine Ready.")

# --- RETRIEVAL LOGIC ---
def retrieve_context(query, has_image=False, k=1):
    """
    Retrieves context based on whether the user provided an image.
    - If Image provided: Uses ORIGINAL logic (PMC-VQA).
    - If Text Only: Uses NEW logic (MedQuAD).
    """
    if not query: return []
    
    # Encode query (Same for both)
    query_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    query_vec = query_vec.astype('float32')
    
    # === PATH A: USER HAS IMAGE (ORIGINAL LOGIC) ===
    if has_image and index is not None:
        print(f"   🔍 Querying VISUAL Database (PMC-VQA) for: {query[:30]}...")
        distances, indices = index.search(query_vec, k)
        
        results = []
        for idx in indices[0]:
            if idx < len(metadata):
                # Using your exact original retrieval style
                results.append(metadata[idx]['answer'])
        return results

    # === PATH B: TEXT ONLY (NEW MEDQUAD LOGIC) ===
    elif not has_image and text_index is not None:
        print(f"   🔍 Querying TEXT Database (MedQuAD) for: {query[:30]}...")
        distances, indices = text_index.search(query_vec, k)
        
        results = []
        for idx in indices[0]:
            if idx < len(text_metadata):
                entry = text_metadata[idx]
                # Format: Combine Question + Answer for better context
                # (Since MedQuAD is text, we give the model both the Q and A to learn from)
                context_chunk = f"Related Medical Fact:\nQuestion: {entry.get('question')}\nAnswer: {entry.get('answer')}"
                results.append(context_chunk)
        return results

    return ["No context available."]