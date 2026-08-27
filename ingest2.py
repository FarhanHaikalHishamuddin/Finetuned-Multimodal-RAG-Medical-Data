import json
import faiss
import os
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
INPUT_FILE = "medquad.jsonl"
OUTPUT_INDEX = "medquad.index"
OUTPUT_META = "medquad_metadata.json"
MODEL_ID = "all-mpnet-base-v2"

def load_dataset(filepath):
    """Smart loader that handles both JSON Lists and JSONL lines."""
    print(f"--- 🕵️ Inspecting {filepath} ---")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # ATTEMPT 1: Try reading as a Standard JSON List [...]
    # This handles the case where everything is on one line wrapped in brackets.
    try:
        data = json.loads(content)
        if isinstance(data, list):
            print(f"   ✅ Detected JSON LIST format (found {len(data)} items).")
            return data
    except json.JSONDecodeError:
        pass # Not a list, let's try lines

    # ATTEMPT 2: Try reading as JSONL (Line-by-Line)
    # This handles normal .jsonl files
    lines = content.split('\n')
    data = []
    print(f"   ℹ️ Trying Line-by-Line parsing...")
    
    for i, line in enumerate(lines):
        if not line.strip(): continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            if i < 3: print(f"   ⚠️ Warning: Could not parse line {i+1}")
            
    if len(data) > 0:
        print(f"   ✅ Detected JSONL format (found {len(data)} lines).")
        return data

    return []

def build_index():
    if not os.path.exists(INPUT_FILE):
        print("❌ Error: Input file not found!")
        return

    # 1. Load Data using the smart loader
    raw_data = load_dataset(INPUT_FILE)
    
    if len(raw_data) < 2:
        print("❌ Error: Still only found 0 or 1 entry. The file format might be corrupted.")
        return

    corpus = []
    texts_to_embed = []
    
    print("--- 🧹 Extracting Questions & Answers ---")
    for entry in raw_data:
        # Flexible key extraction
        q = entry.get('question') or entry.get('input') or entry.get('text')
        a = entry.get('answer') or entry.get('output') or entry.get('response')
        
        if q and a:
            texts_to_embed.append(q)
            corpus.append({"question": q, "answer": a})

    print(f"--- 📊 Ready to process {len(corpus)} valid entries ---")

    # 2. Generate Embeddings
    print(f"--- 🧠 Generating Embeddings... ---")
    embedder = SentenceTransformer(MODEL_ID)
    embeddings = embedder.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
    
    # 3. Build FAISS Index
    print("--- 🏗️ Building FAISS Index ---")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # 4. Save
    print(f"--- 💾 Saving to disk... ---")
    faiss.write_index(index, OUTPUT_INDEX)
    
    with open(OUTPUT_META, 'w', encoding='utf-8') as f:
        json.dump(corpus, f)
        
    print("✅ DONE! Database is rebuilt correctly.")

if __name__ == "__main__":
    build_index()