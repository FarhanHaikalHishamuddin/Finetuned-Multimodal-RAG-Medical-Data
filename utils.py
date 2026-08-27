import re
import evaluate
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load Metrics once at the top level
# These stay in memory for all Flask requests
print("--- 📊 Loading Evaluation Metrics in utils.py ---")
bertscore = evaluate.load("bertscore")
rouge = evaluate.load("rouge")
bleu = evaluate.load("sacrebleu")

# Using the same small embedder for fast vector comparison
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def normalize(text):
    if not text: return ""
    # Remove punctuation and lowercase for strict matching
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

def strict_clean_answer(text):
    """Removes 'The answer is' and other filler phrases."""
    if not text: return ""
    text = text.strip()
    
    # If a tag accidentally slipped into the answer, remove it
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
        
    patterns = [
        r"^the answer is[:\s]*",
        r"^answer is[:\s]*",
        r"^it is[:\s]*",
        r"^answer[:\s]*",
        r"^visual analysis indicates[:\s]*",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    
    if text.endswith("."):
        text = text[:-1]
    return text.strip()

def parse_output(text):
    """Splits AI text into Reasoning and Final Answer."""
    if not text: return "", ""
    if "<think>" in text and "</think>" in text:
        parts = text.split("</think>")
        reasoning = parts[0].replace("<think>", "").strip()
        answer = parts[-1].strip()
        return answer, reasoning
    else:
        # Fallback if the model forgot the tags
        clean_text = text.replace("<think>", "").replace("</think>", "").strip()
        return clean_text, "Logic synthesized from visual features."

def calculate_metrics(gen, ref):
    """Calculates all advanced metrics for the comparison report."""
    # Safety check: if AI fails to generate or reference is missing
    if not ref or not gen:
        return {
            "accuracy": 0.0, "bert_f1": 0.0, "bleu": 0.0, 
            "f1_score": 0.0, "cosine": 0.0
        }
    
    # 1. Prep for strict matching (BLEU/ROUGE/Accuracy)
    clean_gen = normalize(strict_clean_answer(gen))
    clean_ref = normalize(ref)
    
    # 2. BERTScore (Semantic meaning - can the AI paraphrase?)
    b_res = bertscore.compute(predictions=[gen], references=[ref], lang="en", model_type="distilbert-base-uncased")
    
    # 3. BLEU & ROUGE (Literal word overlap)
    bl_res = bleu.compute(predictions=[clean_gen], references=[[clean_ref]])
    r_res = rouge.compute(predictions=[clean_gen], references=[clean_ref])
    
    # 4. Cosine Similarity (Vector alignment)
    g_emb = eval_embedder.encode([gen])
    r_emb = eval_embedder.encode([ref])
    cos = cosine_similarity(g_emb, r_emb)[0][0]
    
    return {
        "accuracy": 1.0 if clean_gen == clean_ref else 0.0,
        "bert_f1": round(float(b_res['f1'][0]), 4),
        "bleu": round(float(bl_res['score']), 2),
        "f1_score": round(float(r_res['rougeL']), 4),
        "cosine": round(float(cos), 4)
    }