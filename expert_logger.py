import json
import os

# This is your master dataset file path
DATASET_PATH = 'train_conversational_25percent.jsonl'

def log_expert_correction(image_name, question, expert_answer):
    new_entry = {
        "messages": [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image_name}, 
                    {"type": "text", "text": question}
                ]
            }, 
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": f"<think>\nExpert Verified Analysis: {expert_answer}\n</think>\n\n{expert_answer}"}
                ]
            }
        ]
    }

    try:
        # 2. Append the new knowledge to the end of the "Whole Dataset"
        with open(DATASET_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(new_entry) + "\n")
        
        print(f"--- [REINFORCEMENT LOG] New knowledge added: {question[:30]}... ---")
        return True
    except Exception as e:
        print(f"--- [REINFORCEMENT ERROR] Failed to save: {e} ---")
        return False

def get_dataset_stats():
    if not os.path.exists(DATASET_PATH):
        return 0
    with open(DATASET_PATH, "r") as f:
        return sum(1 for line in f)