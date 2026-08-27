import torch
from datasets import load_dataset
from transformers import Qwen2_VLForConditionalGeneration, AutoProcessor, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

# 1. Configuration
DATASET_PATH = "train_conversational_25percent.jsonl"
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
OUTPUT_DIR = "./reinforced_medical_model"

def run_retraining_cycle():
    print(f"🚀 Starting Retraining Cycle using: {DATASET_PATH}")

    # 2. Load the WHOLE dataset (Original + Expert Feedback)
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # 3. Load Model and Processor
    model = Qwen2_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # 4. Apply LoRA (The "Reinforcement" layer)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3, 
        save_strategy="epoch",
        logging_steps=10,
        fp16=True,
        report_to="none"
    )

    # 6. Initialize SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        dataset_text_field="text", 
    )

    # 7. Execute Retraining
    trainer.train()

    # 8. Save the Reinforced Brain
    trainer.save_model(OUTPUT_DIR)
    print(f"✅ SUCCESS: Model reinforced and saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_retraining_cycle()