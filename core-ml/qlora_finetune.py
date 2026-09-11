import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import wandb

# Hardcoded purely for script execution defaults; production uses EnvironmentConfig
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models", "symantix-phi3-qlora")

def format_instruction(sample):
    """Formats the instruction for the SFTTrainer"""
    return f"<s>[INST] {sample['instruction']} [/INST]\n{sample['response']}</s>"

def train():
    wandb.init(project="symantix-slm", name="phi3-qlora-4bit")

    print("Configuring 4-bit Quantization via bitsandbytes...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading Base Model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model.config.use_cache = False
    # Explicit Memory Management Requirement
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    print("Configuring LoRA Targets...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    dataset_path = os.path.join(os.path.dirname(__file__), "sre_instructions.jsonl")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=100,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="instruction", # We will format using formatting_func
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=lambda x: [format_instruction(item) for item in x] if isinstance(x, list) else [format_instruction(x)]
    )

    print("Executing QLoRA SFTTrainer loop...")
    trainer.train()

    print("Exporting Merged Weights...")
    # Explicit cache clearing before merge
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    wandb.finish()
    print(f"Fine-Tuning complete. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
