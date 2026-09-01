import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoModel
from typing import Iterator, Dict
import wandb

from dataset_generator import generate_triplets

# Environment Configuration
MODEL_NAME = "BAAI/bge-small-en-v1.5"
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
EPOCHS = 3
MARGIN = 1.0

class TripletIterableDataset(IterableDataset):
    """
    Iterable dataset utilizing python generators to stream data directly into the model,
    preventing Out-Of-Memory (OOM) failures for massive log datasets.
    """
    def __init__(self, log_file_path: str, tokenizer: AutoTokenizer):
        self.log_file_path = log_file_path
        self.tokenizer = tokenizer

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for triplet in generate_triplets(self.log_file_path):
            encoded = self.tokenizer(
                [triplet["anchor"], triplet["positive"], triplet["negative"]],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            yield {
                "anchor_ids": encoded["input_ids"][0],
                "anchor_mask": encoded["attention_mask"][0],
                "positive_ids": encoded["input_ids"][1],
                "positive_mask": encoded["attention_mask"][1],
                "negative_ids": encoded["input_ids"][2],
                "negative_mask": encoded["attention_mask"][2],
            }

def mean_pooling(model_output, attention_mask):
    """Perform mean pooling on token embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def train_model(log_file_path: str):
    """
    Core PyTorch training loop utilizing TripletMarginLoss.
    Explicitly handles GPU cache clearing to ensure memory safety.
    """
    wandb.init(project="symantix-embeddings", name="bge-small-triplet")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    
    dataset = TripletIterableDataset(log_file_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.TripletMarginLoss(margin=MARGIN, p=2)
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        steps = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Move to device
            a_ids, a_mask = batch["anchor_ids"].to(device), batch["anchor_mask"].to(device)
            p_ids, p_mask = batch["positive_ids"].to(device), batch["positive_mask"].to(device)
            n_ids, n_mask = batch["negative_ids"].to(device), batch["negative_mask"].to(device)
            
            # Forward pass
            a_out = model(input_ids=a_ids, attention_mask=a_mask)
            p_out = model(input_ids=p_ids, attention_mask=p_mask)
            n_out = model(input_ids=n_ids, attention_mask=n_mask)
            
            a_emb = mean_pooling(a_out, a_mask)
            p_emb = mean_pooling(p_out, p_mask)
            n_emb = mean_pooling(n_out, n_mask)
            
            # Normalize embeddings
            a_emb = F.normalize(a_emb, p=2, dim=1)
            p_emb = F.normalize(p_emb, p=2, dim=1)
            n_emb = F.normalize(n_emb, p=2, dim=1)
            
            # Pillar A: Zero-Hallucination & Math Verification
            assert a_emb.shape[1] == 384, f"Anchor embedding dimension mismatch. Expected 384, got {a_emb.shape[1]}"
            assert p_emb.shape[1] == 384, f"Positive embedding dimension mismatch. Expected 384, got {p_emb.shape[1]}"
            assert n_emb.shape[1] == 384, f"Negative embedding dimension mismatch. Expected 384, got {n_emb.shape[1]}"
            
            loss = criterion(a_emb, p_emb, n_emb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
            wandb.log({"batch_loss": loss.item()})
            
            # Explicit memory management
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        avg_loss = total_loss / max(1, steps)
        print(f"Epoch {epoch+1}/{EPOCHS} - Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch+1, "avg_loss": avg_loss})
        
    print("Training complete. Exporting model weights.")
    save_path = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, required=True, help="Path to the JSONL simulated logs")
    args = parser.parse_args()
    
    train_model(args.log_file)
