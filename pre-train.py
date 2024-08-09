import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import wandb
from tqdm import tqdm
import random

# Initialize wandb
wandb.init(project="gpt2-pretraining", name="gpt2-small-wikitext-run")

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model configuration (small GPT-2)
config = GPT2Config(
    vocab_size=50257,
    n_positions=512,
    n_ctx=512,
    n_embd=384,
    n_layer=6,
    n_head=6
)

# Initialize model and tokenizer
model = GPT2LMHeadModel(config).to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add padding token to tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Load Wikitext-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")

# Function to create a smaller dataset
def create_small_dataset(dataset, num_samples=1000, min_length=50):
    small_dataset = []
    for item in dataset:
        text = item['text']
        if len(text) >= min_length:
            small_dataset.append(text)
        if len(small_dataset) >= num_samples:
            break
    return small_dataset

# Create small dataset
small_dataset = create_small_dataset(dataset)
print(f"Small dataset size: {len(small_dataset)} samples")

# Custom dataset class
class SmallWikitextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        return {
            'input_ids': encodings.input_ids.squeeze(),
            'attention_mask': encodings.attention_mask.squeeze()
        }

# Create dataset and dataloader
dataset = SmallWikitextDataset(small_dataset, tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training settings
num_epochs = 5
num_training_steps = num_epochs * len(dataloader)
num_warmup_steps = num_training_steps // 10

# Initialize optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

# Training loop
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
        
        wandb.log({
            "batch_loss": loss.item(),
            "learning_rate": scheduler.get_last_lr()[0]
        })
    
    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")
    wandb.log({"epoch": epoch+1, "avg_epoch_loss": avg_epoch_loss})
    
    # Save checkpoint
    checkpoint_path = f"gpt2_small_wikitext_checkpoint_epoch_{epoch+1}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss,
    }, checkpoint_path)
    wandb.save(checkpoint_path)

# Save the final model and tokenizer
model_save_path = "gpt2_small_wikitext_pretrained_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("Pre-training completed!")
wandb.finish()