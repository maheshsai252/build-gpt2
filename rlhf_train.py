import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import wandb
from tqdm import tqdm

# Initialize wandb
wandb.init(project="gpt2-rlhf", name="gpt2-rlhf-run")

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer
model_path = "gpt2_small_wikitext_pretrained_model"  # Path to your pre-trained model
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Reward Model (a simple sentiment classifier in this example)
class RewardModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

reward_model = RewardModel(model.config.n_embd).to(device)
reward_optimizer = Adam(reward_model.parameters(), lr=1e-4)

# Load a dataset for RLHF (using IMDB dataset as an example)
dataset = load_dataset("imdb", split="train[:1000]")  # Using a small subset for demonstration

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# RLHF training loop
num_epochs = 3
for epoch in range(num_epochs):
    total_reward = 0
    for batch in tqdm(tokenized_dataset, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + 20,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
        
        # Get the generated text (excluding the input prompt)
        generated_text = outputs[:, input_ids.shape[1]:]
        
        # Compute reward (sentiment score in this example)
        with torch.no_grad():
            last_hidden_state = model(generated_text).last_hidden_state
            reward = reward_model(last_hidden_state.mean(dim=1)).squeeze()
        
        # Compute PPO loss
        log_probs = model(generated_text).logits.log_softmax(-1)
        selected_log_probs = log_probs.gather(-1, generated_text.unsqueeze(-1)).squeeze(-1)
        policy_loss = -(selected_log_probs * reward).mean()
        
        # Update policy (language model)
        model.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        Adam(model.parameters(), lr=1e-5).step()
        
        # Update reward model (in practice, this would be done separately with human feedback)
        reward_loss = nn.BCELoss()(reward, torch.ones_like(reward) * 0.5)  # Assuming neutral sentiment as target
        reward_optimizer.zero_grad()
        reward_loss.backward()
        reward_optimizer.step()
        
        total_reward += reward.mean().item()
        
        wandb.log({
            "policy_loss": policy_loss.item(),
            "reward_loss": reward_loss.item(),
            "reward": reward.mean().item()
        })
    
    avg_reward = total_reward / len(tokenized_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Reward: {avg_reward:.4f}")
    wandb.log({"epoch": epoch+1, "avg_reward": avg_reward})

# Save the RLHF fine-tuned model
rlhf_model_path = "gpt2_rlhf_finetuned_model"
model.save_pretrained(rlhf_model_path)
tokenizer.save_pretrained(rlhf_model_path)

print("RLHF fine-tuning completed!")
wandb.finish()