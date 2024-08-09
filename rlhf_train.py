import torch
from torch.optim import Adam
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import wandb
from textblob import TextBlob

# Initialize wandb
wandb.init(project="gpt2-rlhf", name="rlhf-run-1")

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2_small_wikitext_pretrained_model")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load a small dataset (e.g., a subset of WikiText)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")

# Define reward model (unchanged)
class RewardModel:
    def __init__(self, target_length=50, keywords=["informative", "clear", "concise"]):
        self.target_length = target_length
        self.keywords = keywords

    def calculate_reward(self, text):
        # ... (reward calculation logic remains the same)
        pass

reward_model = RewardModel()

# RLHF training loop
optimizer = Adam(model.parameters(), lr=1e-5)

for epoch in range(5):  # Adjust number of epochs as needed
    total_reward = 0
    for i, example in enumerate(dataset):
        input_text = example['text']
        
        # Tokenize input with debugging
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        print(f"Input text: {input_text}")
        print(f"Tokenized input shape: {input_ids.shape}")
        
        # Skip empty inputs
        if input_ids.numel() == 0:
            print(f"Skipping empty input at index {i}")
            continue
        
        try:
            # Generate output
            output = model.generate(input_ids, max_length=100, num_return_sequences=1)
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Calculate reward using the reward model
            reward = reward_model.calculate_reward(output_text)
            
            # Backward pass
            loss = -reward.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_reward += reward.item()
            
            # Log to wandb
            wandb.log({
                "reward": reward.item(),
                "loss": loss.item(),
                "epoch": epoch,
                "step": i
            })
        except RuntimeError as e:
            print(f"Error processing input at index {i}: {str(e)}")
            continue
    
    print(f"Epoch {epoch}, Average Reward: {total_reward / len(dataset)}")

# Save the fine-tuned model
model.save_pretrained("gpt2_rlhf_finetuned")

# Finish wandb run
wandb.finish()