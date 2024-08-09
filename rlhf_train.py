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

# Define reward model
class RewardModel:
    def __init__(self, target_length=50, keywords=["informative", "clear", "concise"]):
        self.target_length = target_length
        self.keywords = keywords

    def calculate_reward(self, text):
        # 1. Length reward
        length = len(text.split())
        length_reward = max(0, 1 - abs(length - self.target_length) / self.target_length)

        # 2. Keyword reward
        keyword_count = sum(1 for keyword in self.keywords if keyword.lower() in text.lower())
        keyword_reward = keyword_count / len(self.keywords)

        # 3. Sentiment reward (encourage neutral to positive sentiment)
        sentiment = TextBlob(text).sentiment.polarity
        sentiment_reward = (sentiment + 1) / 2  # Map [-1, 1] to [0, 1]

        # Combine rewards (you can adjust weights as needed)
        total_reward = (length_reward + keyword_reward + sentiment_reward) / 3

        return torch.tensor(total_reward)

reward_model = RewardModel()

# RLHF training loop
optimizer = Adam(model.parameters(), lr=1e-5)

for epoch in range(5):  # Adjust number of epochs as needed
    total_reward = 0
    for i, example in enumerate(dataset):
        input_text = example['text']
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
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
    
    print(f"Epoch {epoch}, Average Reward: {total_reward / len(dataset)}")

# Save the fine-tuned model
model.save_pretrained("gpt2_rlhf_finetuned")

# Finish wandb run
wandb.finish()