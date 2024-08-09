import torch
from torch.optim import Adam
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import wandb
from textblob import TextBlob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # Length reward
        length = len(text.split())
        length_reward = max(0, 1 - abs(length - self.target_length) / self.target_length)

        # Keyword reward
        keyword_count = sum(1 for keyword in self.keywords if keyword.lower() in text.lower())
        keyword_reward = keyword_count / len(self.keywords)

        # Sentiment reward (encourage neutral to positive sentiment)
        sentiment = TextBlob(text).sentiment.polarity
        sentiment_reward = (sentiment + 1) / 2  # Map [-1, 1] to [0, 1]

        # Combine rewards
        total_reward = (length_reward + keyword_reward + sentiment_reward) / 3

        return torch.tensor(total_reward)

reward_model = RewardModel()

# RLHF training loop
optimizer = Adam(model.parameters(), lr=1e-5)

def safe_decode(tokens, tokenizer):
    try:
        # Filter out None values and convert to list of integers
        valid_tokens = [int(t) for t in tokens if t is not None]
        return tokenizer.decode(valid_tokens, skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error in decoding: {str(e)}")
        return ""

for epoch in range(5):  # Adjust number of epochs as needed
    total_reward = 0
    valid_examples = 0
    for i, example in enumerate(dataset):
        input_text = example['text']
        
        # Tokenize input with debugging
        try:
            input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            logger.info(f"Input text: {input_text[:50]}...")  # Log first 50 chars
            logger.info(f"Tokenized input shape: {input_ids.shape}")
        except Exception as e:
            logger.error(f"Error tokenizing input at index {i}: {str(e)}")
            continue
        
        # Skip empty inputs
        if input_ids.numel() == 0:
            logger.warning(f"Skipping empty input at index {i}")
            continue
        
        try:
            # Generate output
            output = model.generate(input_ids, max_length=100, num_return_sequences=1)
            output_text = safe_decode(output[0], tokenizer)
            
            if not output_text:
                logger.warning(f"Empty output generated at index {i}")
                continue
            
            # Calculate reward using the reward model
            reward = reward_model.calculate_reward(output_text)
            
            # Backward pass
            loss = -reward.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_reward += reward.item()
            valid_examples += 1
            
            # Log to wandb
            wandb.log({
                "reward": reward.item(),
                "loss": loss.item(),
                "epoch": epoch,
                "step": i
            })
            
            if i % 100 == 0:  # Log every 100 steps
                logger.info(f"Processed {i} examples in epoch {epoch}")
        
        except Exception as e:
            logger.error(f"Error processing input at index {i}: {str(e)}")
            continue
    
    avg_reward = total_reward / valid_examples if valid_examples > 0 else 0
    logger.info(f"Epoch {epoch}, Average Reward: {avg_reward:.4f}, Valid Examples: {valid_examples}")
    wandb.log({"epoch": epoch, "average_reward": avg_reward, "valid_examples": valid_examples})

# Save the fine-tuned model
model.save_pretrained("gpt2_rlhf_finetuned")

# Finish wandb run
wandb.finish()