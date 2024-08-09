import torch
from torch.optim import Adam
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from textblob import TextBlob
import numpy as np
from tqdm import tqdm
import wandb

# Initialize wandb
wandb.init(project="gpt2-rl-finetuning", name="gpt2-sentiment-tuning")

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer
model_path = "gpt2_small_wikitext_pretrained_model"
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Hyperparameters
learning_rate = 1e-5
num_episodes = 1000
max_steps = 50
temperature = 0.7

# Optimizer
optimizer = Adam(model.parameters(), lr=learning_rate)

# Sentiment analysis function
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Text generation function
def generate_text(model, tokenizer, prompt, max_length):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# REINFORCE algorithm
def reinforce(model, optimizer, prompt, target_sentiment=0.5):
    model.train()
    generated_text = generate_text(model, tokenizer, prompt, max_steps)
    sentiment = get_sentiment(generated_text)
    reward = -abs(sentiment - target_sentiment)  # Negative absolute difference as reward

    # Compute loss
    input_ids = tokenizer.encode(generated_text, return_tensors="pt").to(device)
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss

    # Backpropagate
    (reward * loss).backward()
    optimizer.step()
    optimizer.zero_grad()

    return generated_text, sentiment, reward

# Training loop
for episode in tqdm(range(num_episodes)):
    prompt = "Once upon a time"
    generated_text, sentiment, reward = reinforce(model, optimizer, prompt)

    # # Logging
    # wandb.log({
    #     "episode": episode,
    #     "sentiment": sentiment,
    #     "reward": reward,
    # })

    if episode % 100 == 0:
        print(f"Episode {episode}")
        print(f"Generated text: {generated_text}")
        print(f"Sentiment: {sentiment:.2f}")
        print(f"Reward: {reward:.2f}")
        print("--------------------")

# Save the fine-tuned model
model_save_path = "gpt2_rl_finetuned_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("RL fine-tuning completed!")
wandb.finish()