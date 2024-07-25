from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from train_gpt2 import GPT, GPTConfig
import tiktoken

enc = tiktoken.get_encoding("gpt2")
device = "cpu"
if torch.cuda.is_available():
        device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
print(f"using device: {device}")

model.eval()


def load_model(model_path, device='cuda'):
    # Initialize your model architecture
    model = GPT(GPTConfig(vocab_size=50304))

    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Load the state dict into the model
    model.load_state_dict(state_dict)
    
    # Move the model to the specified device
    model = model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model
model = load_model('model_weights.pt')

print("Model loaded successfully!")

num_return_sequences = 4
max_length = 32
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42)
device_type = "cuda" if device.startswith("cuda") else "cpu"

while xgen.size(1) < max_length:
            # forward the model to get the logits
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
            probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
for i in range(num_return_sequences):
    tokens = xgen[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f" sample {i}: {decoded}")
