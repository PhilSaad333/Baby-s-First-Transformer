import torch.nn.functional as F
from typing import List

def generate_text(model, tokenizer, start_phrase, max_length=100, temperature=1.0):
    model.eval()  # Set the model to evaluation mode
    
    # Tokenize the start phrase
    context = torch.tensor([tokenizer.vocab[char] for char in start_phrase]).unsqueeze(0)
    
    generated = list(context[0].tolist())
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get the predictions
            logits = model(context)
            
            # We only need the last time step for the next token prediction
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add the chosen token to the sequence
            generated.append(next_token.item())
            context = torch.cat([context, next_token], dim=1)
            
            # If we generate an EOS token, stop
            if next_token.item() == tokenizer.vocab.get('<EOS>', None):
                break
    
    return tokenizer.decode(generated)