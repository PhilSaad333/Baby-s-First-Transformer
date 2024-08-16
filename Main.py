from Data import Text8, Tokenizer, DataLoader
from Model import Decoder
from TrainingLoopCPUGPU import train_transformer_cpu_gpu
#from TrainingLoopTPU import train_transformer_tpu #Exists because I was running this on Google Colab TPU
from UseModel import generate_text
import torch

# Hyperparameters
DATA_ROOT = './data'
TEXT_FRAC = 0.1
SEQUENCE_LENGTH = 32
BATCH_SIZE = 64
NUM_TRAIN = 1000
NUM_VAL = 100
NGRAM_LENGTH = 2
NUM_NGRAMS = 40
NUM_HIDDENS = 128
FFN_NUM_HIDDENS = 256
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 0.0001
NUM_EPOCHS = 3
GEN_MAX_LENGTH = 100
GEN_TEMPERATURE = 1.0
MODEL_SAVE_PATH = 'text8savedmodel.pth'

def setup_data_and_model():
    # Load and preprocess data
    text8 = Text8(root=DATA_ROOT)
    raw_text = text8.read()
    text_used = raw_text[:int(len(raw_text)*TEXT_FRAC)]
    tokenizer = Tokenizer(text_used, NGRAM_LENGTH, NUM_NGRAMS)
    tokens = tokenizer.tokenize()
    vocab_size = len(tokenizer.vocab)
    
    data_loader = DataLoader(tokens, SEQUENCE_LENGTH, BATCH_SIZE, num_train=NUM_TRAIN, num_val=NUM_VAL)
    
    # Initialize model
    model = Decoder(
        vocab_size=vocab_size,
        num_hiddens=NUM_HIDDENS,
        ffn_num_hiddens=FFN_NUM_HIDDENS,
        num_heads=NUM_HEADS,
        num_blks=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    return tokenizer, data_loader, model

def train_model():
    tokenizer, data_loader, model = setup_data_and_model()
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train model
    model = train_transformer_cpu_gpu(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, device=device)
    
    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

def generate_text_from_model():
    tokenizer, _, model = setup_data_and_model()
    
    # Load the trained model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    # Generate text
    start_phrase = "the quick brown"
    generated_text = generate_text(model, tokenizer, start_phrase, max_length=GEN_MAX_LENGTH, temperature=GEN_TEMPERATURE)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    # Uncomment the function you want to run
    train_model()
    # generate_text_from_model()