# Baby's First Transformer: a simple decoder-only transformer model

## Overview:

This is my attempt at a simple decoder-only transformer model, trained on text8 data to predict the next token. It's not good, just a practice exercise. It's only here so I have at least something on GitHub.

Rather than converting single characters or words to tokens, we specify some NGRAM_LENGTH and NUM_NGRAMS, and the tokenizer uses the NUM_NGRAMS most common NGRAM_LENGTH-grams as tokens, as well as single characters. This is probably not the most optimal thing as implemented, I just did this for practice.

Some parts of the code were basically completely written by AI (Claude), and others partially or at least assisted, especially where I was worried my own version was pretty suboptimal and would be slow. I've tried to indicated with comments which parts I didn't do much of myself.

## Dependencies:
- Python 3.7+
- PyTorch
- tqdm

## Installation:
1. Clone this repository
2. Install the required dependencies:
   ```
   pip install torch tqdm
   ```

## Usage:
1. Open `main.py`
2. At the bottom of the file, comment/uncomment the appropriate lines to either train the model or generate text
3. Adjust hyperparameters at the top of `main.py` as needed
4. Run the script:
   ```
   python main.py
   ```

## Files:
- `main.py`: Main script to run training or text generation
- `Data.py`: Data loading and preprocessing classes
- `Model.py`: Defines the model architecture
- `TrainingLoopCPUGPU.py`: Contains training functions for CPU/GPU
- `TrainingLoopTPU.py`: Contains training functions for TPU (commented out in the main file, I only have this because I was using the Google Colab TPU)
- `UseModel.py`: Contains the function for generating text, given an initial prompt.
