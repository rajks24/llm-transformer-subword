# Subword Transformer Language Model (LLM) - From Scratch with PyTorch

This project demonstrates how to build a **subword-level transformer-based language model** entirely from scratch using:

- 🧱 SentencePiece (BPE tokenization)
- 🔥 PyTorch TransformerEncoder
- 🌀 Top-k and Top-p sampling for generation
- 🧪 WikiText-103 as source corpus

### 📁 Project Structure

```shell
llm-transformer-subword/
├── data
│   ├── tokenized_wikitext103.pt
│   └── wikitext103.txt
├── notebook.ipynb
├── README.md
├── wikitext_bpe.model
└── wikitext_bpe.vocab
```

### ✨ Key Features

- ✅ Full tokenizer training using `SentencePieceTrainer`
- ✅ Custom transformer model with:
  - Positional encoding
  - Attention-based encoding blocks
  - Configurable size (layers, heads, embedding)
- ✅ Generation with:
  - Temperature
  - Top-k and Top-p filtering
- ✅ Output cleanup post-decoding for more natural text

### 📌 Prompt Example

```text
Prompt: "In conclusion, the results of the analysis suggest"
Generated: "In conclusion, the results of the analysis suggest that the system was developed as a component of the overall design in a distributed manner..."
```

### 🧰 Tech Stack

- **Python 3.11+**
- **PyTorch** – for model architecture, training loop, and tensor ops

| Library                                    | Purpose                                             |
| ------------------------------------------ | --------------------------------------------------- |
| `torch`, `torch.nn`, `torch.nn.functional` | Deep learning framework, model definition, training |
| `sentencepiece`                            | Subword tokenizer (BPE) for tokenizing text         |
| `datasets` (🤗 HuggingFace)                | To load datasets like WikiText-103                  |
| `tqdm`                                     | Progress bars during training                       |
| `os`, `math`, `time`                       | General utility modules (standard Python libs)      |
| `matplotlib.pyplot`                        | Plotting (e.g., attention weights)                  |
| `seaborn`                                  | Heatmaps for attention visualization                |

### 🔍 Key Components Explained

### SentencePiece BPE Tokenizer

This project uses [SentencePiece](https://github.com/google/sentencepiece) to implement a subword tokenizer based on the Byte Pair Encoding (BPE) algorithm. Unlike word-level or character-level tokenizers, SentencePiece breaks down text into subword units, enabling:

- Robust handling of out-of-vocabulary words (e.g., "tokenization" → `["▁token", "ization"]`)
- Reduced vocabulary size while preserving semantic structure
- Language-agnostic preprocessing (no need for whitespace tokenization)

In this project:

- We trained a BPE tokenizer on `wikitext103.txt` using `vocab_size=16,000` or `32,000`
- The trained model is saved as wikitext_bpe.model and used throughout for encoding and decoding
- Cleaned up decoding (e.g., replacing `@-@` with `-`, fixing punctuation) helps generate readable output

### PyTorch Transformer

The model is built from scratch using PyTorch's `TransformerEncoder` module:

- Custom positional encoding and attention blocks
- Configurable depth, width, and number of attention heads
- Final output is a language model trained to predict next subword tokens

### Hugging Face datasets

We use the 🤗 `datasets` library to load WikiText-103, a large corpus of Wikipedia articles commonly used in LLM training. It provides:

- Fast, memory-efficient data access
- Easy dataset management via `load_dataset("wikitext", "wikitext-103-raw-v1")`

### 📚 Ideal For

- NLP enthusiasts
- ML engineers exploring language models
- Anyone wanting to learn how LLMs work under the hood

📌 Credits
Created by _[Rajesh Singh](https://www.kaggle.com/rajinh)_ 😀
