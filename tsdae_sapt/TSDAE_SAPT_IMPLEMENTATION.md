# Sentence-Adaptive Pre-Training (SAPT) for BGE-M3 using TSDAE

## Project Goal
Adapt the **BGE-M3** base embedding model to natively understand the syntax, slang, and structural noise of informal Indonesian text. 

By running Unsupervised **TSDAE (Transformer-based Sequential Denoising Auto-Encoder)** on the `prdect-id` dataset, the encoder will learn to map informal product reviews into robust dense vectors at the sentence level, acting as the perfect foundation for downstream informal query retrieval.

---

## Dataset Processing (`prdect-id`)

The `prdect-id` dataset contains Indonesian product reviews and sentiment labels. 
* **Crucial Rule:** TSDAE is entirely unsupervised. We will **discard the sentiment labels** and metadata. 
* We only extract the raw review text string. The model learns semantics by attempting to reconstruct the deleted words from the review string itself.

---

## Architecture Setup

1. **Noise Function**: `DenoisingAutoEncoderDataset` will automatically apply deletion noise to the input reviews (deleting $\sim 60\%$ of the tokens).
2. **Encoder**: The frozen `BAAI/bge-m3` backbone.
3. **Pooling**: Extract the `[CLS]` token to act as the strict informational bottleneck.
4. **Decoder**: A shallow Transformer decoder attached to the `[CLS]` vector.
5. **Loss Objective**: Cross-Entropy loss between the decoder's output and the original, uncorrupted review text.

---

## Step-by-Step Implementation Script (`train_sapt.py`)

Ensure your environment has the required libraries:
`pip install sentence-transformers pandas nltk datasets`

```python
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer, models, datasets, losses
from torch.utils.data import DataLoader

# 1. Download NLTK punkt for sentence splitting (required by TSDAE noise function)
nltk.download('punkt')
nltk.download('punkt_tab')

def load_informal_corpus(csv_path: str):
    """
    Loads the prdect-id dataset and extracts only the text column.
    Drops all sentiment labels as TSDAE is unsupervised.
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # ASSUMPTION: The text column in prdect-id is named 'Review' or 'text'. Adjust as needed.
    text_column = 'Text' if 'Text' in df.columns else df.columns[0] 
    
    # Drop nulls and extract as a flat Python list of strings
    train_sentences = df[text_column].dropna().astype(str).tolist()
    print(f"Loaded {len(train_sentences)} informal Indonesian sentences.")
    return train_sentences

def train_tsdae():
    # 2. Load the informal review dataset
    train_sentences = load_informal_corpus('prdect-id.csv')

    # 3. Create the TSDAE Denoising Dataset
    # This wrapper automatically applies word deletion noise during training
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
    
    # Use a small batch size as BGE-M3 is large (adjust based on your GPU VRAM)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # 4. Initialize the BGE-M3 Base Model
    model_name = 'BAAI/bge-m3'
    print(f"Initializing base encoder: {model_name}")
    
    # We load it as modular components to control the pooling layer explicitly
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), 
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 5. Define the Denoising AutoEncoder Loss
    # tie_encoder_decoder=True reuses the encoder weights for the decoder, 
    # massively saving memory and improving alignment.
    loss = losses.DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)

    # 6. Execute SAPT Training
    print("Starting TSDAE SAPT Training...")
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=1,                 # 1 epoch is usually sufficient for TSDAE on large datasets
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True
    )

    # 7. Save the SAPT-adapted base model
    output_path = './output/bge-m3-sapt-informal-id'
    model.save(output_path)
    print(f"SAPT model successfully saved to {output_path}")

if __name__ == '__main__':
    train_tsdae()