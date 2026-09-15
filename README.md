# RedactedGPT

Can a language model guess what's under the black bars? This project fine-tunes **BERT**,
**DeBERTa**, and **GPT-2** to predict redacted words in declassified CIA documents, and compares
how the three architectures handle the task.

## Results

Best **top-10 test accuracy**  the redacted word appears in the model's ten highest-ranked
predictions  after a hyperparameter sweep over batch size and learning rate.

| Model | Checkpoint | Params | Best top-10 test accuracy |
|-------|-----------|-------:|--------------------------:|
| **BERT** | `bert-base-cased` | 109M | **93.0%** |
| DeBERTa | `microsoft/deberta-base` | 100M | 91.7% |
| GPT-2 | `gpt2` | 124M | 62.2% |

**Why the bidirectional models win.** Predicting a hidden word from text on *both* sides of it is
exactly the masked language modelling objective BERT and DeBERTa were pretrained on. GPT-2 is
autoregressive  it only sees preceding context  so it is solving a harder version of the problem
with less information. On the prompt *"...producing the propaganda video [MASK] Qaeda issued
follow-"*, BERT recovers `al` from the following token; GPT-2 cannot.

BERT was also the only model to show meaningful overfitting, its validation loss flattening while
training loss kept falling  consistent with it being the largest of the three.

Loss and top-10 accuracy curves for all three models are produced by `src/plot_results.py`.

## How it works

1. **`src/data_processing.py`**  converts six declassified CIA documents and the 9/11 Commission
   Report to page images, OCRs them with Tesseract, and cleans the extracted text. It then ranks
   word frequencies to pick 72 domain-specific keywords (names like *Bin Laden*, acronyms like
   *CIA*), adds each to the model tokenizer's vocabulary, and builds masked-token training CSVs of
   roughly 16,000 examples per model.
2. **`src/train_bert.py`**, **`train_deberta.py`**, **`train_gpt2.py`**  fine-tune each model on
   those CSVs via the HuggingFace `Trainer`, logging validation loss and top-10 accuracy.
3. **`src/plot_results.py`**  renders the training curves used above.

## Setup

System dependencies for the OCR pipeline:

```bash
# Debian / Ubuntu
sudo apt-get install poppler-utils tesseract-ocr
# macOS
brew install poppler tesseract
```

Python dependencies:

```bash
pip install -r requirements.txt
```

## Running

All paths resolve from `config.py`, which defaults to `./data`. Override it if your corpus lives
elsewhere:

```bash
export REDACTEDGPT_DATA=/path/to/data
```

Then:

```bash
python src/data_processing.py     # PDFs -> OCR text -> training CSVs
python src/train_bert.py          # fine-tune BERT
python src/train_deberta.py       # fine-tune DeBERTa
python src/train_gpt2.py          # fine-tune GPT-2
python src/plot_results.py        # training curves
```

Training was run on GPU; the fine-tuning scripts are impractical on CPU.

## Data

Five sample declassified CIA documents are included in `data/PDFs/` so the pipeline can be run end
to end. The full training corpus a larger set from the
[CIA FOIA Reading Room](https://www.cia.gov/readingroom/) plus page images of the 9/11 Commission
Report  is not in the repo for size reasons. See [`data/README.md`](data/README.md).

Model checkpoints and generated intermediates are gitignored.

## Contributors

A two-person course project, with equal contributions.

- **[Aiden Rosebush](https://github.com/aidenrosebush)**  found, processed, and cleaned the raw
  data and generated the custom training sets for each model; diagnosed and mitigated the GPU RAM
  overflow during evaluation; trained BERT to completion.
- **[Esmat Sahak](https://github.com/esmatsahak)**  generated the initial GPT-2 and BERT results,
  adapted those methods to DeBERTa and GPT-2, and collected the final results.

Scoping decisions were made jointly.

## Notes

Developed as a course project for ECE1786 (Creative Applications of Natural Language Processing),
University of Toronto, Fall 2023. Originally written as Colab notebooks and since converted to
standalone scripts.
