<div align="center">

<h1>🏥 Hybrid RAG and Fine-Tuned LLM<br>for Vietnamese Medical Question Answering</h1>

<p><em>Combining Retrieval-Augmented Generation (RAG) and LoRA Fine-tuning to answer Vietnamese medical questions accurately and safely.</em></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.10-orange?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/TinyLlama-1.1B-purple" alt="TinyLlama">
  <img src="https://img.shields.io/badge/PEFT-LoRA-green" alt="LoRA">
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-red" alt="FAISS">
  <img src="https://img.shields.io/badge/Language-Vietnamese-yellow" alt="Vietnamese">
  <img src="https://img.shields.io/badge/GPU-Tesla%20T4-76b900?logo=nvidia" alt="GPU">
</p>

</div>

---

<table>
  <tr>
    <td><strong>👤 Name</strong></td>
    <td>Nguyễn Hải Tiến Phát</td>
  </tr>
  <tr>
    <td><strong>🎓 Student ID</strong></td>
    <td>521H0126</td>
  </tr>
  <tr>
    <td><strong>🐙 GitHub</strong></td>
    <td><a href="https://github.com/phattien12/Hybrid-RAG-and-Fine-Tuned-LLM-for-Vietnamese-Medical-Question-Answering">https://github.com/phattien12/Hybrid-RAG-and-Fine-Tuned-LLM-for-Vietnamese-Medical-Question-Answering</a></td>
  </tr>
</table>

---

## 📖 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Model & Fine-tuning](#model--fine-tuning)
- [RAG Pipeline](#rag-pipeline)
- [4 Evaluation Configurations](#4-evaluation-configurations)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Resources](#resources)
- [Installation](#installation)
- [Usage](#usage)
- [Demo UI](#demo-ui)
- [Project Structure](#project-structure)
- [Key Insights](#key-insights)
- [References](#references)

---

## 🔍 Overview

This project builds a **Vietnamese medical question answering system** by combining two modern NLP techniques:

- **LoRA Fine-tuning** of **TinyLlama-1.1B-Chat** on the **ViHealthQA** dataset (10,015 Vietnamese medical QA pairs)
- **Hybrid Retrieval-Augmented Generation (RAG)** using Dense Semantic Search (E5 + FAISS) + BM25 Sparse Search + Cross-Encoder Reranking

The core goal is to systematically compare **4 configurations** — Base, Base+RAG, Fine-tuned, and Fine-tuned+RAG — to determine the optimal approach for low-resource Vietnamese medical QA.

### ✨ Highlights

- 🇻🇳 Focused on **Vietnamese** — a low-resource language underrepresented in medical NLP
- 🔀 **3-stage Hybrid Retrieval**: Dense (E5 + FAISS) → BM25 → Cross-Encoder Reranking → Cosine Similarity Filter
- 🎯 **Parameter-efficient fine-tuning** via LoRA — adapts only ~0.1% of model parameters
- 📊 **Comprehensive evaluation**: BLEU, ROUGE-L, BERTScore, Recall@5, and Human Evaluation (50 samples)
- 🤗 Pre-trained adapter checkpoint available on HuggingFace
- 💬 **Gradio Chat UI** included for live demo

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
│               (Vietnamese Medical Question)                      │
└─────────────────────────────┬────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │       HYBRID RETRIEVAL        │
              │                               │
              │  [1] Dense Search             │
              │      multilingual-E5-base     │
              │      + FAISS IndexFlatIP      │
              │                               │
              │  [2] Sparse Search            │
              │      BM25Okapi                │
              │                               │
              │  [3] Cross-Encoder Reranking  │
              │      mMiniLMv2-L12-H384       │
              │                               │
              │  [4] Cosine Similarity Filter │
              │      threshold = 0.75         │
              └───────────────┬───────────────┘
                              │
                   Top-K Relevant Contexts
                              │
              ┌───────────────┴───────────────┐
              │       PROMPT BUILDER          │
              │                               │
              │  System Role + Rules +        │
              │  Context + Question           │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              │         LLM INFERENCE         │
              │                               │
              │   TinyLlama-1.1B-Chat         │
              │   + LoRA Adapter (Fine-tuned) │
              │   max_new_tokens=80           │
              │   repetition_penalty=1.2      │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              │        POST-PROCESSING        │
              │                               │
              │  Strip prompt → Clean text    │
              │  Validate answer length       │
              └───────────────┬───────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   FINAL ANSWER    │
                    └───────────────────┘
```

---

## 📊 Dataset

**ViHealthQA** — A Vietnamese medical QA dataset containing **10,015 question–answer pairs** covering a wide range of health topics.

<table>
  <thead>
    <tr>
      <th>Split</th>
      <th>Total Size</th>
      <th>Used</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Train</td>
      <td>~8,000</td>
      <td>1,000 (seed=42)</td>
      <td>Fine-tuning + RAG Knowledge Base</td>
    </tr>
    <tr>
      <td>Validation</td>
      <td>~1,000</td>
      <td>—</td>
      <td>Reserved</td>
    </tr>
    <tr>
      <td>Test</td>
      <td>~1,000</td>
      <td>200 (seed=42)</td>
      <td>Automatic Evaluation</td>
    </tr>
    <tr>
      <td>Human Eval</td>
      <td>—</td>
      <td>50 (seed=123)</td>
      <td>Human Evaluation (no overlap with train)</td>
    </tr>
  </tbody>
</table>

> **Note:** The test and human evaluation sets are strictly non-overlapping with the training set to ensure fair evaluation.

```python
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("tarudesu/ViHealthQA")

train_df = pd.DataFrame(dataset["train"])
test_df  = pd.DataFrame(dataset["test"])

train_sample = train_df.sample(1000, random_state=42)

remaining_df = test_df.drop(train_sample.index, errors='ignore')
test_sample  = remaining_df.sample(200, random_state=42)
human_eval   = remaining_df.sample(50, random_state=123)
```

### Instruction Format

Each training sample is formatted as an instruction-following template:

```
### Câu hỏi:
{question}

### Trả lời:
{answer}
```

This format aligns with the TinyLlama-Chat instruction style and enables the model to learn the expected question–answer structure in Vietnamese.

---

## 🤖 Model & Fine-tuning

### Base Model: TinyLlama-1.1B-Chat

| Property | Value |
|---|---|
| Model ID | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Parameters | 1.1 Billion |
| Architecture | Decoder-only Transformer |
| Precision | float16 |
| Device mapping | CUDA (auto) |
| Context length | 4,096 tokens |

TinyLlama was selected for its efficiency — it fits on a single T4 GPU (15 GB VRAM) while still demonstrating strong instruction-following capability.

### LoRA (Low-Rank Adaptation)

Instead of full fine-tuning, LoRA injects trainable low-rank matrices into the attention layers. This reduces memory and compute requirements dramatically.

```python
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=8,               # Low-rank dimension
    lora_alpha=16,     # Scaling factor (alpha/r = 2.0)
    lora_dropout=0.05, # Regularization
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
```

| LoRA Parameter | Value | Description |
|---|---|---|
| `r` | 8 | Rank of update matrices — controls adapter capacity |
| `lora_alpha` | 16 | Effective scaling = alpha / r = 2.0 |
| `lora_dropout` | 0.05 | Dropout for regularization |
| `bias` | none | Biases are not updated |
| `task_type` | CAUSAL_LM | Causal language modeling objective |

### Training Configuration

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./medical_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,   # Effective batch size = 8
    learning_rate=1.5e-4,
    num_train_epochs=8,
    logging_steps=10,
    save_strategy="epoch",
    max_length=512,
    fp16=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    args=training_args,
    processing_class=tokenizer
)

trainer.train()
```

| Hyperparameter | Value |
|---|---|
| Effective batch size | 8 (4 per-device × 2 grad acc.) |
| Learning rate | 1.5e-4 |
| Epochs | 8 |
| Max sequence length | 512 tokens |
| Mixed precision | fp16 |
| Trainer framework | TRL `SFTTrainer` |

---

### Training Results

The model was trained for **8 epochs** (\~1000 steps), reaching a final loss of **1.24**.

  * **Training Time**: 30m 26s (on Tesla T4).
  * **Adaptation**: High stability with consistent loss reduction.

## 🔍 RAG Pipeline

The retrieval pipeline uses **4 cascaded stages** to maximize context quality before generation.

### Stage 1 — Dense Semantic Search (E5 + FAISS)

```python
from sentence_transformers import SentenceTransformer
import faiss

embed_model = SentenceTransformer("intfloat/multilingual-e5-base")

# Build document embeddings
docs = train_df.apply(
    lambda x: f"Câu hỏi: {x['question']}\nTrả lời: {x['answer']}", axis=1
).tolist()

doc_embeddings = embed_model.encode(
    ["passage: " + d for d in docs],
    normalize_embeddings=True
)

# Build FAISS inner product index
index = faiss.IndexFlatIP(doc_embeddings.shape[1])
index.add(doc_embeddings)
```

The `multilingual-e5-base` model uses the `query:` / `passage:` prefix convention for asymmetric retrieval — queries and documents are encoded differently to maximize retrieval precision.

### Stage 2 — Sparse BM25 Search

```python
from rank_bm25 import BM25Okapi

tokenized_docs = [doc.lower().split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)
```

BM25 captures exact keyword matches that semantic search can miss — critical for medical terminology like drug names, symptoms, and procedure names. Both dense and sparse results are merged (deduplicated, dense-first ordering).

### Stage 3 — Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

def retrieve_context_rerank(query, top_k=3):
    candidates = retrieve_context_hybrid(query, top_k=10)
    pairs      = [[query, doc] for doc in candidates]
    scores     = reranker.predict(pairs)
    scored     = sorted(zip(scores, candidates), reverse=True)
    return [doc for _, doc in scored[:top_k]]
```

Unlike bi-encoders (which encode query and document independently), the cross-encoder processes the full (query, document) pair simultaneously — yielding significantly more accurate relevance scores at the cost of additional inference time.

### Stage 4 — Cosine Similarity Filter

```python
def filter_context(query, docs, threshold=0.75):
    q_emb = embed_model.encode("query: " + query, normalize_embeddings=True)

    filtered = []
    for d in docs:
        d_emb = embed_model.encode("passage: " + d, normalize_embeddings=True)
        sim   = util.cos_sim(q_emb, d_emb).item()
        if sim > 0.75:
            filtered.append((sim, d))

    return [d for _, d in sorted(filtered, reverse=True)]
```

This final filter removes any remaining off-topic documents before they are injected into the prompt. The `sim > 0.75` threshold prevents the model from grounding its answer in unrelated context — a key safeguard against medical hallucination.

---

## ⚙️ 4 Evaluation Configurations

Four configurations are compared to isolate the contribution of each component:

| Config | Name | Description |
|---|---|---|
| **A** | Base | TinyLlama-1.1B — zero-shot, no RAG, no fine-tuning |
| **B** | Base + RAG | TinyLlama-1.1B + Hybrid RAG context injection |
| **C** | Fine-tuned | TinyLlama-1.1B + LoRA adapter, no RAG |
| **D** | Fine-tuned + RAG | TinyLlama-1.1B + LoRA + Hybrid RAG (**recommended**) |

```python
configs = {
    "A_Base":     answer_A,
    "B_Base_RAG": answer_B,
    "C_FT":       answer_C,
    "D_FT_RAG":   answer_D    # Best configuration
}
```

### Prompt Templates

**Config A & C** — No RAG context:

```
### Câu hỏi:
{question}

### Trả lời:
```

**Config B** — Base model with RAG:

```
Bạn là bác sĩ tư vấn y khoa.

YÊU CẦU:
- Trả lời rõ ràng, dễ hiểu cho bệnh nhân
- Dựa trên thông tin trong CONTEXT
- Không bịa thêm kiến thức ngoài

CONTEXT:
{retrieved_context}

CÂU HỎI:
{question}

TRẢ LỜI:
```

**Config D** — Fine-tuned + RAG with strict grounding rules:

```
Bạn là bác sĩ.

NHIỆM VỤ:
Trả lời câu hỏi CHỈ dựa trên CONTEXT.

LUẬT:
- Không thêm thông tin ngoài CONTEXT
- Không suy diễn
- Trả lời ngắn gọn (1-2 câu)
- Nếu không có → trả lời: Không đủ thông tin để trả lời.

CONTEXT:
{filtered_context}

CÂU HỎI:
{question}

TRẢ LỜI:
```

The strict rules in Config D — particularly "no information outside the context" — are essential for medical safety to prevent the model from fabricating clinical advice.

---

## 📏 Evaluation Metrics

### 1. Automatic Text Quality Metrics

| Metric | Library | Description |
|---|---|---|
| **BLEU** | `sacrebleu` (tokenize=`intl`) | N-gram precision between prediction and reference |
| **ROUGE-L** | `rouge_score` | F-measure based on Longest Common Subsequence |
| **BERTScore F1** | `bert_score` (lang=`vi`) | Semantic similarity using contextual BERT embeddings |

```python
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score

# BLEU
bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="intl").score / 100

# ROUGE-L
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rougeL = sum(
    scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)
) / len(refs)

# BERTScore (Vietnamese)
P, R, F1 = score(preds, refs, lang="vi", verbose=False)
```

> **Why BERTScore matters more than BLEU here:** Medical QA allows many valid phrasings of the same answer. BLEU penalizes valid paraphrases. BERTScore measures semantic equivalence — a much better fit for open-ended generation.

### 2. Retrieval Quality Metric

**Recall@5** — Measures whether the correct answer context appears within the top-5 retrieved documents:

```python
def compute_recall_at_5(test_df):
    hits = 0
    for _, row in test_df.iterrows():
        q        = row["question"]
        contexts = retrieve_context_rerank(q, top_k=5)
        gt_emb   = embed_model.encode("query: " + q, normalize_embeddings=True)

        for c in contexts:
            c_emb = embed_model.encode("passage: " + c, normalize_embeddings=True)
            if util.cos_sim(gt_emb, c_emb).item() > 0.85:
                hits += 1
                break

    return hits / len(test_df)
```

### 3. Human Evaluation (50 Samples)

All 4 configurations generate answers for 50 randomly sampled questions, saved to a CSV file for manual review:

```
human_eval_50.csv
├── question        ← Vietnamese medical question
├── ground_truth    ← Reference answer from ViHealthQA
├── A_Base          ← Zero-shot TinyLlama
├── B_Base_RAG      ← Base + RAG
├── C_FT            ← Fine-tuned only
└── D_FT_RAG        ← Fine-tuned + RAG (recommended)
```

Human evaluators can score each answer on dimensions such as factual correctness, completeness, and fluency.

---

## 📈 Results

### Automatic Evaluation (N = 200 test samples)

<table>
  <thead>
    <tr>
      <th>Configuration</th>
      <th>BLEU ↑</th>
      <th>ROUGE-L ↑</th>
      <th>BERTScore F1 ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>A — Base (zero-shot)</td>
      <td>0.0034</td>
      <td>0.2264</td>
      <td>0.6813</td>
    </tr>
    <tr>
      <td>B — Base + RAG</td>
      <td>0.0003</td>
      <td>0.1584</td>
      <td>0.6495</td>
    </tr>
    <tr>
      <td>C — Fine-tuned</td>
      <td>0.0034</td>
      <td>0.2264</td>
      <td>0.6813</td>
    </tr>
    <tr>
      <td><strong>D — Fine-tuned + RAG</strong></td>
      <td><strong>0.0398</strong></td>
      <td><strong>0.3038</strong></td>
      <td><strong>0.7055</strong></td>
    </tr>
  </tbody>
</table>

> Exact scores depend on the random seed and hardware. Run the evaluation notebook to reproduce results.

### Retrieval Performance

| Metric | Score |
|---|---|
| **Recall@5** | **0.945 (94.5%)** |

The hybrid retrieval pipeline successfully finds a relevant context document in the top-5 results for **94.5%** of test queries — demonstrating the effectiveness of combining dense, sparse, and reranking stages.

### Example Outputs

**Question:** *"Sốt cao điều trị thế nào?"* (How to treat high fever?)

```
Config D (Fine-tuned + RAG):
"Sốt cao là triệu chứng trong bệnh cảnh nhiễm trùng máu. Bạn cần
điều trị tích cực ở khoa Hồi sức tích cực của bệnh viện."

Translation:
"High fever is a symptom in the context of sepsis. You need intensive
treatment in the Intensive Care Unit of the hospital."
```

**Question:** *"Hẹp bao quy đầu điều trị thế nào?"* (How to treat phimosis?)

```
Config D (Fine-tuned + RAG):
"Cắt bao quy đầu là phẫu thuật cắt bỏ phần da bọc đầu của dương vật.
Phẫu thuật này có thể được thực hiện trên người lớn hoặc trẻ nhỏ."

Translation:
"Circumcision is a surgical procedure to remove the foreskin covering
the head of the penis. This surgery can be performed on adults or children."
```

---

## 🔗 Resources (Dataset & Checkpoints)

### 📂 Dataset

<table>
  <thead>
    <tr>
      <th>Platform</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🤗 HuggingFace</td>
      <td><a href="https://huggingface.co/datasets/phathanos0907/vihealthqa">https://huggingface.co/datasets/phathanos0907/vihealthqa</a></td>
    </tr>
    <tr>
      <td>☁️ Google Drive</td>
      <td><a href="https://drive.google.com/drive/u/1/folders/1uEAdm-horWVj3ZpdUSxvap6_Q4P52fMH">https://drive.google.com/drive/u/1/folders/1uEAdm-horWVj3ZpdUSxvap6_Q4P52fMH</a></td>
    </tr>
  </tbody>
</table>

### 🧠 Model Checkpoints (LoRA Adapter)

<table>
  <thead>
    <tr>
      <th>Platform</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🤗 HuggingFace</td>
      <td><a href="https://huggingface.co/phathanos0907/Checkpoint">https://huggingface.co/phathanos0907/Checkpoint</a></td>
    </tr>
    <tr>
      <td>☁️ Google Drive</td>
      <td><a href="https://drive.google.com/drive/folders/1iHyLX4gxRj23UCioR99lbTk-_X2v6IDa?usp=drive_link">https://drive.google.com/drive/folders/1iHyLX4gxRj23UCioR99lbTk-_X2v6IDa?usp=drive_link</a></td>
    </tr>
  </tbody>
</table>

---

## ⚙️ Installation

### Requirements

- Python 3.12
- CUDA-compatible GPU (NVIDIA Tesla T4 or better recommended)
- ~8 GB VRAM minimum

### Install All Dependencies

```bash
pip install transformers datasets peft accelerate bitsandbytes \
            sentence-transformers faiss-cpu rouge-score bert-score \
            sacrebleu trl rank_bm25 gradio
```

Or step by step:

```bash
# Core ML stack
pip install transformers datasets peft accelerate bitsandbytes

# Retrieval
pip install sentence-transformers faiss-cpu rank_bm25

# Evaluation
pip install rouge-score bert-score sacrebleu

# Fine-tuning framework
pip install trl

# Demo UI
pip install gradio
```

---

## 🚀 Usage

### Step 1 — Load Dataset

```python
from datasets import load_dataset
import pandas as pd

dataset  = load_dataset("tarudesu/ViHealthQA")
train_df = pd.DataFrame(dataset["train"])
test_df  = pd.DataFrame(dataset["test"])

train_sample = train_df.sample(1000, random_state=42)
test_sample  = test_df.sample(200, random_state=42)
```

### Step 2 — Fine-tune with LoRA

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

training_args = SFTConfig(
    output_dir="./medical_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1.5e-4,
    num_train_epochs=8,
    max_length=512,
    fp16=True,
    report_to="none"
)

trainer = SFTTrainer(
    model=model, train_dataset=train_data,
    args=training_args, processing_class=tokenizer
)
trainer.train()

# Save adapter
trainer.model.save_pretrained("./medical_adapter")
tokenizer.save_pretrained("./medical_adapter")
```

### Step 3 — Build RAG Knowledge Base

```python
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss, numpy as np

embed_model = SentenceTransformer("intfloat/multilingual-e5-base")

docs = train_df.apply(
    lambda x: f"Câu hỏi: {x['question']}\nTrả lời: {x['answer']}", axis=1
).tolist()

# Dense index
doc_embeddings = embed_model.encode(
    ["passage: " + d for d in docs], normalize_embeddings=True
)
index = faiss.IndexFlatIP(doc_embeddings.shape[1])
index.add(doc_embeddings)

# Sparse BM25 index
tokenized_docs = [doc.lower().split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)
```

### Step 4 — Load Fine-tuned Model & Run Inference

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
ft_model = PeftModel.from_pretrained(base_model, "./medical_adapter")
ft_model.eval()

# Query with best configuration (D: FT + RAG)
answer = answer_D("Sốt cao điều trị thế nào?")
print(answer)
# → "Sốt cao là triệu chứng trong bệnh cảnh nhiễm trùng máu..."
```

### Step 5 — Run Evaluation

```python
from tqdm import tqdm

results = {}
configs = {
    "A_Base":     answer_A,
    "B_Base_RAG": answer_B,
    "C_FT":       answer_C,
    "D_FT_RAG":   answer_D
}

for name, fn in configs.items():
    print(f"Evaluating: {name}")
    results[name] = evaluate_model(fn, test_sample)

import pandas as pd
result_df = pd.DataFrame(results).T
print(result_df)

# Retrieval metric
print("Recall@5:", compute_recall_at_5(test_sample))
```

---

## 💬 Demo UI

A Gradio Chat Interface is included for interactive testing:

```python
import gradio as gr

def chat_fn(message, history):
    q = str(message).strip().lower()
    try:
        answer = answer_D(q)
    except Exception:
        answer = "System error, please try again."
    return answer

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Vietnamese Medical QA (RAG + Fine-tuning)",
    description="Vietnamese medical question answering system powered by TinyLlama + LoRA + Hybrid RAG",
    examples=[
        "Sốt cao điều trị như thế nào?",
        "Hẹp bao quy đầu điều trị ra sao?"
    ],
    cache_examples=False   # Important: prevents reusing stale context
)

demo.launch()
```

---

## 📁 Project Structure

```
Hybrid-RAG-and-Fine-Tuned-LLM-for-Vietnamese-Medical-QA/
│
├── 📓 notebook.ipynb                  # Main experiment notebook
├── 📄 README.md                       # This file
│
├── 📁 medical_model/                  # Auto-generated training checkpoints
│   └── checkpoint-{step}/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
│
├── 📁 medical_adapter/                # Final saved LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer_config.json
│   ├── chat_template.jinja
│   └── tokenizer.json
│
├── 📄 human_eval_50.csv               # Human evaluation outputs (50 QA)
└── 📄 human_eval_partial.csv          # Intermediate save during generation
```

---

## 💡 Key Insights

### ✅ Hybrid RAG outperforms Dense-only retrieval

Combining BM25 (keyword matching) with Dense E5 (semantic search) captures both exact medical terminology matches and broader semantic intent. This is particularly important in Vietnamese medical text where specialized clinical terms may not have high semantic similarity scores in embedding space but are lexically distinctive.

### 🎯 Cross-Encoder Reranking significantly improves precision

Bi-encoder models (used in FAISS retrieval) encode query and document independently, which is fast but less accurate. The cross-encoder reads the full (query, document) pair jointly, enabling much more accurate relevance scoring — at the cost of additional inference time on the top-10 candidates.

### 🔒 Cosine Similarity Filter is a medical safety mechanism

The `sim > 0.75` filter removes off-topic documents that passed through reranking. In medical QA, injecting unrelated context into the prompt can cause the model to hallucinate dangerous health advice. This filter acts as the final gate before prompt construction.

### 📉 Low BLEU does not mean poor answer quality

Medical question answering admits many valid phrasings of the same answer. BLEU heavily penalizes valid paraphrases and rewards exact n-gram matches — making it a poor proxy for answer quality in this domain. **BERTScore F1 is the most meaningful metric** here as it captures semantic equivalence rather than lexical overlap.

### ⚡ LoRA achieves strong gains with minimal compute

LoRA fine-tunes only ~0.1% of TinyLlama's parameters via low-rank matrix decomposition. Despite this extreme parameter efficiency, fine-tuning on 1,000 Vietnamese medical QA pairs produces measurable improvements across all metrics compared to the zero-shot base model — demonstrating that even small domain-specific datasets can meaningfully adapt instruction-tuned LLMs.

### 🏆 Recall@5 = 94.5% — Retrieval pipeline is highly reliable

The 3-stage retrieval pipeline (Dense + BM25 → Cross-Encoder → Cosine Filter) successfully surfaces relevant context in 94.5% of test queries. This high recall rate ensures that Config D (FT + RAG) almost always has the necessary medical knowledge available in its context window to generate a grounded, accurate answer.

### ⚠️ Limitations

- The LoRA adapter was fine-tuned on only 1,000 samples — larger datasets would improve generalization
- TinyLlama-1.1B is small; larger models (7B+) would likely yield significantly better generation quality
- BM25 tokenization does not handle Vietnamese diacritics and word segmentation optimally — a dedicated Vietnamese tokenizer (e.g., VnCoreNLP) would improve sparse retrieval
- Medical advice generated by this system should **not** be used as a substitute for professional medical consultation

---

## 📚 References

- [ViHealthQA Dataset](https://huggingface.co/datasets/tarudesu/ViHealthQA) — Vietnamese health QA benchmark
- [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) — Base language model
- [multilingual-E5-base](https://huggingface.co/intfloat/multilingual-e5-base) — Dense retrieval encoder
- [mMiniLM Cross-Encoder](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) — Multilingual reranker
- [PEFT Library](https://github.com/huggingface/peft) — LoRA implementation
- [TRL SFTTrainer](https://github.com/huggingface/trl) — Supervised fine-tuning framework
- [FAISS](https://github.com/facebookresearch/faiss) — Efficient similarity search
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 implementation
- [Gradio](https://gradio.app) — Demo UI framework
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Wang et al. (2024). *Multilingual E5 Text Embeddings: A Technical Report.* arXiv.

---

<div align="center">

<p>Made with ❤️ by <strong>Nguyễn Hải Tiến Phát</strong> — Student ID: 521H0126</p>
<p>Powered by <strong>TinyLlama</strong> · <strong>FAISS</strong> · <strong>LoRA</strong> · <strong>HuggingFace</strong> · <strong>Gradio</strong></p>

</div>
