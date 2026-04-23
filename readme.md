Name: Nguyen Hai Tien Phat

MSSV: 521H0126

🔗 Link Github: https://github.com/phattien12/Hybrid-RAG-and-Fine-Tuned-LLM-for-Vietnamese-Medical-Question-Answering

# 🩺 ViHealth-QA: Hybrid RAG & Fine-Tuned LLM for Vietnamese Medical QA

A state-of-the-art **Vietnamese Medical Question Answering** system that integrates a fine-tuned **TinyLlama** model with a **Hybrid Retrieval-Augmented Generation (RAG)** pipeline. This project optimizes factual accuracy and reduces hallucinations in the medical domain using lightweight open-source technology.

[](https://www.python.org/)
[](https://pytorch.org/)
[](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
[](https://github.com/facebookresearch/faiss)

-----

## 📌 Project Overview

This system addresses the challenge of providing reliable medical advice in Vietnamese by combining the linguistic fluency of a domain-adapted LLM with the factual grounding of a high-recall retrieval system.

**Key Achievements:**

  * **Recall@5 of 94.5%**: Nearly all relevant medical contexts are found within the top 5 results.
  * **+1070% BLEU Gain**: Significant improvement in response accuracy compared to base models.
  * **Hybrid Architecture**: Combines Dense (Semantic) and Sparse (Keyword) search with Cross-Encoder re-ranking.

-----

🔗 Resources (Dataset & Checkpoints)

📂 Dataset

     HuggingFace: https://huggingface.co/datasets/phathanos0907/vihealthqa
     
     Google Drive: https://drive.google.com/drive/u/1/folders/1uEAdm-horWVj3ZpdUSxvap6_Q4P52fMH
     
🧠 Model Checkpoints (LoRA Adapter)

     HuggingFace: https://huggingface.co/phathanos0907/Checkpoint
     
     Google Drive: https://drive.google.com/drive/folders/1iHyLX4gxRj23UCioR99lbTk-_X2v6IDa?usp=drive_link

## 🏗️ System Architecture

The pipeline follows a "Retrieve-then-Rank-then-Generate" flow to ensure the model only answers based on verified medical data.

1.  **Input**: User asks a medical question in Vietnamese.
2.  **Hybrid Retrieval**:
      * **Dense (FAISS)**: Uses `multilingual-e5-base` to find semantic meaning.
      * **Sparse (BM25)**: Matches specific medical keywords (e.g., drug names, symptoms).
3.  **Re-ranking**: A Cross-Encoder (`mmarco-mMiniLMv2`) scores the relevance of retrieved segments.
4.  **Generation**: A fine-tuned **TinyLlama-1.1B** processes the context and query to generate a structured Vietnamese response.

-----

## 🧠 Model & Training (Fine-Tuning)

We applied **LoRA (Low-Rank Adaptation)** to TinyLlama-1.1B to specialize it in the Vietnamese medical lexicon without the high cost of full parameter fine-tuning.

### LoRA Settings

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **r** | 8 | Rank of the update matrices |
| **Alpha** | 16 | Scaling factor |
| **Dropout** | 0.05 | Prevents overfitting |
| **Precision** | FP16 | Balanced speed and memory |

### Training Results

The model was trained for **8 epochs** (\~1000 steps), reaching a final loss of **1.24**.

  * **Training Time**: 30m 26s (on Tesla T4).
  * **Adaptation**: High stability with consistent loss reduction.

-----

## 📊 Evaluation & Benchmarks

We compared four different configurations to find the optimal setup. **Model D** (Fine-tuned + RAG) significantly outperformed all others.

### Automatic Metrics

| Model Configuration | BLEU | ROUGE-L | BERTScore |
| :--- | :--- | :--- | :--- |
| **A: Base Model** | 0.0034 | 0.2264 | 0.6813 |
| **B: Base + RAG** | 0.0003 | 0.1584 | 0.6495 |
| **C: Fine-Tuned (FT)** | 0.0034 | 0.2264 | 0.6813 |
| **D: FT + RAG (Winner)** | **0.0398** | **0.3038** | **0.7055** |

### Why FT + RAG?

1.  **Fine-Tuning** helps the model learn the *style* and *grammar* of Vietnamese medical advice.
2.  **RAG** provides the actual *facts* and *evidence*, preventing the model from making up medical information (hallucination).

-----

## 📂 Dataset: ViHealthQA

The project utilizes the `tarudesu/ViHealthQA` dataset, a high-quality collection of **10,015 Vietnamese medical QA pairs**.

  * **Knowledge Base**: Used to build the FAISS index.
  * **Human Evaluation**: 50 complex cases were manually checked for fluency and accuracy (see `human_eval_50.csv`).

-----

## 🚀 Installation & Setup

### 1\. Prerequisites

Ensure you have a GPU environment (Google Colab, Kaggle, or local CUDA).

```bash
pip install transformers datasets peft accelerate bitsandbytes \
sentence-transformers faiss-cpu rouge-score bert-score sacrebleu \
trl rank_bm25 gradio
```

### 2\. Usage

```python
# Launch the Gradio Web Interface
import gradio as gr
# [Code to load model and retriever]
demo.launch()
```

-----

## 📂 Project Structure

```text
├── medical_adapter/      # Fine-tuned LoRA weights
├── A Hybrid Retrieval-Augmented and Fine-Tuned Large Language Model for Vietnamese Medical Question Answering.ipynb       # Complete code (Training + Evaluation)
├── human_eval_50.csv     # Side-by-side human preference data
├── README.md             # Documentation
└── output.png            # Architecture/Result visualization
```

-----

## 🔮 Future Roadmap

  * **Citations**: Add source links to every answer for transparency.
  * **Safety Filtering**: Implement a triage layer to identify emergencies and redirect to emergency services.
  * **Multi-turn Memory**: Enable the bot to remember previous symptoms in a conversation.
  * **Larger LLMs**: Experiment with PhoGPT or Llama-3-8B for better reasoning.

-----

## ⚠️ Disclaimer

**This project is for research and educational purposes only.** The information provided by the AI does not constitute professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or other qualified health provider with any questions you may have regarding a medical condition.

-----

## 📜 License

This project is licensed under the **MIT License**.

-----

👨‍💻 Author: Phat

Research Interests: Computer Vison, MultiModal Image Processing

## *Developed with ❤️ for the Vietnamese Medical AI Community.*
