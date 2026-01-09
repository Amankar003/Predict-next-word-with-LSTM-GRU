# 🧠 Predict Next Word with LSTM & GRU

**🔎 Natural Language Processing (NLP) | Deep Learning | Next-Word Prediction**

This repository contains a **deep learning-based solution** that predicts the *next word* in a sentence using two powerful sequence models:

✔ **LSTM (Long Short-Term Memory)**  
✔ **GRU (Gated Recurrent Units)**

This model learns language patterns and predicts the next word based on text input — a foundational technique in language modeling used in keyboards, chatbots, writing assistants, and smart text systems.

---

## 📁 Repository Structure

📦 Predict-next-word-with-LSTM-GRU
┣ 📜 Hamlet.txt # Text dataset (Shakespeare corpus)
┣ 📜 Predict next word using LSTM & GRU.ipynb # Complete Jupyter notebook
┣ 📜 next_word_lstm.h5 # Trained LSTM model
┣ 📜 tokenizer.pickle # Tokenizer for text processing
┗ 📜 README.md # This file


---

## 🧠 Project Overview

Next-word prediction is a *sequence modeling task* where the model is trained to predict the most likely next word given a sequence of previous words. This task is core to many NLP applications such as:

✔ Autocomplete keyboards  
✔ Text suggestion systems  
✔ Conversational AI / Chatbots  
✔ Language modeling research

🔥 This project uses a dataset (e.g., *Hamlet.txt*) as training text, then builds both LSTM and GRU-based neural networks to learn patterns and generate predictions. :contentReference[oaicite:0]{index=0}

---

## 📌 Key Concepts

### 🧠 LSTM (Long Short-Term Memory)

- A type of Recurrent Neural Network (RNN)  
- Designed to **remember long-range dependencies** in sequences  
- Solves vanishing gradient problems seen in vanilla RNNs

### 🔁 GRU (Gated Recurrent Unit)

- A simplified variant of LSTM  
- Fewer parameters → faster training with comparable performance  
- Useful for many NLP sequence tasks  

Both models are excellent for sequence text tasks like next-word prediction. :contentReference[oaicite:1]{index=1}

---

## 🛠️ How It Works (High-Level)

### 1. 📚 Data Preparation

- Load raw text corpus (`Hamlet.txt`)
- Clean and normalize the text
- Tokenize into words
- Create input sequences of words
- Pad sequences so each input is the same length

### 2. 🧮 Model Training

- Use Tokenizer to convert words → integer indices
- Train LSTM and/or GRU model to learn context patterns
- Optimize model with backpropagation

### 3. 🧪 Prediction

Given a seed sequence (e.g., `"to be or"`), the model predicts the next word by:

- Converting the seed text to tokens
- Feeding into the model
- Outputting the most probable next word

📌 After training, the model is **exported** (`next_word_lstm.h5`) and can be loaded anytime for predictions.

---

## 🧾 Notebook: Step-by-Step

Open the main notebook:

```bash
Predict next word using LSTM & GRU.ipynb

The notebook contains:

📌 Loading dataset
📌 Text preprocessing
📌 Building the model
📌 Training loops
📌 Evaluation & sample predictions


| Tool               | Purpose                          |
| ------------------ | -------------------------------- |
| Python             | Core programming language        |
| TensorFlow / Keras | Neural networks                  |
| Jupyter Notebook   | Interactive coding + experiments |
| NLP Tokenizer      | Text preprocessing               |
