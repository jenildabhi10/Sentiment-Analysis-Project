# 💬 Sentiment Analysis using Hugging Face Transformers

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

### 📘 Overview
This project performs **Sentiment Analysis** on text data using the **Hugging Face Transformers** library in Python.  
It reads sentences from a CSV file, analyzes their sentiment using a pretrained model, and adds a new column with predicted sentiment labels (like `POSITIVE` or `NEGATIVE`).

You can run this project easily in a Jupyter notebook or as a standalone Python script.

---

## 🧾 Table of Contents
1. [Demo](#demo)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Code Example](#code-example)
7. [Results](#results)
8. [Model Information](#model-information)
9. [Requirements](#requirements)
10. [Contributing](#contributing)
11. [License](#license)

---

## 🎯 Demo

**Input CSV (`sentiment_analysis.csv`):**
| text |
|------|
| I love this product! |
| This is the worst thing ever. |

**Output CSV (`sentiment_analysis_with_sentiment.csv`):**
| text | sentiment |
|------|------------|
| I love this product! | POSITIVE |
| This is the worst thing ever. | NEGATIVE |

---

## 🌟 Features
✅ Sentiment analysis using **Hugging Face Transformers**  
✅ Automatically adds a new `sentiment` column  
✅ Works with any text dataset in CSV format  
✅ Compatible with **Jupyter Notebook** or plain Python  
✅ Clean and well-documented code  

---

## 📁 Project Structure

```
Sentiment-Analysis-Project/
│
├── Sentiment_analysis.ipynb              # Main notebook
├── sentiment_analysis.csv                # Input text data
├── sentiment_analysis_with_sentiment.csv # Output with predicted sentiments
├── requirements.txt                      # List of dependencies
├── .gitignore                            # Files to ignore in Git
├── LICENSE                               # MIT License
└── README.md                             # Project documentation
```

---

## ⚙️ Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/jenildabhi10/Sentiment-Analysis-Project.git
   cd Sentiment-Analysis-Project
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On macOS/Linux
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 🧩 Run via Jupyter Notebook
1. Open your terminal and run:
   ```bash
   jupyter notebook
   ```
2. Open `Sentiment_analysis.ipynb`
3. Run all the cells step by step.

### 🐍 Or run directly in Python
```bash
python sentiment_analysis_script.py
```

---

## 🧠 Code Example

```python
import pandas as pd
from transformers import pipeline

# Load model
sentiment_pipeline = pipeline("sentiment-analysis")

# Load data
df = pd.read_csv("sentiment_analysis.csv")

# Analyze sentiments
results = []
for text in df["text"]:
    result = sentiment_pipeline(str(text))[0]
    results.append(result['label'])

# Add results to DataFrame
df["sentiment"] = results

# Save new file
df.to_csv("sentiment_analysis_with_sentiment.csv", index=False)
print("✅ Sentiment analysis complete!")
```

---

## 📊 Results

Example output file (`sentiment_analysis_with_sentiment.csv`):

| text | sentiment |
|------|------------|
| The movie was amazing! | POSITIVE |
| It was boring and too long. | NEGATIVE |
| Great acting and storyline. | POSITIVE |

---

## 🤗 Model Information

The model used is:
> **distilbert-base-uncased-finetuned-sst-2-english**

It’s a lightweight, high-performance **Transformer model** trained for **binary sentiment classification** (Positive / Negative).

📘 [Model Card on Hugging Face](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)

---

## 🧩 Requirements

Install all dependencies with:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pandas transformers torch
```

---

## 🤝 Contributing

Contributions are always welcome!

1. Fork the project 🍴  
2. Create a new branch (`git checkout -b feature-branch`)  
3. Commit your changes (`git commit -m "Added a cool feature"`)  
4. Push the branch (`git push origin feature-branch`)  
5. Open a **Pull Request** 🚀  

---

## 🪪 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 💬 Acknowledgements
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [Pandas](https://pandas.pydata.org/)
- [OpenAI ChatGPT](https://chat.openai.com/) for guidance 😉

---

⭐ **If you like this project, don’t forget to give it a star on GitHub!**
