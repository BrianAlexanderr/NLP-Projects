# Movie Review Sentiment Classification (Indonesian)

This repository explores **state-of-the-art NLP models** for classifying Indonesian movie reviews into **positive** and **negative** sentiments.  

The dataset originates from Kaggle's English movie review dataset, which was **translated into Indonesian** and further **augmented** to increase variety.

---

## 📊 Dataset

- **Source**: Kaggle movie review dataset (originally in English).  
- **Steps taken**:
  - Translated into **Indonesian**.
  - Augmented by **masking certain words** to increase variety and robustness.  
- **Classes**:  
  - **Positive** 🎉  
  - **Negative** 😞  

---

## 🧠 Models Tested

The following **pre-trained NLP models** were fine-tuned and compared:

- DistilIndoBERT
- IndoBERT
- IndoBERT-Lite
- mBERT

---

## ⚙️ Methodology

1. **Data Preprocessing**  
   - Translation (EN → ID).  
   - Augmentation with word masking.  
   - Tokenization using model-specific tokenizers.  

2. **Training**  
   - Fine-tuned each model on Indonesian dataset.  
   - Used stratified train/validation split.  

3. **Evaluation**  
   - Metrics: Accuracy, F1-score, Precision, Recall.  
   - Compared across models.  

---

## 📈 Results (Example)

| Model            | Accuracy | F1-score |
|------------------|----------|----------|
| DistilIndoBERT   | 96.00%   | 0.96     |
| IndoBERT         | 93.00%   | 0.93     |
| IndoBERT-Lite    | 96.00%   | 0.96     |
| mBERT            | 95.00%   | 0.95     |

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   note: you might need to see torch documentation incase you wanted to run with your gpu
   
4. Train a model:
   ```bash
   python train.py --model distilindobert
   ```

5. Evaluate:
   ```bash
   python evaluate.py --model distilindobert
   ```

---

## 📌 Notes

- This project compares **multiple Indo-centric NLP models** with **multilingual baselines**.  
- Augmentation with masking helped to simulate more diverse language usage.  
- Future work: Try **LLaMA-based** or **Mistral-based** models fine-tuned for classification.  

---

## 📄 License

This project is released under the **MIT License**.  
