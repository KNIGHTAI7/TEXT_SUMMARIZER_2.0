📑AI-Powered Text Summarization 2.0

Live Demo: https://textsummarizer20-fvwwvwzdutuvshgtwmnyxx.streamlit.app/


🚀 Overview

AI-Powered Text Summarization 2.0 is an advanced NLP application that moves beyond simple statistical methods to provide deep-sequence analysis. By integrating Recurrent Neural Networks (RNNs) and a custom preprocessing engine, the system delivers high-precision domain classification, noise-free text cleaning, and context-aware summaries.


✨ Key Features

RNN-Driven Classification: Utilizes an LSTM (Long Short-Term Memory) network to understand the sequential context of text, providing more accurate domain predictions than traditional models.

Advanced Noise Reduction: A custom Regex-powered pipeline that automatically strips out non-semantic noise such as dates, page numbers, URLs, and redundant metadata.

Analytical Title Generation: Instead of simply pulling the first sentence, the system analyzes the most semantically dense parts of the input to generate a representative headline.

Strict Summarization: Powered by the TextRank algorithm, ensuring a concise and informative summary consistently delivered in 5–6 sentences.

Minimalist Dark UI: A professional, developer-centric interface built with Streamlit, featuring a high-contrast dark theme and JetBrains Mono typography.


🛠️ Technical Stack
Deep Learning: TensorFlow / Keras (LSTM Architecture)

NLP Engine: Sumy (TextRank), NLTK

Frontend: Streamlit

Data Processing: Regex, Scikit-learn (Label Encoding), NumPy, Pickle


📊 Pipeline Logic

Ingestion & Cleaning: Raw text is passed through a multi-stage Regex filter to remove structural noise.

Sequence Vectorization: Text is tokenized and padded to a fixed length for the LSTM model.

Domain Inference: The RNN predicts the subject matter (e.g., Finance, Tech, Health) based on word sequence patterns.

Extractive Summarization: The TextRank algorithm ranks sentence importance to extract a cohesive summary.

Title Synthesis: The top-ranked sentence is transformed into a clean, capitalized analytical title.

📂 Project Structure Plaintext

├── app.py                # Main Streamlit application & UI

├── rnn_domain_model.h5   # Trained LSTM model weights

├── assets.pkl            # Pre-processing assets (Tokenizer/LabelEncoder)

├── requirements.txt      # Environment dependencies

└── README.md             # Project documentation
