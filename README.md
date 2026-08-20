<div align="center">

# Multi-Module NLP System for Real-Time Conversational Analysis

### An AI-powered conversational intelligence platform for real-time language understanding, sentiment analysis, toxicity detection, summarization, and communication insights.

<img src="https://readme-typing-svg.demolab.com?font=Poppins&weight=600&size=22&pause=1000&color=4F8EF7&center=true&vCenter=true&width=800&lines=Real-Time+Conversational+Intelligence;Multi-Module+Natural+Language+Processing;Sentiment+%7C+Emotion+%7C+Toxicity+Analysis;AI-Powered+Conversation+Insights;Final+Year+Engineering+Project" />

---

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react)
![NodeJS](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=node.js&logoColor=white)
![Express](https://img.shields.io/badge/Express-000000?style=for-the-badge&logo=express)
![MongoDB](https://img.shields.io/badge/MongoDB-13AA52?style=for-the-badge&logo=mongodb)
![Socket.io](https://img.shields.io/badge/Socket.io-black?style=for-the-badge&logo=socket.io)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge)

</div>

---

# Overview

Digital communication platforms have evolved tremendously, yet they primarily focus on **message exchange** rather than **understanding the content and context of conversations**.

This project introduces a **Multi-Module NLP System** capable of performing intelligent analysis of conversations in real time by integrating multiple Natural Language Processing modules into a unified pipeline.

Rather than functioning as a conventional chat application, the platform serves as an **AI-driven conversational intelligence system** capable of extracting meaningful insights from user interactions.

---

# Getting Started

The application has four local processes: MongoDB, the Node/Express server,
the Python NLP service, and the Vite client. Start them in that order.

```powershell
# 1. Start MongoDB (if it is not already running)
mongod

# 2. Start the backend
cd server
copy .env.example .env
npm run server

# 3. Start the NLP service in a second terminal
cd nlp-service
$env:PYTHONPATH = "."
C:\nlp-venvs\ptp-nlp-service\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

# 4. Start the client in a third terminal
cd client
copy .env.example .env
npm run dev
```

The Python environment is kept at `C:\nlp-venvs\ptp-nlp-service` to avoid
Windows long-path installation errors under OneDrive. Leave `NLP_SERVICE_URL`
unset in `server/.env` if chat should run without optional NLP analysis.

---

# Features

###  Conversational Intelligence

- Real-Time Conversation Processing
- Conversation Analytics
- Context-Aware NLP Pipeline
- Live AI Inference

---

###  Sentiment Analysis

Detects whether messages express

- Positive sentiment
- Negative sentiment
- Neutral sentiment

---

###  Emotion Detection

Identifies emotional states including

- Happiness
- Anger
- Sadness
- Fear
- Surprise
- Disgust

---

###  Tone Detection

Analyzes communication style

- Formal
- Informal
- Friendly
- Professional
- Aggressive
- Polite

---

###  Toxicity Detection

Automatically detects

- Toxic language
- Offensive speech
- Hate speech
- Abusive messages
- Harassment

---

###  Conversation Summarization

Generate concise summaries of

- Long conversations
- Group discussions
- Important decisions
- Key takeaways

---

###  AI Smart Reply

Suggests context-aware replies using modern NLP models.

---

###  Conversation Insights Dashboard

Visualizes

- Sentiment trends
- Emotional progression
- Toxicity frequency
- Conversation statistics
- Communication quality

---

# System Architecture

```text
                    User Conversation
                           │
                           ▼
                Text Preprocessing Layer
      ─────────────────────────────────────
      • Cleaning
      • Tokenization
      • Stop-word Removal
      • Lemmatization
                           │
                           ▼
              Multi-Module NLP Engine
      ─────────────────────────────────────
      • Sentiment Analysis
      • Emotion Detection
      • Tone Detection
      • Toxicity Detection
      • Hate Speech Detection
      • Topic Extraction
      • Conversation Summarization
      • Smart Reply Generation
                           │
                           ▼
            Conversational Intelligence Layer
      ─────────────────────────────────────
      • Conversation Summary
      • Emotional Trends
      • Toxicity Alerts
      • AI Suggestions
      • Visual Analytics
```

---

# Tech Stack

## Frontend

- React.js
- Tailwind CSS
- Socket.io Client

---

## Backend

- Node.js
- Express.js
- Socket.io

---

## Database

- MongoDB

---

## Machine Learning

- Python
- PyTorch
- HuggingFace Transformers
- Scikit-Learn
- NLTK
- SpaCy

---

# NLP Models

The system explores multiple transformer-based language models including

- BERT
- RoBERTa
- DistilBERT
- BART
- T5

These models are comparatively evaluated to determine their effectiveness for conversational intelligence tasks.

---

# 📊 Evaluation Metrics

The performance of the implemented modules is evaluated using

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- Inference Latency
- Response Time
- Model Comparison

---

# Research Motivation

Existing communication platforms primarily facilitate messaging but provide limited understanding of conversational context.

Most existing solutions perform isolated NLP tasks such as sentiment analysis or toxicity detection independently.

This project proposes a unified architecture capable of integrating multiple NLP modules into a single conversational intelligence system capable of analyzing digital communication in real time.

---

# Objectives

- Develop a modular NLP pipeline for conversational understanding.
- Perform real-time sentiment and emotion analysis.
- Detect toxicity and offensive language.
- Generate AI-powered conversation summaries.
- Provide intelligent reply suggestions.
- Visualize communication insights.
- Compare multiple NLP models to evaluate performance.

---

# 📂 Repository Structure

```text
├── client/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── assets/
│
├── server/
│   ├── routes/
│   ├── controllers/
│   ├── middleware/
│   ├── models/
│   ├── services/        # NLP service client + async message analysis pipeline
│   └── lib/
│
├── nlp-service/          # Python/FastAPI multi-module NLP engine
│   └── app/
│       ├── api/           # /health, /analyze/* routes
│       ├── context/        # Adaptive Context Activation (Phase 6)
│       ├── models/          # sentiment/ emotion/ toxicity/ summarization/ topic/
│       ├── preprocessing/
│       ├── inference/
│       ├── evaluation/
│       └── schemas/
│
├── training/             # Fine-tuning pipeline scaffold (Phase 9, not yet executed)
│   ├── shared/            # shared Trainer-based fine-tuning utility
│   ├── sentiment/ emotion/ toxicity/   # one train.py per task
│
├── experiments/           # Reproducible experiment framework + real results (Phase 9)
│   ├── common/             # dataset-agnostic metrics + charting
│   ├── datasets_loader.py, baselines.py, transformer_eval.py, context_experiment.py, run_all.py
│   ├── results/             # generated JSON + PNG output from the last run
│   ├── README.md            # how to reproduce
│   └── FINDINGS.md          # research-paper-style write-up of results
│
└── README.md
```

---

# Future Enhancements

- Voice Sentiment Analysis
- Speech Emotion Recognition
- Sarcasm Detection
- Fake News Detection
- Multilingual NLP
- Personalized AI Assistant
- Explainable AI (XAI)
- Federated Learning Support
- LLM Fine-Tuning
- Multi-Agent Conversational Intelligence

---

# Current Status

> ✅ Phase 11 complete; future enhancements remain planned

This repository is being developed as part of a Final Year Engineering Project focused on Conversational AI, Natural Language Processing, and Machine Learning.

**Implementation progress** (see [nlp-service/README.md](nlp-service/README.md) for full detail):

- ✅ Phase 1-2: Existing MERN chat audited and stabilized
- ✅ Phase 3: Python FastAPI NLP service scaffolded
- ✅ Phase 4: First end-to-end pipeline (sentiment) proven Node → Python → model → Node → React
- ✅ Phase 5: Emotion, toxicity, summarization, and topic extraction modules added
- ✅ Phase 6: Adaptive Context Activation (ACA) algorithm implemented and verified against the spec's canonical examples
- ✅ Phase 7: ACA wired into the live per-message pipeline - real chat messages now get context-aware analysis
- ✅ Phase 8: Conversation-level intelligence dashboard (`GET /api/messages/analytics/:id`) - sentiment progression, dominant emotion, toxicity frequency, topic, live summary, context usage stats, and a data-driven sentiment-trend insight
- ✅ Phase 9: Experiment/training/evaluation framework - see [experiments/FINDINGS.md](experiments/FINDINGS.md) for full results. Real, reproducible metrics generated from public datasets: TF-IDF+LogReg baseline vs pretrained Transformer for sentiment/emotion/toxicity, and a full-context vs Adaptive Context Activation comparison on real dialogue data. A fine-tuning pipeline scaffold ([training/](training)) is built and import-verified but not executed (CPU-only fine-tuning deferred, per spec)
- ✅ Phase 10-11: Performance/usability optimization, automated smoke tests, and documentation polish

---

---

# 📜 License

This project is intended for educational and research purposes.

---

<div align="center">

### ⭐ If you found this project interesting, consider starring the repository!

Built with ❤️ using Machine Learning, NLP, and Modern Web Technologies.

</div>
