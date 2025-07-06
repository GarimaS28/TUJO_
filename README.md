# TUJO: AI-Powered Mood Detection Diary

A full-stack AI-based mental wellness app that enables users to log their daily thoughts, detect their mood using deep learning, and receive personalized activity recommendations — all visualized through interactive mood trends.

---

## 🧠 Overview

TUJO is an intelligent diary application built as a 2025 course project. It leverages state-of-the-art natural language processing to analyze user journal entries and classify them into emotional categories. The app provides weekly mood trends and tailored wellness recommendations based on user behavior, mood, and profession.

---

## 🚀 Features

- 📝 **Daily Journal Logging** – Users can write their thoughts and save them securely.
- 🤖 **Emotion Detection** – Real-time mood classification using a custom fine-tuned BERT model.
- 📈 **Mood Trends Dashboard** – Visualizes weekly mood graphs using Chart.js.
- 🎯 **Personalized Suggestions** – Activity recommendations based on mood, profession, and interests.
- 🔐 **Authentication** – Secure signup/login with JWT-based session management.
- 🌐 **Responsive Full-Stack App** – Flask + MongoDB backend with Node.js auth and React frontend.

---

## 🧩 Tech Stack

| Layer       | Technology                                  |
|-------------|---------------------------------------------|
| **Frontend**| React.js, Chart.js                          |
| **Backend** | Flask (AI API), Node.js & Express (Auth)    |
| **Database**| MongoDB                                     |
| **ML/NLP**  | PyTorch, HuggingFace Transformers, BERT     |
| **Auth**    | JWT (JSON Web Tokens)                       |

---

## 🧪 AI Model Details

- **Model**: Fine-tuned BERT-based classifier
- **Dataset**: Custom + open-source emotional text datasets
- **Emotions**: Joy, Sadness, Anger, Fear, Surprise, Disgust, Trust, Anticipation
- **Serving**: Real-time inference via Flask API

---

## 📊 Mood Dashboard

Users can see their emotional journey through a **weekly mood trend graph**, helping them identify emotional patterns and take proactive mental wellness steps.

---

## 🔐 Authentication & User Personalization

- JWT-secured login and signup
- User metadata (profession, interests) collected at signup
- Personalized activity suggestions based on metadata + mood

---


