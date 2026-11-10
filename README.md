# 🩺 MedQueryDoc AI

**Stop Ctrl+F-ing through medical PDFs. Ask in plain English, get answers from Llama-2.**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![Pinecone](https://img.shields.io/badge/Vector%20DB-Pinecone-00B3A4)
![Flask](https://img.shields.io/badge/Flask-Web%20UI-black?logo=flask)
![License](https://img.shields.io/badge/License-GPLv3-green)

<!-- 👇 REPLACE THIS LINE: drag your demo GIF or screenshot into the GitHub editor and it will upload automatically -->
![MedQueryDoc AI demo](https://github.com/user-attachments/assets/YOUR-DEMO-GIF-HERE)

> ⚠️ **For educational purposes only. This is not medical advice.**

---

## ✨ Why this exists

Medical PDFs are long, dense, and painful to search. MedQueryDoc AI lets you **chat with them**: ask a question, and it finds the right passages and answers in plain language, running **Llama-2 locally** (no OpenAI bills).

## 🧠 Features

- 📄 Ingest any medical PDF
- 🔍 Fast semantic search with Pinecone
- 🤖 Retrieval-Augmented Generation (RAG) with LangChain
- 💻 Local Llama-2 inference, no paid API required
- 🌐 Clean chat interface built with Flask
- 🧩 Modular code that is easy to extend

## 🛠️ Built with

Python · LangChain · Flask · Meta Llama-2 · Pinecone · Hugging Face Embeddings

## 🔄 How it works

```
PDFs → split into chunks → embeddings → Pinecone
                                          ↓
Your question → similarity search → top chunks → Llama-2 → answer
```

---

## 🚀 Quick start

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/Medical-Chatbot-using-Llama-2.git
cd Medical-Chatbot-using-Llama-2
```

### 2. Create a virtual environment

Conda:

```bash
conda create -p ./venv python=3.10 -y
conda activate ./venv
```

Or venv:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Pinecone key

Create a `.env` file in the root directory:

```ini
PINECONE_API_KEY=your_pinecone_api_key_here
```

> The new Pinecone SDK no longer needs `PINECONE_API_ENV`.

### 5. Add your documents

```
data/
├── medical_doc_1.pdf
├── medical_doc_2.pdf
```

### 6. Download the model

Download `llama-2-7b-chat.ggmlv3.q4_0.bin` from [TheBloke/Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML) and place it in the `model/` folder.

### 7. Index the documents

```bash
python store_index.py
```

This loads the PDFs, splits them into chunks, generates embeddings, and stores the vectors in Pinecone.

### 8. Run the app

```bash
python app.py
```

Open **http://localhost:8080** and start asking questions.

---

## 🏗️ Project structure

```
├── app.py                 # Flask application
├── store_index.py         # PDF ingestion & vector indexing
├── src/
│   ├── helper.py          # PDF loading, chunking, embeddings
│   └── prompt.py          # Prompt template
├── model/
│   └── llama-2-7b-chat.ggmlv3.q4_0.bin
├── templates/
│   └── chat.html
├── static/
│   └── style.css
├── data/                  # PDF documents
└── .env
```

## 🚧 Roadmap

- [ ] Source citations in answers
- [ ] Document upload from the UI
- [ ] Streaming responses
- [ ] Multilingual medical support
- [ ] Cloud-hosted inference
- [ ] Authentication & access control

## 🤝 Contributing

Ideas and pull requests are welcome. Open an issue to discuss what you'd like to change.

## ⭐ Like it?

If this saved you some scrolling, **drop a star** on the repo. It helps other developers find it.

## 📜 License

Licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

Built on open-source work in LLMs, NLP, and vector databases. Thanks to the community that makes this possible.
