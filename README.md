# Document Chat

I wanted a simple way to ask questions about a document without having to search through it myself. So I built this.

Upload a PDF, Word document, or text file. Ask a question in plain English. The chatbot looks through the document and answers using what it finds there.

If the answer isn't in the document, it's supposed to say so instead of making something up.

Everything runs locally with Ollama and Llama 3.2. No paid API or API key needed.

## What it can do

- Read PDF, DOCX, and TXT files
- Answer questions using the uploaded document rather than general knowledge
- Handle follow-up questions such as "What about the other project?"
- Compare information found in different parts of a document
- Show supporting page numbers for PDFs when available
- Tell you when the requested information isn't there

You can upload one document at a time. There's also a button to clear the conversation and start fresh.

## Getting it running

You'll need Python, Ollama, and the files in this repository. I built and tested it using Python 3.11 in a Conda environment on Windows.

**1. Get Ollama running**

Install Ollama from ollama.com, open it, and download the model.

```powershell
ollama pull llama3.2
```

You can check that the model is installed with `ollama list`. Keep Ollama running while using the chatbot.

**2. Set up Python**

Open a terminal in the project folder. If you use Conda, these commands will create an environment and install the required packages.

```powershell
conda create -n rag_env python=3.11 -y
conda activate rag_env
python -m pip install -r requirements.txt
```

If you already have the environment, just activate it and install the requirements.

**3. Start the chatbot**

```powershell
python -m streamlit run app.py
```

Open the local address Streamlit shows in your terminal, upload a file, and start asking questions.

The first run may take a little longer because ChromaDB downloads its embedding model. That download is normally a one-time thing.

## What's happening behind the scenes

The app extracts text from your file and breaks it into smaller, overlapping pieces. ChromaDB stores embeddings for those pieces so the chatbot can find relevant information when you ask something.

For shorter documents, it sends the full extracted pages to Llama 3.2. That helped with a problem I ran into early on, where the chatbot would miss details that were clearly present. For longer documents, it retrieves the eight most relevant chunks.

Llama 3.2 then uses the retrieved text to put together an answer. The app also tries to keep track of what you're referring to in follow-up questions, so "that project" doesn't accidentally become a different project.

The interface is built with Streamlit. The document index lives in memory, so you'll need to upload the file again after restarting the app.

## How I tested it

My first working version wasn't particularly reliable. It sometimes missed an entire education entry, mixed up details from two different projects, or gave a graduation year when asked for an enrollment date.

I kept testing and fixing those problems. On a set of 20 questions using a resume, the scores improved from **34/40 to 37/40 to 40/40**.

Those questions covered factual answers, comparisons, follow-up questions, missing information, and attempts to make the model invent details.

That's a result from one test document, not a claim that the chatbot will get everything right. I'd still expect to find edge cases with longer or more complicated files.

## Things to know

- Scanned PDFs aren't supported yet. The PDF needs to contain extractable text.
- Answers can still be wrong or incomplete. It's a small local model, not a perfect fact checker.
- Long documents rely on retrieval, which can miss a relevant passage.
- PDF page references depend on the model identifying the excerpts it used.
- The app doesn't keep a permanent document library. Uploads are indexed in memory.

## Project files

- `app.py` handles the Streamlit interface and chat history.
- `rag.py` handles document reading, retrieval, conversation context, and answers.
- `requirements.txt` lists the Python packages.
- `.gitignore` keeps local environment files out of the repository.

I kept this project intentionally small. I wanted to understand how the pieces of a RAG application fit together without turning it into something unnecessarily complicated.
