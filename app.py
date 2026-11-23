"""Minimal Streamlit interface for the local document chatbot."""

import hashlib

import ollama
import streamlit as st

from rag import MODEL, DocumentIndex, ask

st.set_page_config(page_title="Document Chat", page_icon="📄")
st.title("Document Chat")

try:
    ollama.show(MODEL)
except ollama.ResponseError:
    st.error(f"Model '{MODEL}' is missing. Run `ollama pull {MODEL}` in a terminal.")
    st.stop()
except Exception as exc:
    st.error(f"Cannot connect to Ollama. Make sure it is running. Details: {exc}")
    st.stop()

state = st.session_state
state.setdefault("index", None)
state.setdefault("doc_key", None)
state.setdefault("messages", [])

uploaded = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

if uploaded is None:
    if state.index is not None:
        state.index.close()
    state.index, state.doc_key, state.messages = None, None, []
else:
    file_bytes = uploaded.getvalue()
    # Hash contents so replacing a file with the same name and size still reindexes it.
    file_key = (uploaded.name, hashlib.sha256(file_bytes).hexdigest())
    if state.doc_key != file_key:
        if state.index is not None:
            state.index.close()
        state.index, state.doc_key, state.messages = None, None, []
        try:
            with st.spinner("Reading and indexing your document..."):
                state.index = DocumentIndex(uploaded.name, file_bytes)
            state.doc_key = file_key
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error("Couldn't index the document. Technical details:")
            st.exception(exc)

if st.button("Clear conversation"):
    state.messages = []

for message in state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("pages"):
            st.caption("Source: page " + ", ".join(map(str, message["pages"])))

prompt = st.chat_input(
    "Ask something about the document" if state.index else "Upload a document to start",
    disabled=state.index is None,
)

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Looking through the document..."):
                answer, pages = ask(state.index, prompt, state.messages)
        except Exception as exc:
            st.error("Couldn't get an answer. Technical details:")
            st.exception(exc)
        else:
            st.write(answer)
            if pages:
                st.caption("Source: page " + ", ".join(map(str, pages)))
            state.messages.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer, "pages": pages},
            ])
