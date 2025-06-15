import streamlit as st
import PyPDF2
import docx
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI

# Folders for persistence
UPLOAD_DIR = "uploaded_docs"
CACHE_DIR = "cached_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

SAMPLE_QUESTIONS = [
    "How many annual leave days do I get?",
    "What is the sick leave policy?",
    "Can I carry forward unused leave?",
    "What is the parental leave policy?",
    "What are the rules for resignation?",
]

def chunk_text(text, max_chunk_length=500):
    chunks = []
    for i in range(0, len(text), max_chunk_length):
        chunks.append(text[i:i+max_chunk_length])
    return chunks

def get_embeddings(texts, client, model="text-embedding-3-small"):
    response = client.embeddings.create(input=texts, model=model)
    return [np.array(e.embedding) for e in response.data]

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_pdf_text(file_path):
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def extract_docx_text(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def save_cached_data(filename, text, embeddings):
    # Save text
    with open(os.path.join(CACHE_DIR, filename + ".txt"), "w", encoding="utf-8") as f:
        f.write(text)
    # Save embeddings as numpy array
    np.save(os.path.join(CACHE_DIR, filename + ".npy"), np.array(embeddings, dtype=object), allow_pickle=True)

def load_cached_data(filename):
    # Load text
    with open(os.path.join(CACHE_DIR, filename + ".txt"), "r", encoding="utf-8") as f:
        text = f.read()
    # Load embeddings
    embeddings = np.load(os.path.join(CACHE_DIR, filename + ".npy"), allow_pickle=True)
    return text, embeddings.tolist()

def compute_and_cache_embeddings(filename, file_path, filetype, client):
    # Extract text
    if filetype == "pdf":
        text = extract_pdf_text(file_path)
    elif filetype == "docx":
        text = extract_docx_text(file_path)
    else:
        return None, None
    # Chunk text
    chunks = chunk_text(text)
    # Get embeddings
    embeddings = get_embeddings(chunks, client)
    # Save
    save_cached_data(filename, text, embeddings)
    return text, embeddings

def list_cached_docs():
    txt_files = [f[:-4] for f in os.listdir(CACHE_DIR) if f.endswith(".txt")]
    return txt_files

st.set_page_config(page_title="HR Policy Q&A Chatbot", layout="centered")
st.markdown("<h1 style='text-align: center; margin-bottom: 1em;'>🤖 HR Policy Q&A Chatbot</h1>", unsafe_allow_html=True)

# --- Load API Key ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Please set your OPENAI_API_KEY in your .env file.")
    st.stop()
client = OpenAI(api_key=api_key)

# --- Upload Section ---
uploaded_file = st.file_uploader("Upload a new HR policy document (PDF or DOCX)", type=["pdf", "docx"])
if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext not in ["pdf", "docx"]:
        st.warning("Only PDF and DOCX files are supported.")
    else:
        file_path = save_uploaded_file(uploaded_file)
        filename = uploaded_file.name
        # Compute/cached embeddings
        with st.spinner("Processing and embedding your document (only runs once per file)..."):
            compute_and_cache_embeddings(filename, file_path, file_ext, client)
        st.success(f"Uploaded and processed {filename}")

# --- List and select cached docs ---
cached_docs = list_cached_docs()
if not cached_docs:
    st.info("Please upload at least one policy document to start Q&A.")
    st.stop()

selected_doc = st.selectbox("Choose a document to chat with:", cached_docs)
text, embeddings = load_cached_data(selected_doc)
chunks = chunk_text(text)

st.success(f"Loaded {selected_doc}")

# --- Chat UI Section ---
st.markdown("----")
st.markdown("<h3 style='margin-bottom:0.5em;'>Suggested questions:</h3>", unsafe_allow_html=True)
cols = st.columns(len(SAMPLE_QUESTIONS))
selected_question = ""
for i, q in enumerate(SAMPLE_QUESTIONS):
    if cols[i].button(q, key=f"suggested_{i}"):
        selected_question = q

st.markdown("")
st.subheader("Your question")
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_question = st.text_input("Type your question about HR policies:", value=selected_question, key="user_question_input")
if st.button("Get Answer") and user_question.strip():
    with st.spinner("Searching your HR policy..."):
        # Find most similar chunks
        q_emb = get_embeddings([user_question], client)[0]
        sims = [np.dot(q_emb, np.array(e)) / (np.linalg.norm(q_emb) * np.linalg.norm(np.array(e))) for e in embeddings]
        top_indices = np.argsort(sims)[-3:][::-1]
        top_chunks = [chunks[i] for i in top_indices]
        context = "\n\n".join(top_chunks)

        prompt = (
            f"You are an HR assistant. Answer the user's question ONLY using the information below from the HR policy. "
            f"If the answer is not found, say 'I couldn't find this information in the provided HR policy.'\n\n"
            f"HR Policy Excerpt:\n{context}\n\n"
            f"Question: {user_question}\n"
            f"Answer:"
        )
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are an HR assistant who answers using the company policy document."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content

    st.session_state['chat_history'].append({"question": user_question, "answer": answer})

# --- Chat History Bubbles ---
if st.session_state['chat_history']:
    st.write("---")
    st.markdown("<h4>Chat History</h4>", unsafe_allow_html=True)
    for idx, qa in enumerate(st.session_state['chat_history']):
        align = "left" if idx % 2 == 0 else "right"
        st.markdown(
            f"""
            <div style='
                text-align: {align};
                margin-bottom: 0.6em;
            '>
              <div style='
                  display: inline-block;
                  background: #F0F4FA;
                  padding: 0.9em 1.2em;
                  border-radius: 1.2em;
                  min-width: 15em;
                  max-width: 80%;
                  font-size: 1.08em;
                  box-shadow: 0 2px 6px rgba(0,0,0,0.03);
                  color: #222;
                  margin-bottom: 0.18em;
              '>
                <b>Q:</b> {qa['question']}
              </div><br>
              <div style='
                  display: inline-block;
                  background: #D0E2FB;
                  padding: 0.9em 1.2em;
                  border-radius: 1.2em;
                  min-width: 15em;
                  max-width: 80%;
                  font-size: 1.08em;
                  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
                  color: #222;
                  margin-top: 0.08em;
              '>
                <b>A:</b> {qa['answer']}
              </div>
            </div>
            """, unsafe_allow_html=True
        )
