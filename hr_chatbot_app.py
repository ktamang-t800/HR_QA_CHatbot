import streamlit as st
import PyPDF2
import docx
import numpy as np

SAMPLE_QUESTIONS = [
    "How many annual leave days do I get?",
    "What is the sick leave policy?",
    "Can I carry forward unused leave?",
    "What is the parental leave policy?",
    "What are the rules for resignation?",
]
 
def chunk_text(text, max_chunk_length=500):
    """Split text into small chunks, hard cut every N characters."""
    chunks = []
    for i in range(0, len(text), max_chunk_length):
        chunks.append(text[i:i+max_chunk_length])
    return chunks

def get_embeddings(texts, client, model="text-embedding-3-small"):
    """Get embeddings for a list of texts."""
    response = client.embeddings.create(input=texts, model=model)
    return [np.array(e.embedding) for e in response.data]

def find_most_similar_chunks(question, chunks, client, top_k=3):
    """Find top_k chunks most similar to the question."""
    all_texts = [question] + chunks
    embeddings = get_embeddings(all_texts, client)
    q_emb, chunk_embs = embeddings[0], embeddings[1:]
    # Compute cosine similarity
    similarities = [np.dot(q_emb, ce) / (np.linalg.norm(q_emb) * np.linalg.norm(ce)) for ce in chunk_embs]
    # Get top_k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

st.set_page_config(page_title="HR Policy Q&A Chatbot")

st.title("HR Policy Q&A Chatbot")

uploaded_files = st.file_uploader(
    "Upload your HR policy documents (PDF or DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

def extract_pdf_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def extract_docx_text(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

all_text = ""
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith(".pdf"):
            file_text = extract_pdf_text(uploaded_file)
        elif uploaded_file.name.lower().endswith(".docx"):
            file_text = extract_docx_text(uploaded_file)
        else:
            file_text = ""
        all_text += f"\n\n---- {uploaded_file.name} ----\n\n" + file_text

    st.success(f"Uploaded {len(uploaded_files)} files!")
# User Q&A Section
if all_text:
    st.write("---")
    st.subheader("Ask a question about your HR policies:")

    # FAQ AUTO-SUGGEST FEATURE STARTS HERE
    st.markdown("#### Suggested questions:")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if cols[i].button(q):
            st.session_state['user_question'] = q

    user_question = st.text_input(
        "Your question",
        placeholder="e.g., What is the leave policy?",
        value=st.session_state.get('user_question', "")
    )
    # FAQ AUTO-SUGGEST FEATURE ENDS HERE

    if st.button("Get Answer") and user_question.strip():
        with st.spinner("Searching your HR policy..."):
            import os
            from dotenv import load_dotenv
            from openai import OpenAI

            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)

            # 1. Chunk the text
            chunks = chunk_text(all_text, max_chunk_length=500)

            # 2. Find top relevant chunks using embeddings
            top_chunks = find_most_similar_chunks(user_question, chunks, client, top_k=3)
            context = "\n\n".join(top_chunks)

            # 3. Ask the question using only the relevant context
            prompt = (
                f"You are an HR assistant. Answer the user's question ONLY using the information below from the HR policy. "
                f"If the answer is not found, say 'I couldn't find this information in the provided HR policy.'\n\n"
                f"HR Policy Excerpt:\n{context}\n\n"
                f"Question: {user_question}\n"
                f"Answer:"
            )

            response = client.chat.completions.create(
                model="gpt-4.1-nano",  # use your working model here!
                messages=[
                    {"role": "system", "content": "You are an HR assistant who answers using the company policy document."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content

        st.success("Answer:")
        st.write(answer)
        # Optionally clear the pre-filled question after answer
        st.session_state['user_question'] = ""
