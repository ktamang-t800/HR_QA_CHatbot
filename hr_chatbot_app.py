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
    chunks = []
    for i in range(0, len(text), max_chunk_length):
        chunks.append(text[i:i+max_chunk_length])
    return chunks

def get_embeddings(texts, client, model="text-embedding-3-small"):
    response = client.embeddings.create(input=texts, model=model)
    return [np.array(e.embedding) for e in response.data]

def find_most_similar_chunks(question, chunks, client, top_k=3):
    all_texts = [question] + chunks
    embeddings = get_embeddings(all_texts, client)
    q_emb, chunk_embs = embeddings[0], embeddings[1:]
    similarities = [np.dot(q_emb, ce) / (np.linalg.norm(q_emb) * np.linalg.norm(ce)) for ce in chunk_embs]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

st.set_page_config(page_title="HR Policy Q&A Chatbot", layout="centered")

# Custom CSS for style improvements
st.markdown("""
    <style>
        .main {background-color: #F8FAFB;}
        .block-container {padding-top: 2rem;}
        .suggested-btn {margin: 0.2em 0.4em 0.2em 0.0em;}
        .stButton>button {
            border-radius: 1.5em;
            padding: 0.6em 1.5em;
            border: 1px solid #555;
            background-color: #F6F8F9;
            transition: background 0.15s;
        }
        .stButton>button:hover {
            background-color: #E3EAF2;
            border: 1.5px solid #2563eb;
            color: #2563eb;
        }
        .stTextInput>div>div>input {
            border-radius: 0.8em;
        }
        .stTextInput>div>div {
            border-radius: 0.8em;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; margin-bottom: 1em;'>🤖 HR Policy Q&A Chatbot</h1>", unsafe_allow_html=True)

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

if all_text:
    st.markdown("----")
    st.markdown("<h3 style='margin-bottom:0.5em;'>Suggested questions:</h3>", unsafe_allow_html=True)
    cols = st.columns(len(SAMPLE_QUESTIONS))
    selected_question = ""
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if cols[i].button(q, key=f"suggested_{i}"):
            selected_question = q

    st.markdown("")

    st.subheader("Your question")
    # Chat session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    user_question = st.text_input("Type your question about HR policies:", value=selected_question, key="user_question_input")

    if st.button("Get Answer") and user_question.strip():
        with st.spinner("Searching your HR policy..."):
            import os
            from dotenv import load_dotenv
            from openai import OpenAI

            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)

            chunks = chunk_text(all_text, max_chunk_length=500)
            top_chunks = find_most_similar_chunks(user_question, chunks, client, top_k=3)
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

        # Save Q&A to chat history
        st.session_state['chat_history'].append({"question": user_question, "answer": answer})


        # --- Chat History with Avatars and Polished Bubbles ---
    if st.session_state['chat_history']:
        st.write("---")
        st.markdown("<h4 style='margin-bottom:1em;'>Chat History</h4>", unsafe_allow_html=True)
        chat_icon_user = "👤"
        chat_icon_ai = "🤖"
        for idx, qa in enumerate(st.session_state['chat_history']):
            st.markdown(
                f"""
                <div style='display: flex; flex-direction: {'row' if idx % 2 == 0 else 'row-reverse'}; align-items: flex-start; margin-bottom: 0.9em;'>
                  <div style='font-size:2em;margin:0 0.7em 0 0.7em;'>{chat_icon_user if idx % 2 == 0 else chat_icon_ai}</div>
                  <div>
                    <div style='
                        display: inline-block;
                        background: {"#F0F4FA" if idx % 2 == 0 else "#E5F6E3"};
                        padding: 0.7em 1.2em;
                        border-radius: 1.4em;
                        min-width: 10em;
                        max-width: 80vw;
                        font-size: 1.08em;
                        color: #222;
                        margin-bottom: 0.16em;
                        box-shadow: 0 1.5px 5px rgba(0,0,0,0.03);
                    '>
                        <b>{'You' if idx % 2 == 0 else 'AI'}:</b> <span>{qa['question'] if idx % 2 == 0 else qa['answer']}</span>
                    </div>
                    <div style='
                        display: inline-block;
                        background: {"#D0E2FB" if idx % 2 == 0 else "#FFF7D6"};
                        padding: 0.7em 1.2em;
                        border-radius: 1.4em;
                        min-width: 10em;
                        max-width: 80vw;
                        font-size: 1.08em;
                        color: #222;
                        margin-top: 0.11em;
                        box-shadow: 0 1.5px 6px rgba(0,0,0,0.06);
                    '>
                        <b>{'AI' if idx % 2 == 0 else 'You'}:</b> <span>{qa['answer'] if idx % 2 == 0 else qa['question']}</span>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )
