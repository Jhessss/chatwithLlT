import streamlit as st
import os
from googletrans import Translator
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# Função para carregar documentos PDF
def load_documents(uploaded_files):
    # Diretório temporário para salvar os arquivos PDF
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)

    # Salva os arquivos PDF temporariamente
    temp_files = []
    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        temp_file = open(temp_file_path, "wb")
        temp_file.write(uploaded_file.read())
        temp_file.close()
        temp_files.append(temp_file_path)

    # Carrega os documentos PDF
    pdf_loader = PyPDFLoader(file_path=temp_files[0])  # Passa apenas o primeiro arquivo
    documents = pdf_loader.load()

    # Remove os arquivos temporários
    for temp_file_path in temp_files:
        os.remove(temp_file_path)

    return documents

# Interface do usuário para upload de arquivos PDF
uploaded_files = st.file_uploader("Faça o upload do(s) arquivo(s) PDF:", type=["pdf"], accept_multiple_files=True)

# Inicializa ou recupera a lista de documentos
if 'documents' not in st.session_state:
    st.session_state['documents'] = []

# Verifica se foram feitos uploads de arquivos
if uploaded_files:
    # Carrega os documentos PDF
    documents = load_documents(uploaded_files)

    # Adiciona os novos documentos à lista existente
    st.session_state['documents'].extend(documents)

# Restante do código
# ...
# Criação de embeddings
# Criação de embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': "cpu"})

# Divisão de texto em trechos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(st.session_state['documents'])

# Verifica se há pelo menos um texto dividido antes de prosseguir
if not text_chunks:
    st.error("Nenhum texto foi encontrado nos documentos. Verifique se os arquivos PDF estão corretos.")
    st.stop()

# Verifica se há pelo menos um embedding antes de prosseguir
if not embeddings:
    st.error("Nenhum embedding foi gerado. Verifique o modelo de embeddings.")
    st.stop()

# Vectorstore
vector_store = FAISS.from_documents(text_chunks, embeddings)


# Criação de LLM
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama",
                    config={'max_new_tokens': 128, 'temperature': 0.01})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                              memory=memory)

translator = Translator()
def translate_to_portuguese(text):
    translation = translator.translate(text, dest='pt')
    return translation.text

st.title("Chat Laudos usando Llama2")
def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Pergunte sobre os Laudos."]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Olá! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Pergunta:", placeholder="Faça sua pergunta sobre o laudo aqui.", key='input')
            submit_button = st.form_submit_button(label='Enviar')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Inicialização da sessão
initialize_session_state()
# Exibe o histórico do chat
display_chat_history()