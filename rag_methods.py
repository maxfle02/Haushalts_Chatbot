import os
import dotenv
from time import time
import streamlit as st

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


dotenv.load_dotenv()

# Limit the number of documents to be loaded from the database
DB_DOCS_LIMIT = 10

# Function to stream the response from the LLM
def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})

# --- Indexing Phase ---

def load_doc_to_db():
    # Überprüfe, ob die `loaded_documents.txt` Datei existiert und lade die vorhandenen Namen
    try:
        with open("loaded_documents.txt", "r") as file:
            loaded_documents = {line.strip() for line in file.readlines()}  # Verwende ein Set für schnellere Suche
    except FileNotFoundError:
        loaded_documents = set()  # Falls die Datei nicht existiert

    # Verwende Loader je nach Dokumenttyp
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = [] 
        for doc_file in st.session_state.rag_docs:
            # Prüfe, ob der Dateiname bereits in der `loaded_documents.txt` Datei steht
            if doc_file.name in loaded_documents:
                st.warning(f"The document '{doc_file.name}' already exists in the database and will not be reloaded.")
                continue  # Überspringe dieses Dokument

            # Füge Dokument zur Datenbank hinzu, falls es noch nicht vorhanden ist
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("data", exist_ok=True)
                    file_path = f"./data/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                        # Dokumentname in `loaded_documents.txt` schreiben
                        with open("loaded_documents.txt", "a") as log_file:
                            log_file.write(f"{doc_file.name}\n")

                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error loading document {doc_file.name}: {e}")
                    
                    finally:
                        os.remove(file_path)

                else:
                    st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT}).")

        if docs:
            split_and_load_docs(docs)
            st.toast(f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.", icon="✅")


def initialize_vector_db(docs):
    vector_db = Chroma.from_documents(
        persist_directory="data",
        documents=docs,
        embedding=OpenAIEmbeddings(),
        collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
    )

    # We need to manage the number of collections that we have in memory, we will keep the last 20
    chroma_client = vector_db._client
    collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
    print("Number of collections:", len(collection_names))
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db

def split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )

    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(docs)
    else:
        st.session_state.vector_db.add_documents(document_chunks)

# --- Retrieval Augmented Generation (RAG) Phase ---

def get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the conversation, focusing on the most recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(llm):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """    Du bist ein Haushaltsassistent-Bot und hilfst bei Fragen rund um die Bedienung von Haushaltsgeräten. 
        Deine Hauptaufgabe besteht darin, Nutzern klare und praktische Anweisungen zu geben, die leicht verständlich und sofort umsetzbar sind. 
        Die häufigsten Anfragen betreffen die Bedienung von Waschmaschinen, Geschirrspülern, Kühlschränken, Mikrowellen und anderen technischen Geräten im Haushalt.
            {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})

