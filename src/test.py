import os
import dotenv
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

dotenv.load_dotenv()
CHROMA_DB = "/app/chroma_db"
persist_dir = CHROMA_DB
embedding = OpenAIEmbeddings()

# Prüfen, ob das Verzeichnis existiert
if os.path.exists(persist_dir):
    try:
        # Bestehende Datenbank laden
        vector_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding,
        )
        print("Bestehende Vector DB geladen.")
    except Exception as e:
        raise RuntimeError(f"Fehler beim Laden der Vector DB: {e}")

else:
    print("No directory found")


def get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the conversation, focusing on the most recent messages.",
            ),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(llm):
    retriever_chain = get_context_retriever_chain(vector_db, llm)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """    Du bist ein Haushaltsassistent-Bot und hilfst bei Fragen rund um die Bedienung von Haushaltsgeräten. 
        Deine Hauptaufgabe besteht darin, Nutzern klare und praktische Anweisungen zu geben, die leicht verständlich und sofort umsetzbar sind. 
        Die häufigsten Anfragen betreffen die Bedienung von Waschmaschinen, Geschirrspülern, Kühlschränken, Mikrowellen und anderen technischen Geräten im Haushalt.
            {context}""",
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
        ]
    )
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


from datasets import Dataset
from langchain.schema import HumanMessage, AIMessage

# Define questions and ground truths
questions = [
    "Welche Sicherheitsvorkehrungen sollten beim Aufstellen des Geschirrspülers beachtet werden?",
    "Wie wird der Klarspüler im Geschirrspüler nachgefüllt?",
    "Was sind die empfohlenen Maßnahmen, um Energie und Wasser zu sparen?",
    "Welche Programme sind für stark verschmutztes Geschirr geeignet?",
    "Was ist zu tun, wenn die Wasserzulauf-Anzeige leuchtet?",
    "Welche Reiniger sollten im Geschirrspüler nicht verwendet werden?",
    "Wie wird das Siebsystem des Geschirrspülers gereinigt?",
    "Welche Schritte sind bei der ersten Inbetriebnahme des Geschirrspülers erforderlich?",
    "Wie kann die automatische Türöffnung während der Trocknungsphase aktiviert werden?",
    "Was sind die Voraussetzungen für die Nutzung der Home Connect App?",
]

ground_truths = [
    [
        "Beim Aufstellen des Geschirrspülers ist sicherzustellen, dass die Netzanschlussleitung nicht eingeklemmt oder beschädigt wird, und dass ein Mindestabstand von 5 cm zu Wasserleitungen eingehalten wird."
    ],
    [
        "Der Klarspüler wird bis zur Markierung 'max' in den Vorratsbehälter gefüllt. Übergelaufener Klarspüler sollte aus dem Spülraum entfernt werden, um Schaumbildung zu vermeiden."
    ],
    [
        "Um Energie und Wasser zu sparen, sollte das Eco-50°-Programm genutzt und das Gerät immer vollständig beladen werden. Bei geringer Beladung kann die Zusatzfunktion 'Halbe Beladung' verwendet werden."
    ],
    [
        "Für stark verschmutztes Geschirr wird das Programm 'Intensiv 70°' empfohlen, das speziell für eingebrannte und hartnäckige Speisereste geeignet ist."
    ],
    [
        "Wenn die Wasserzulauf-Anzeige leuchtet, sollten der Wasserhahn geöffnet, der Zulaufschlauch auf Knicke überprüft und die Siebe im Wasseranschluss gereinigt werden."
    ],
    [
        "Handspülmittel und chlorhaltige Reiniger dürfen nicht verwendet werden, da sie zu Schäden am Gerät und Gesundheitsrisiken führen können."
    ],
    [
        "Das Siebsystem wird gereinigt, indem es aus dem Gerät entnommen, die Siebe unter fließendem Wasser gereinigt und anschließend wieder korrekt eingesetzt werden."
    ],
    [
        "Vor der ersten Nutzung müssen Spezialsalz und Klarspüler eingefüllt, die Enthärtungsanlage eingestellt und das Gerät mit einem Programm bei höchster Temperatur ohne Geschirr betrieben werden."
    ],
    [
        "Die automatische Türöffnung während der Trocknungsphase kann in den Grundeinstellungen aktiviert werden, indem die Option 'o01' oder 'o02' ausgewählt wird."
    ],
    [
        "Für die Nutzung der Home Connect App muss das Gerät mit einem WLAN-Heimnetzwerk verbunden sein. Die App führt Schritt für Schritt durch den Anmeldeprozess."
    ],
]

# Initialize RAG Chain
llm_stream_openai = ChatOpenAI(
    model="gpt-4o",  # Use the appropriate model, e.g., "o1-preview"
    temperature=0,
    streaming=True,
)

llm_stream = llm_stream_openai
conversation_rag_chain = get_conversational_rag_chain(llm_stream)

# Ensure vector_db is correctly initialized and returns a retriever object
retriever = (
    vector_db.as_retriever()
)  # Ensure this returns a retriever instance with `get_relevant_documents`

answers = []
contexts = []

# Perform inference and context retrieval
for query in questions:
    # Prepare conversation messages
    messages = [HumanMessage(content=query)]
    response_message = ""
    for chunk in conversation_rag_chain.pick("answer").stream(
        {"messages": messages[:-1], "input": messages[-1].content}
    ):
        response_message += chunk

    answers.append(response_message)

    # Retrieve relevant documents for context
    # Retrieve relevant documents for context
    try:
        retrieved_docs = retriever.invoke(query)
        contexts.append([doc.page_content for doc in retrieved_docs])
    except AttributeError as e:
        print(f"Error retrieving documents: {e}")
        contexts.append([])  # Append an empty context list in case of failure


# Convert to dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths,
}
data

# # Convert dict to dataset
dataset = Dataset.from_dict(data)

# # Inspect dataset
print(dataset)

# Rename and restructure the dataset
data_fixed = {
    "question": data["question"],  # Keep the questions as-is
    "answer": data["answer"],  # Keep the answers as-is
    "retrieved_contexts": data["contexts"],  # Rename 'contexts' to 'retrieved_contexts'
    "reference": [
        " ".join(gt) for gt in data["ground_truths"]
    ],  # Convert 'ground_truths' lists to strings
}

# Create a new Dataset object
dataset_fixed = Dataset.from_dict(data_fixed)

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# Evaluate the fixed dataset
result = evaluate(
    dataset=dataset_fixed,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

# Convert the result to a pandas DataFrame for easier inspection
df = result.to_pandas()

# Display the results
print(df)
