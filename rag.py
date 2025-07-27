from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Cargar documento
loader = PyPDFLoader("docs/test.pdf")
docs = loader.load()

# 2. Dividir en fragmentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. Crear embeddings y vectorstore
embeddings = OllamaEmbeddings(model="all-minilm")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 4. Recuperador
retriever = vectorstore.as_retriever()

# 5. Modelo LLM
llm = ChatOllama(model="llama3")

# 6. Prompt
prompt = ChatPromptTemplate.from_template(
    """Eres un asistente útil. Usa solo el contexto para responder.

Contexto: {context}

Pregunta: {question}

Respuesta:"""
)

# 7. Cadena RAG
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Probar
if __name__ == "__main__":
    while True:
        query = input("Pregunta: ")
        if query.lower() == "salir":
            break
        print("Respuesta:", rag_chain.invoke(query))