"""
###############################################################################
¿QUÉ ES UN SISTEMA RAG?
------------------------------------------------------------------------------
RAG = Retrieval-Augmented Generation (Generación Aumentada con Recuperación).
Consiste en dos fases:
1. Recuperación: encontrar los fragmentos más relevantes dentro de tus
   documentos para una pregunta dada.
2. Generación: un LLM crea la respuesta final a partir de esos fragmentos,
   reduciendo al mínimo las “alucinaciones”.

FLUJO CONCEPTUAL
----------------
Usuario → Pregunta
        ↓
Retriever (busca similitud en vectores)
        ↓
LLM + Prompt con contexto → Respuesta
------------------------------------------------------------------------------
En este script todo ocurre localmente con:
- PyPDFLoader      → lee PDFs
- RecursiveCharacterTextSplitter → trocea el texto
- Chroma           → base vectorial
- all-minilm       → modelo de embeddings
- Llama 3 (Ollama) → LLM
###############################################################################
"""

# -------------------------------------------------------------------
# 1. IMPORTACIONES
#    Cada módulo cumple un papel específico dentro del pipeline RAG.
# -------------------------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader       # Lee PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter # Trocea texto
from langchain_chroma import Chroma                                # BD vectorial
from langchain_ollama import OllamaEmbeddings, ChatOllama          # Embeds + LLM
from langchain_core.prompts import ChatPromptTemplate              # Prompt template
from langchain_core.runnables import RunnablePassthrough           # Passthrough
from langchain_core.output_parsers import StrOutputParser          # Parseo a str

# -------------------------------------------------------------------
# 2. CARGAR EL DOCUMENTO
#    PyPDFLoader convierte cada página del PDF en un objeto Document
#    con atributos: page_content (str) y metadata (dict).
# -------------------------------------------------------------------
loader = PyPDFLoader("docs/test.pdf")
docs = loader.load()  # Lista de Document, 1 por página

# -------------------------------------------------------------------
# 3. DIVIDIR EN FRAGMENTOS (CHUNKS)
#    Teoría de los chunks:
#    - chunk_size   ≈ nº de caracteres por trozo.
#    - chunk_overlap= cuántos caracteres se repiten entre trozos
#                     para conservar contexto.
#    Objetivo: que cada trozo quepa en el contexto del LLM y a la vez
#              mantenga coherencia semántica.
# -------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # ~1000 caracteres por chunk
    chunk_overlap=200  # 200 caracteres de solapamiento
)
splits = text_splitter.split_documents(docs)  # Lista más grande de trozos pequeños

# -------------------------------------------------------------------
# 4. CREAR EMBEDDINGS Y VECTORSTORE
#    - Embeddings: vectores densos que representan el significado
#      semántico de cada chunk (modelo all-minilm).
#    - Chroma: base de datos vectorial que indexa esos vectores y
#      permite búsqueda por similitud coseno.
# -------------------------------------------------------------------
embeddings = OllamaEmbeddings(model="all-minilm")  # Modelo local de Ollama
vectorstore = Chroma.from_documents(
    documents=splits,      # Todos los trozos
    embedding=embeddings   # Función para convertir texto → vector
)

# -------------------------------------------------------------------
# 5. RECUPERADOR
#    Objeto que, dada una pregunta, retorna los k trozos más similares
#    (por defecto k=4). Internamente usa la similitud coseno entre
#    el embedding de la pregunta y los embeddings de los trozos.
# -------------------------------------------------------------------
retriever = vectorstore.as_retriever()  # k=4 por defecto

# -------------------------------------------------------------------
# 6. MODELO LLM
#    Llama 3 ejecutado localmente vía Ollama. Genera la respuesta
#    final a partir del prompt completo.
# -------------------------------------------------------------------
llm = ChatOllama(model="llama3")

# -------------------------------------------------------------------
# 7. PLANTILLA DE PROMPT
#    Define cómo se ensambla el prompt final:
#    - {context}  : trozos recuperados (formateados en texto plano)
#    - {question} : pregunta del usuario
# -------------------------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """Eres un asistente útil. Usa solo el contexto para responder.

Contexto: {context}

Pregunta: {question}

Respuesta:"""
)

# -------------------------------------------------------------------
# 8. FUNCIÓN AUXILIAR
#    Une los trozos recuperados en un único string para el prompt.
# -------------------------------------------------------------------
def format_docs(docs):
    """
    Recibe una lista de Document y devuelve su contenido concatenado.
    """
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------------------------------------------
# 9. CADENA RAG
#    Pipeline declarativo de LangChain:
#    - Paso 1: recibe la pregunta.
#    - Paso 2: retriever obtiene trozos relevantes.
#    - Paso 3: format_docs une los trozos.
#    - Paso 4: se inyectan en el prompt.
#    - Paso 5: Llama 3 genera la respuesta.
#    - Paso 6: se parsea a string plano.
# -------------------------------------------------------------------
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------------------------------------------------------
# 10. BUCLE INTERACTIVO
#    Permite probar la cadena en consola.
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("🔍 RAG activo. Escribe 'salir' para terminar.\n")
    while True:
        query = input("Pregunta: ")
        if query.lower() == "salir":
            break
        respuesta = rag_chain.invoke(query)
        print("Respuesta:", respuesta, "\n")