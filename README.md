# IntelliDoc AI 📄🤖
### Intelligent PDF Question Answering System Using Retrieval-Augmented Generation (RAG)

IntelliDoc AI is an AI-powered document assistant that allows users to upload PDF documents and ask questions about their content using **Retrieval-Augmented Generation (RAG)**.

The application extracts text from PDF documents, divides it into smaller chunks, generates vector embeddings, and stores them in ChromaDB. When a user asks a question, the system retrieves relevant document content and uses a locally running Large Language Model (LLM) through Ollama to generate context-aware answers with source references.

## ✨ Key Features

- **PDF Upload:** Upload and process PDF documents.
- **Text Extraction:** Extract text and page information using PyPDF.
- **Text Chunking:** Divide extracted text into smaller, manageable chunks.
- **Embedding Generation:** Generate vector embeddings using Ollama.
- **Vector Database:** Store document embeddings in ChromaDB.
- **Semantic Search:** Retrieve relevant document content based on user questions.
- **AI-Powered Responses:** Generate context-aware answers using RAG.
- **Source Citations:** Display relevant source excerpts and page references.
- **Streaming Responses:** Display AI-generated answers progressively.
- **Local AI Processing:** Run language and embedding models locally using Ollama.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Streamlit | User interface |
| PyPDF | PDF text extraction |
| LangChain Text Splitters | Document chunking |
| Ollama | Local LLM runtime |
| Llama 3.2 (3B) | Answer generation |
| Nomic Embed Text | Text embedding generation |
| ChromaDB | Vector database |
| Requests | Communication with Ollama API |
| Pytest | Application testing |

## 🏗️ System Architecture

IntelliDoc AI follows a Retrieval-Augmented Generation (RAG) architecture.

### Document Processing Workflow

1. **PDF Upload:** The user uploads a PDF document.
2. **Text Extraction:** Text and page information are extracted from the PDF.
3. **Text Chunking:** The extracted content is divided into smaller chunks.
4. **Embedding Generation:** Each chunk is converted into a numerical vector representation.
5. **Vector Storage:** The embeddings and metadata are stored in ChromaDB.

### Question Answering Workflow

1. **User Query:** The user enters a question about the uploaded document.
2. **Query Embedding:** The question is converted into a vector embedding.
3. **Semantic Retrieval:** ChromaDB retrieves the most relevant document chunks.
4. **Context Preparation:** Retrieved information is prepared as context.
5. **Answer Generation:** The context and question are passed to the LLM through Ollama.
6. **Response Display:** The application displays the generated answer along with source references.

## 📂 Project Structure

```text
IntelliDoc-AI/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
│
├── components/
│   ├── sidebar.py
│   ├── chat_interface.py
│   └── document_info.py
│
├── config/
│   └── settings.py
│
├── core/
│   ├── pdf_processor.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── rag_pipeline.py
│
├── utils/
│   └── helpers.py
│
├── tests/
│   ├── test_pdf_processor.py
│   └── test_rag_pipeline.py
│
├── data/
├── vector_db/
│
└── .streamlit/
    └── config.toml
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or above
- Ollama
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/agnass13/IntelliDoc-AI
cd IntelliDoc-AI
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

Download Ollama from:

https://ollama.com/

Install and start Ollama on your system.

### 4. Download Required AI Models

Run the following commands:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

### 5. Configure Environment Variables

Create a `.env` file using `.env.example` as a reference.

Default configuration:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
