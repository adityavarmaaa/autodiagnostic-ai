# 🚗 AutoDiagnostic AI

> **An AI-powered automotive diagnostic assistant built with RAG, vector search, local LLMs, and web-based knowledge retrieval.**

AutoDiagnostic AI is an end-to-end automotive diagnostic system that takes a **vehicle, complaint, and diagnostic trouble code (DTC)** and uses a retrieval-first AI pipeline to generate an evidence-grounded response.

Instead of asking an LLM to answer everything from its pretrained memory, the system first searches a persistent automotive knowledge base. When sufficient information is not available locally, it falls back to web search, processes the retrieved information, stores the new knowledge in ChromaDB, and then gives the relevant evidence to a local LLM for final generation.

---

# 🎯 Problem

Automotive diagnostic information is spread across:

- Vehicle manuals
- Diagnostic documents
- Technical information
- Repair information
- Owner complaints
- Troubleshooting discussions
- Web sources

A normal LLM-based chatbot may not have the exact information required for a specific vehicle, model year, complaint, or DTC.

The goal of AutoDiagnostic AI is therefore:

> **Retrieve relevant automotive knowledge first, then use an LLM to reason over that evidence.**

---

# 🧠 Core Idea

The system follows:

```text
USER QUESTION
     ↓
SEARCH KNOWLEDGE
     ↓
IS USEFUL INFORMATION AVAILABLE?
     │
     ├── YES
     │    ↓
     │  USE LOCAL EVIDENCE
     │
     └── NO
          ↓
      SEARCH WEB
          ↓
      PROCESS WEB DATA
          ↓
        CHUNK
          ↓
       EMBED
          ↓
   STORE IN CHROMADB
          ↓
      USE EVIDENCE
          ↓
      LOCAL LLM
          ↓
       RESPONSE




                                ┌───────────────────────┐
                         │         USER          │
                         │                       │
                         │ Vehicle               │
                         │ Complaint             │
                         │ DTC                   │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │       STREAMLIT       │
                         │     User Interface    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │       DIAGNOSTIC SERVICE       │
                    │          ORCHESTRATOR           │
                    └───────────────┬────────────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
               ┌──────────────────┐   ┌──────────────────┐
               │     CHROMADB     │   │   TAVILY SEARCH  │
               │ Persistent       │   │   Web Retrieval  │
               │ Knowledge Base   │   └────────┬─────────┘
               └────────┬─────────┘            │
                        │                      ▼
                        │             ┌──────────────────┐
                        │             │ Web Documents    │
                        │             └────────┬─────────┘
                        │                      │
                        │                      ▼
                        │             ┌──────────────────┐
                        │             │ Clean + Chunk    │
                        │             └────────┬─────────┘
                        │                      │
                        │                      ▼
                        │             ┌──────────────────┐
                        │             │   Embeddings     │
                        │             │  embeddinggemma  │
                        │             └────────┬─────────┘
                        │                      │
                        │                      ▼
                        │             ┌──────────────────┐
                        └────────────►│     CHROMADB     │
                                      │ Persistent KB    │
                                      └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │ Relevant Evidence│
                                      │     Context      │
                                      └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │      OLLAMA      │
                                      │    phi3:mini     │
                                      └────────┬─────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │ Final Diagnostic │
                                      │     Response     │
                                      └──────────────────┘





END-TO-END PIPELINE

The complete pipeline is:

User Input
   ↓
Build Diagnostic Query
   ↓
Search Persistent Knowledge Base
   ↓
Semantic Similarity Retrieval
   ↓
Vehicle Relevance Filtering
   ↓
Check Whether Useful Evidence Exists
   │
   ├──────────── YES ────────────┐
   │                             │
   │                             ▼
   │                      Retrieved Evidence
   │
   └──────────── NO ─────────────┐
                                 │
                                 ▼
                          Tavily Web Search
                                 ↓
                         Collect Web Results
                                 ↓
                           Clean Web Text
                                 ↓
                              Chunking
                                 ↓
                       Vehicle Relevance Check
                                 ↓
                            Embeddings
                                 ↓
                       Persistent ChromaDB
                                 ↓
                         Retrieved Evidence
                                 │
                                 └───────────────┐
                                                 ▼
                                        Build LLM Context
                                                 ↓
                                        Ollama / phi3:mini
                                                 ↓
                                        Generate Response
                                                 ↓
                                             Streamlit



🚘 1. USER INPUT

The user provides three important pieces of information:

Vehicle
Complaint
DTC

Example:

Vehicle:
Maruti Baleno 2024

Complaint:
Engine warning light is ON

DTC:
P3339

The application sends this information to the backend diagnostic service.

🔎 2. QUERY CONSTRUCTION

The backend creates a diagnostic search query from the user's inputs.

Conceptually:

"Maruti Baleno 2024"
"Engine warning light is ON"
"P3339"
automotive diagnostic technical repair

The web-search layer also normalizes the query toward diagnostic information such as:

common problems
owner complaints
reported issues
faults
warning signs
troubleshooting
repair
service issues
forum

This helps the search focus on real diagnostic information, instead of only vehicle specifications.

📚 3. LOCAL KNOWLEDGE BASE

Before searching the web, the system searches its persistent knowledge base.

The knowledge base is powered by:

ChromaDB

Documents can come from automotive manuals and other ingested sources.

📄 DOCUMENT INGESTION PIPELINE

Local documents follow:

Automotive Document
       ↓
Text Extraction
       ↓
Text Cleaning
       ↓
Chunking
       ↓
Embedding Generation
       ↓
ChromaDB

For example:

100-page manual
      ↓
many smaller chunks
      ↓
embedding for each chunk
      ↓
stored in ChromaDB

We do not send an entire manual to the LLM for every question.

Instead, we retrieve only the relevant pieces.

🧩 4. CHUNKING

Chunking means breaking large documents into smaller pieces.

Example:

Large Manual
    ↓
┌───────────────┐
│ Chunk 1       │
├───────────────┤
│ Chunk 2       │
├───────────────┤
│ Chunk 3       │
├───────────────┤
│ Chunk 4       │
└───────────────┘

Why?

Because the system needs to find the specific information related to the user's question rather than searching the entire manual every time.

🧠 5. EMBEDDINGS

Text is converted into vectors using:

embeddinggemma

Conceptually:

"Engine warning light is ON"
           ↓
      Embedding Model
           ↓
     [0.12, -0.43, ...]

The vector represents the semantic meaning of the text.

This allows the system to search by meaning, not only exact keywords.

🗃️ 6. CHROMADB VECTOR SEARCH

The embeddings are stored in:

ChromaDB

When a user asks a question:

User Query
    ↓
Create Query Embedding
    ↓
Search ChromaDB
    ↓
Find Similar Knowledge Chunks

For example:

User:
"Engine warning light is ON"

Can retrieve:
"The malfunction indicator lamp illuminates
when the ECU detects a system fault."

The wording does not need to be identical.

🚘 7. VEHICLE RELEVANCE FILTERING

Semantic similarity alone is not always enough.

A query about:

Maruti Baleno

could potentially retrieve generic information from:

Hyundai
Toyota
Honda

because they may contain similar terms such as:

engine
battery
warning light
ECU
DTC

Therefore, the diagnostic pipeline also checks vehicle relevance.

Conceptually:

Semantic Similarity
        +
Vehicle Relevance
        ↓
Better Evidence
✅ 8. LOCAL KNOWLEDGE DECISION

After searching ChromaDB, the system checks:

Is the retrieved evidence useful enough?
If YES
ChromaDB
   ↓
Relevant Evidence
   ↓
Continue to LLM
If NO
ChromaDB
   ↓
Insufficient Evidence
   ↓
Web Search

This is the core fallback mechanism.

🌐 9. WEB SEARCH FALLBACK

The system uses:

Tavily

when the local knowledge base does not contain sufficient information.

The search is designed to find diagnostic-oriented information including:

Common problems
Owner complaints
Reported issues
Faults
Warning signs
Troubleshooting
Repairs
Service issues
Forums

The search results contain information such as:

Title
URL
Content
🧹 10. WEB CONTENT PROCESSING

The retrieved web information goes through:

Web Result
    ↓
Extract Content
    ↓
Clean Text
    ↓
Create Documents
    ↓
Chunk Text

The goal is to transform raw external information into the same type of structured knowledge used by the RAG system.

💾 11. PERSISTENT WEB KNOWLEDGE

This is one of the major improvements made to the project.

Earlier approach
Search Web
    ↓
Get Result
    ↓
Give Result to LLM
    ↓
Answer

The web information was mainly useful only for that request.

Improved approach
Search Web
    ↓
Collect Content
    ↓
Clean
    ↓
Chunk
    ↓
Generate Embeddings
    ↓
Store in ChromaDB
    ↓
Use as Evidence
    ↓
Generate Answer

This means newly discovered information is persisted inside the knowledge base.

Future queries can therefore retrieve the stored knowledge.

🧠 IMPORTANT: THIS IS NOT MODEL TRAINING

The LLM itself is not retrained.

We are not doing:

Web Data
   ↓
Fine-tune LLM

Instead we are doing:

Web Data
   ↓
Clean
   ↓
Chunk
   ↓
Embed
   ↓
Store in ChromaDB
   ↓
Retrieve Later

So:

LLM = Reasoning + Generation

ChromaDB = Persistent Knowledge

Tavily = External Web Retrieval

Embedding Model = Semantic Representation
🔗 12. BUILDING THE CONTEXT

Once useful chunks have been retrieved, the system creates the LLM context.

Conceptually:

User Question
      +
Relevant Evidence
      ↓
Diagnostic Context

Example:

Vehicle:
Maruti Baleno 2024

Complaint:
Engine warning light is ON

DTC:
P3339

Retrieved Evidence:
- Relevant automotive information
- Technical information
- Retrieved source content

This context is given to the LLM.

🤖 13. LOCAL LLM GENERATION

The project uses:

Ollama

with:

phi3:mini

The local LLM is responsible for:

Understanding the user's question
Interpreting retrieved evidence
Reasoning over the evidence
Producing the final response

The LLM is therefore the generation and reasoning layer.

🧠 RAG EXPLAINED SIMPLY

RAG means:

R = Retrieve
A = Augment
G = Generate
Retrieve

Find relevant automotive information.

Augment

Put that information together with the user's question.

Generate

Ask the LLM to produce the final answer.

So:

Question
   ↓
Retrieve Evidence
   ↓
Add Evidence to Context
   ↓
LLM
   ↓
Answer
🔥 WHY RAG?

Instead of:

User → LLM → Answer

we use:

User
 ↓
Knowledge Retrieval
 ↓
Relevant Evidence
 ↓
LLM
 ↓
Answer

This is useful because automotive information can be:

Vehicle-specific
Model-year specific
DTC-specific
Stored in external documents
Updated over time
🧩 COMPONENT RESPONSIBILITIES
app.py

The Streamlit application.

Responsible for:

User Interface
User Input
Triggering Diagnosis
Displaying Results
backend/diagnostic.py

The main orchestration layer.

Responsible for:

Query creation
Local retrieval
Vehicle filtering
Web fallback
Web knowledge persistence
Context construction
LLM generation

This is the main controller of the diagnostic workflow.

backend/rag.py

Responsible for the knowledge retrieval layer.

Handles:

ChromaDB
Document ingestion
Text ingestion
External web chunk ingestion
Embeddings
Similarity search
Persistent storage
Context building
backend/embeddings.py

Responsible for:

Text
 ↓
Embedding Model
 ↓
Vector

Current embedding model:

embeddinggemma
backend/web_search.py

Responsible for:

Tavily Search
Query Normalization
Web Result Collection
Content Processing
Chunk Creation
backend/documents.py

Responsible for document-related processing such as:

Text Cleaning
Chunking
Document Processing
backend/llm.py

Responsible for connecting the application to:

Ollama

and generating responses using the local LLM.

backend/prompts.py

Contains the prompts used to guide the LLM's diagnostic response.

backend/config.py

Central configuration for:

Ollama
LLM Model
Embedding Model
ChromaDB Path
Chunk Configuration
Retrieval Configuration
Web Search Configuration
📁 PROJECT STRUCTURE
autodiagnostic-ai/
│
├── app.py
│
├── backend/
│   ├── config.py
│   ├── diagnostic.py
│   ├── documents.py
│   ├── embeddings.py
│   ├── ingest_documents.py
│   ├── llm.py
│   ├── prompts.py
│   ├── rag.py
│   └── web_search.py
│
├── data/
│   ├── manuals/
│   ├── chroma/
│   └── requirements.txt
│
├── scripts/
│   └── ingest_documents.py
│
├── requirements.txt
│
└── README.md
🔄 COMPLETE REQUEST EXAMPLE

Suppose the user enters:

Vehicle:
Maruti Baleno 2024

Complaint:
Engine warning light is ON

DTC:
P3339

The system executes:

1. Streamlit receives the input
          ↓
2. DiagnosticService creates the query
          ↓
3. ChromaDB is searched
          ↓
4. Relevant results are retrieved
          ↓
5. Vehicle relevance is checked
          ↓
6. If useful → use local evidence
          ↓
7. If insufficient → Tavily searches the web
          ↓
8. Web content is collected
          ↓
9. Web text is cleaned
          ↓
10. Content is chunked
          ↓
11. Chunks are embedded
          ↓
12. New knowledge is stored in ChromaDB
          ↓
13. Relevant evidence is assembled
          ↓
14. Context is sent to Ollama
          ↓
15. phi3:mini generates the response
          ↓
16. Streamlit displays the result
🚀 KEY IMPROVEMENTS
Initial Version
Local Documents
      ↓
Chunk
      ↓
Embed
      ↓
ChromaDB
      ↓
Retrieve
      ↓
LLM

This worked when the required information already existed locally.

Improvement 1 — Web Fallback
Local Knowledge
      ↓
Not Enough?
      ↓
Web Search

This allows the system to handle vehicles and problems not already present in the local documents.

Improvement 2 — Better Diagnostic Search

The search logic was improved to focus on:

Problems
Complaints
Faults
Troubleshooting
Repairs
Service Issues
Forums

instead of primarily searching vehicle specifications.

Improvement 3 — Vehicle Relevance

The system checks vehicle relevance in addition to semantic similarity.

Improvement 4 — Persistent Web Knowledge

The biggest knowledge architecture improvement:

Web Search
    ↓
Clean
    ↓
Chunk
    ↓
Embed
    ↓
Store Permanently
    ↓
Reuse Later

This makes the knowledge base capable of growing over time.

📊 BEFORE VS AFTER
Before
                 ┌───────────────┐
                 │ Local Manuals │
                 └───────┬───────┘
                         ↓
                    ChromaDB
                         ↓
                       LLM
                         ↓
                      Answer

Problem:

New Vehicle
    ↓
No Local Knowledge
    ↓
Limited Response
After
                 ┌───────────────┐
                 │ Local Manuals │
                 └───────┬───────┘
                         ↓
                    ChromaDB
                         │
                         │
User Query ──────────────┤
                         │
                         ▼
                 Enough Evidence?
                    /         \
                  YES          NO
                   │            │
                   │            ▼
                   │       Tavily Web
                   │            ↓
                   │        Clean/Chunk
                   │            ↓
                   │        Embeddings
                   │            ↓
                   │        ChromaDB
                   │            │
                   └─────┬──────┘
                         ↓
                  Relevant Context
                         ↓
                    Ollama / LLM
                         ↓
                      Answer
🛠️ TECHNOLOGY STACK
Technology	Role
Python	Core backend
Streamlit	User interface
RAG	Retrieval + generation architecture
ChromaDB	Persistent vector database
Ollama	Local AI runtime
phi3:mini	Local LLM
embeddinggemma	Embedding model
Tavily	Web search
Vector Search	Semantic retrieval
⚙️ SETUP
Clone
git clone https://github.com/adityavarmaaa/autodiagnostic-ai.git
cd autodiagnostic-ai
Create virtual environment
Windows
python -m venv .venv
.venv\Scripts\Activate.ps1
Install dependencies
pip install -r requirements.txt
Install Ollama Models
ollama pull phi3:mini
ollama pull embeddinggemma

Check:

ollama list
🔐 ENVIRONMENT VARIABLES

Create a local .env file:

OLLAMA_HOST=http://127.0.0.1:11434
LLM_MODEL=phi3:mini
EMBEDDING_MODEL=embeddinggemma
TAVILY_API_KEY=your_tavily_api_key

Do not commit .env or real API keys to GitHub.

Recommended .gitignore:

.env
.venv/
__pycache__/
*.pyc
data/chroma/
▶️ RUN

Start the application:

streamlit run app.py
📚 ADDING KNOWLEDGE

Local automotive documents can be placed inside:

data/manuals/

The ingestion pipeline processes them into:

Document
 ↓
Text
 ↓
Chunks
 ↓
Embeddings
 ↓
ChromaDB

Web-discovered knowledge follows a similar path:

Web Result
 ↓
Text
 ↓
Chunks
 ↓
Embeddings
 ↓
ChromaDB

Both become searchable through the same knowledge layer.

🔬 IMPORTANT ARCHITECTURAL PRINCIPLE

The system separates three major responsibilities:

┌─────────────────────────────┐
│       KNOWLEDGE             │
│                             │
│ ChromaDB + Web + Documents  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│        RETRIEVAL            │
│                             │
│ Embeddings + Vector Search  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│        GENERATION           │
│                             │
│      Ollama / phi3:mini     │
└─────────────────────────────┘

This separation allows the knowledge layer to grow without retraining the LLM every time new automotive information is discovered.

🧪 CURRENT LIMITATION

One current limitation is web-content extraction.

Some search results may provide only short snippets instead of complete article content.

The architecture already supports:

Web
 ↓
Chunk
 ↓
Embed
 ↓
Persist

The next quality improvement is to fetch and extract more complete content from the original source when the search provider returns only a small snippet.

🚀 FUTURE ROADMAP

Potential improvements include:

1. Better full-page web extraction
2. Better DTC-specific retrieval
3. Hybrid keyword + vector search
4. Duplicate web knowledge detection
5. Better source ranking
6. Better citations
7. Diagnostic confidence scoring
8. Evaluation datasets
9. Structured diagnostic reasoning
10. Production monitoring and observability
⚠️ DISCLAIMER

AutoDiagnostic AI is an experimental AI-assisted diagnostic system.

Its output should not replace:

Professional automotive technicians
Official manufacturer documentation
Vehicle-specific service procedures
Professional diagnostic equipment

AI-generated information should be independently verified before performing repairs.

👨‍💻 PROJECT SUMMARY

AutoDiagnostic AI evolved from a simple document-based RAG system into a more dynamic diagnostic knowledge pipeline.

The progression was:

                  START
                    ↓
          Local Automotive Documents
                    ↓
               RAG Pipeline
                    ↓
             ChromaDB Search
                    ↓
              Local LLM
                    ↓
              Basic Diagnosis
                    ↓
          ┌───────────────────┐
          │    IMPROVEMENT    │
          └─────────┬─────────┘
                    ↓
             Web Search Fallback
                    ↓
          Diagnostic-Oriented Search
                    ↓
           Vehicle Relevance Filter
                    ↓
            Web Knowledge Chunking
                    ↓
               Embeddings
                    ↓
          Persistent ChromaDB Storage
                    ↓
             Reusable Knowledge
                    ↓
          Local LLM Grounded Response
In one sentence:

AutoDiagnostic AI is a retrieval-first automotive diagnostic system that searches persistent automotive knowledge, falls back to the web when needed, stores newly discovered knowledge in ChromaDB, and uses a local LLM to generate a response from the retrieved evidence.

👨‍💻 Built By

Aditya Varma

AI / Product Builder

Focus Areas:

Artificial Intelligence
RAG
Vector Databases
Local LLMs
Automotive AI
Knowledge Retrieval
AI Product Engineering
