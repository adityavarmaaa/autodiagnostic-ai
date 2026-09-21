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
