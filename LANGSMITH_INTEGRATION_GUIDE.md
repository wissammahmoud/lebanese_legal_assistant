# LangSmith & OpenEvals Integration Walkthrough

This document outlines the changes made to integrate **LangSmith** for full-trajectory tracing and **OpenEvals** for automated quality evaluation of the Lebanese Legal Assistant.

## 🚀 Overview of Changes

We have implemented a comprehensive observability and evaluation layer that tracks every stage of your RAG pipeline.

### 1. Full Trajectory Tracing (LangSmith)
Every request now generates a detailed "trace" in LangSmith. This allows you to visualize:
- **Input Query**: The original user question.
- **Rewritten Query**: How the `QueryRewriterService` transformed the question into legal terminology.
- **Embedding Generation**: The latency and input/output of the OpenAI embedding call.
- **Milvus Retrieval**: The exact list of law articles and rulings retrieved, including their **scores**, **metadata**, and **source types**.
- **Final LLM Generation**: The prompt sent to GPT-4o-mini and the final response generated.

**Files Modified:**
- `app/services/rag_service.py` (Root Trace)
- `app/services/query_rewriter_service.py`
- `app/services/embedding_service.py`
- `app/services/vector_store_service.py`
- `app/services/llm_service.py`

### 2. Automated Quality Evaluation (OpenEvals)
We integrated `OpenEvals` to perform "Online Evaluation". For every response generated, a second, more powerful LLM (GPT-4o) acts as a **Judge**.
- **Accuracy Check**: It compares the generated answer against the retrieved Lebanese legal context.
- **Feedback Loop**: It generates a score and reasoning, which is logged and can be viewed in LangSmith to identify problematic retrievals or hallucinations.

### 3. Best Practices Applied
- **Non-Blocking Evals**: Evaluation runs are wrapped in try-except blocks to ensure that if the judge fails, the user still gets their answer.
- **Semantic Trace Names**: Used descriptive names like "Milvus Vector Search" and "Rewrite Query" for easier debugging in the LangSmith UI.
- **Circuit Breaker Persistence**: Maintained the existing circuit breakers to ensure tracing doesn't interfere with system resilience.

## 🛠️ What YOU need to do

To make this work flawlessly, you must configure your environment variables:

1. **LangSmith API Key**:
   - Go to [LangSmith](https://smith.langchain.com/) and create an API key.
   - Add it to your `.env` file:
     ```env
     LANGCHAIN_TRACING_V2=true
     LANGCHAIN_API_KEY=ls__your_api_key_here
     LANGCHAIN_PROJECT=lebanese-legal-assistant
     ```

2. **OpenAI API Key**:
   - Ensure `OPENAI_API_KEY` is set, as the OpenEVALS judge uses `gpt-4o` for high-quality evaluation.

3. **Restart the Service**:
   - Restart your `uvicorn` server to pick up the new tracing decorators and environment variables.

## 📈 Monitoring your Pipeline

Once running, you can visit the [LangSmith Projects Page](https://smith.langchain.com/projects). Click on your project to see:
- **Latent Bottlenecks**: See which stage (Milvus vs. OpenAI) is taking the most time.
- **Retrieval Precision**: Inspect the "Milvus Vector Search" step to see if the retrieved laws were actually relevant.
- **Hallucination Detection**: Filter for low evaluation scores from the OpenEvals judge.

---
*Created by Antigravity AI*
