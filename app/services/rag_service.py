import re
import structlog
import pybreaker
from typing import List, Optional
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.services.llm_service import LLMService
from app.services.query_rewriter_service import QueryRewriterService
from app.services.drafting_service import DraftingService
from app.models.schemas import ChatRequest, ChatResponse, SourceDocument
from langsmith import traceable
from openevals.llm import create_llm_as_judge
from langchain_openai import ChatOpenAI
from app.core.config import settings

log = structlog.get_logger()

_GREETING_RESPONSE = (
    "أهلاً وسهلاً! أنا ADL Legal Assistant، مساعدك القانوني المتخصص في القانون اللبناني. "
    "كيف يمكنني مساعدتك اليوم؟ يسعدني الإجابة على استفساراتك القانونية المتعلقة بالقوانين والتشريعات والاجتهادات اللبنانية."
)

_OFF_TOPIC_RESPONSE = (
    "أعتذر، أنا مساعد قانوني متخصص حصرياً في القانون اللبناني ولا أستطيع الإجابة على أسئلة خارج نطاق هذا التخصص. "
    "هل لديك استفسار قانوني يتعلق بالقوانين أو التشريعات أو الاجتهادات القضائية اللبنانية؟"
)

_INTENT_SYSTEM_PROMPT = """\
Classify the user's message into exactly one of these three categories:
- greeting: a social greeting, farewell, or small talk with no legal content (e.g. "hello", "how are you", "هو ار يو", "مرحبا", "شكراً")
- legal: any question, scenario, or request related to Lebanese law, legal procedures, court rulings, contracts, or legal rights
- off_topic: anything else that is not a legal question and not a greeting (e.g. weather, sports, cooking, jokes, technical questions unrelated to law)

Reply with ONLY one word: greeting, legal, or off_topic.\
"""

class RAGService:
    def __init__(self):
        self.query_rewriter = QueryRewriterService()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()
        self.llm_service = LLMService()
        self.drafting_service = DraftingService()
        
        # Initialize an LLM-as-a-judge for online evaluation
        self.judge = create_llm_as_judge(
            judge=ChatOpenAI(model="gpt-4o", api_key=settings.OPENAI_API_KEY),
            prompt="Is the answer legally accurate and helpful based on the context provided?"
        )

    @traceable(run_type="chain", name="RAG Pipeline")
    async def process_query(self, request: ChatRequest) -> ChatResponse:
        query = request.query
        log.info("Processing query", query=query)

        # 0. Intent gate
        intent = await self._classify_intent(query)
        log.info("Intent classified", intent=intent)
        if intent == "greeting":
            return ChatResponse(response=_GREETING_RESPONSE, sources=[])
        if intent == "off_topic":
            return ChatResponse(response=_OFF_TOPIC_RESPONSE, sources=[])

        # 1. Retrieve & Prepare
        messages, sources, context_text = await self._prepare_rag_context(request)

        # 4. Generate Response
        try:
            response_text = await self.llm_service.generate_response(messages)
            
            # 5. Online Evaluation (OpenEvals + LangSmith)
            if context_text:
                self._run_online_eval(query, context_text, response_text)

            return ChatResponse(response=response_text, sources=sources)
        except Exception as e:
            log.error("Failed to generate response", error=str(e))
            return ChatResponse(
                response="I apologize, but I am unable to process your request at this moment.", 
                error=str(e)
            )

    @traceable(run_type="chain", name="Streaming RAG Pipeline")
    async def stream_query(self, request: ChatRequest):
        query = request.query
        log.info("Streaming query", query=query)

        # 0. Intent gate — short-circuit before touching rewriter / embeddings / Milvus
        intent = await self._classify_intent(query)
        log.info("Intent classified", intent=intent)
        if intent == "greeting":
            yield {"type": "sources", "sources": []}
            yield {"type": "content", "content": _GREETING_RESPONSE}
            return
        if intent == "off_topic":
            yield {"type": "sources", "sources": []}
            yield {"type": "content", "content": _OFF_TOPIC_RESPONSE}
            return

        # 1. Retrieve & Prepare
        messages, sources, context_text = await self._prepare_rag_context(request)

        # 2. Yield Sources first (so UI can show them immediately)
        yield {"type": "sources", "sources": [s.dict() for s in sources]}

        # 3. Stream bits of response
        full_response = ""
        async for chunk in self.llm_service.stream_response(messages):
            full_response += chunk
            yield {"type": "content", "content": chunk}

        # 4. Background Evaluation
        if context_text and full_response:
             self._run_online_eval(query, context_text, full_response)

    async def _prepare_rag_context(self, request: ChatRequest):
        query = request.query
        history = request.history

        # 0. Rewrite query
        rewritten_query = await self.query_rewriter.rewrite(query)
        log.info("Query rewritten", original=query, rewritten=rewritten_query)

        # 1. Generate Embedding
        vector = None
        try:
            vector = await self.embedding_service.get_embedding(rewritten_query)
        except Exception as e:
            log.error("Embedding failed.", error=str(e))

        # 2. Retrieve Context
        context_text = ""
        sources = []
        retrieval_error = None

        if vector:
            try:
                # Build a metadata filter if the user asked about a specific article number
                expr = self._build_article_filter(query)
                raw_results = await self.vector_store.search(vector, expr=expr)

                # If the filter returned nothing, fall back to unfiltered vector search
                if expr and not raw_results:
                    log.info("Filtered search returned no results, falling back to vector-only search")
                    raw_results = await self.vector_store.search(vector)

                log.info("Retrieved documents from Milvus", count=len(raw_results))
                for r in raw_results:
                    log.info("Document retrieved", id=r["id"], score=r["score"], source=r["source"])
                    sources.append(SourceDocument(
                        id=r["id"],
                        score=r["score"],
                        text=r["text"],
                        source_type=r["source"],
                        metadata=r["metadata"]
                    ))
                context_chunks = [f"Source: {r['source']}\nMetadata: {r['metadata']}\nText: {r['text']}" for r in raw_results]
                context_text = "\n\n".join(context_chunks)
            except pybreaker.CircuitBreakerError:
                retrieval_error = "Legal database is temporarily unavailable."
            except Exception as e:
                retrieval_error = "Failed to retrieve legal context."

        # 2.5 Identify Drafting Request
        drafting_context = ""
        template_id = self.drafting_service.identify_request(query)
        if template_id:
            template = self.drafting_service.get_template(template_id)
            if template:
                drafting_context = f"\n\n### DOCUMENT DRAFTING INSTRUCTION:\n" \
                                   f"User wants to draft: {template['name']}\n" \
                                   f"Required Fields: {', '.join(template['required_fields'])}\n" \
                                   f"Description: {template['description']}\n"

        # 3. Construct Messages
        system_instruction = self._get_system_prompt()
        context_block = ""
        if context_text:
            context_block = f"\n\n### LEBANESE LEGAL CONTEXT:\n{context_text}\n"
        elif retrieval_error:
            context_block = f"\n\n### SYSTEM NOTICE:\n{retrieval_error}. Proceeding with general knowledge.\n"
        
        messages = [{"role": "system", "content": system_instruction + context_block + drafting_context}]
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": query})

        # Log the full prompt for debugging
        log.info("Full Prompt Constructed", messages=messages)

        return messages, sources, context_text

    async def _classify_intent(self, query: str) -> str:
        """Returns 'greeting', 'legal', or 'off_topic'. Falls back to 'legal' on error."""
        try:
            resp = await self.llm_service.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0,
                max_tokens=5,
            )
            label = resp.choices[0].message.content.strip().lower()
            if label in ("greeting", "legal", "off_topic"):
                return label
            return "legal"
        except Exception as e:
            log.warning("Intent classification failed, defaulting to legal", error=str(e))
            return "legal"

    def _build_article_filter(self, query: str) -> str | None:
        """
        Detects if the user is asking about a specific article number and returns
        a Milvus filter expression to pin retrieval to that exact article.
        Handles Arabic-Indic numerals (٢٤) and Western numerals (24).
        """
        pattern = r'(?:مادة|المادة|article)\s*([٠-٩\d]+)'
        match = re.search(pattern, query, re.IGNORECASE)
        if not match:
            return None
        num_str = match.group(1).translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789'))
        article_num = int(num_str)
        expr = f'metadata["article_number"] == {article_num}'
        log.info("Article filter detected", article_number=article_num, expr=expr)
        return expr

    def _run_online_eval(self, query: str, context: str, answer: str):
        try:
            eval_result = self.judge(
                inputs={"query": query, "context": context},
                outputs={"answer": answer}
            )
            log.info("Online Evaluation Result", 
                     score=eval_result.get("score"), 
                     reasoning=eval_result.get("reasoning"))
        except Exception as e:
            log.warning("Online evaluation skipped due to error", error=str(e))

    def _get_system_prompt(self) -> str:
        return """أنت "ADL Legal Assistant"، مساعد ذكاء اصطناعي محترف متخصص حصرياً في القانون اللبناني.
دورك هو مساعدة المحامين وطلاب الحقوق والمهنيين القانونيين من خلال تقديم معلومات قانونية دقيقة بناءً على القوانين والاجتهادات اللبنانية.

القواعد الأساسية:
1. اعتمد بشكل أساسي على المعلومات المتوفرة في سياق البحث (Context) للإجابة على سؤال المستخدم.
2. إذا كان سياق البحث (Context) يحتوي على معلومات ذات صلة مباشرة أو غير مباشرة بسؤال المستخدم، فقم بتحليلها واستخلاص الإجابة منها بشكل مهني.
3. التزم بالدقة القانونية واذكر القانون ذي الصلة أو رقم المادة عند توفره في السياق أو من معلوماتك الداخلية.
4. في حال كان سياق البحث (Context) لا يحتوي على أي معلومات مفيدة إطلاقاً للإجابة على السؤال، قل حرفياً:
"لا يوجد نص قانوني واضح يجيب بشكل مباشر على هذا السؤال، لكن يمكن الرجوع إلى المبادئ العامة لـ..."
5. لا تقم أبداً باختراع نصوص قانونية غير موجودة.
6. إذا كان السؤال يتطلب استنتاجاً منطقياً من النصوص المتوفرة، فقم بذلك مع التوضيح.

هيكلية الإجابة (التزم بهذا الترتيب دائماً):

**الإجابة القانونية:**
[عرض الإجابة بناءً على السياق المتوفر]

---
**المصادر القانونية المعتمدة:**
لكل مصدر استخدمته من السياق، اذكر السطر التالي بالتفصيل:
- للمواد القانونية: اسم القانون | رقم المادة | التصنيف (مدني / جزائي / تجاري ...)
- للاجتهادات القضائية: رقم القرار | السنة | المحكمة | تاريخ الجلسة | الرئيس | الأعضاء

مثال للمواد:
📄 قانون أصول المحاكمات المدنية — المادة 24 — تصنيف: مدني

مثال للاجتهادات:
⚖️ قرار رقم 128 لعام 2021 — محكمة التمييز الجزائية — جلسة 15/09/2021 — الرئيس: سهير الحركة

داًئماً حافظ على نبرة مهنية ورسمية.
"""
