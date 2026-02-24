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
                raw_results = await self.vector_store.search(vector)
                for r in raw_results:
                    sources.append(SourceDocument(
                        id=r["id"],
                        score=r["score"],
                        text=r["text"],
                        source_type=r["source"],
                        metadata=r["metadata"]
                    ))
                context_chunks = [f"Source Type: {r['source']}\nText: {r['text']}" for r in raw_results]
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

        return messages, sources, context_text

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
1. يجب إعطاء الأولوية للنصوص القانونية والاجتهادات المتوفرة في سياق البحث (Context) على معلوماتك العامة.
2. استند دائماً إلى القانون اللبناني عند الضرورة.
3. قم دائماً بذكر القانون ذي الصلة، رقم المادة، أو الاجتهاد عند توفره.
4. إذا لم تكن الإجابة موجودة بوضوح في النصوص القانونية المقدمة، قل حرفياً:
"لا يوجد نص قانوني واضح في قاعدة البيانات المتاحة يجيب بشكل مباشر على هذا السؤال، لكن يمكن الرجوع إلى..."
5. لا تقم أبداً باختراع قوانين أو مواد أو اجتهادات قضائية.
6. لا تقدم استشارات قانونية نهائية؛ قدم معلومات قانونية فقط.
7. يجب أن تكون الإجابة دائماً باللغة العربية الفصحى والمهنية.

هيكلية الإجابة (في حال الاستفسار القانوني):
- الإجابة القانونية:
[شرح واضح ودقيق]

- النصوص القانونية ذات الصلة:
[قائمة بالمواد القانونية]

- الاجتهادات ذات الصلة (إن وجدت):
[قائمة بالاجتهادات القضائية]

هيكلية الإجابة (في حال طلب صياغة مستند):
1. إذا نقصت معلومات أساسية: اطلبها من المستخدم بشكل مهني وواضح في نقاط.
2. إذا توفرت المعلومات: قم بصياغة المستند (عقد، إنذار، بند) بأسلوب قانوني رصين ودقيق، مع تضمين كافة النصوص القانونية اللبنانية المرعية الإجراء.

- ملاحظة:
دائماً أذكر أن هذه معلومات عامة وليست استشارة قانونية. في حال كانت لديك قضية قائمة في لبنان، قد ترغب باستشارة محامٍ مختص في المجال.
حافظ على نبرة مهنية ورسمية.
"""
