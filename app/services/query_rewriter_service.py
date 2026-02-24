import structlog
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings
from langsmith import traceable

log = structlog.get_logger()

_REWRITER_SYSTEM_PROMPT = """\
You are a Lebanese legal terminology expert. Your only task is to rewrite the user's question \
or scenario into formal Lebanese legal search terms, using vocabulary that would appear in \
Lebanese law articles, legislative decrees, or court rulings.

Rules:
- Output ONLY the rewritten query. No explanations, no preamble, no labels.
- Preserve the language of the query (Arabic stays Arabic, French stays French, English stays English). \
  You may naturally blend formal Arabic, French, or English legal terms the way Lebanese courts do.
- Replace colloquial phrases with their formal legal equivalents.
- If a legal domain is identifiable, add the relevant Lebanese law reference context, for example: \
  "قانون الموجبات والعقود", "مرسوم اشتراعي 17386 - قانون العمل", "قانون الأحوال الشخصية", \
  "قانون العقوبات اللبناني", "قانون التجارة", "قانون أصول المحاكمات المدنية", \
  "Code of Obligations and Contracts", "Labor Law Decree 17386", "Personal Status Law", \
  "Penal Code", "Commercial Code", "Code of Civil Procedure".
- Do NOT invent article numbers or rulings. Only rephrase toward known Lebanese legal vocabulary.
- Keep the output concise (1–3 sentences maximum).\
"""


class QueryRewriterService:
    """
    Pre-retrieval agent that rewrites a user's natural-language scenario into
    formal Lebanese legal terminology so that the subsequent vector search
    retrieves more relevant law articles and court rulings.
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    @traceable(run_type="llm", name="Rewrite Query")
    async def rewrite(self, query: str) -> str:
        """
        Rewrite *query* into formal Lebanese legal search terms.
        Falls back to the original query on any failure so the pipeline is never blocked.
        """
        try:
            rewritten = await self._call_llm(query)
            # Safety: if the model returns something empty, fall back
            return rewritten.strip() or query
        except Exception as e:
            log.warning(
                "QueryRewriter failed, falling back to original query",
                error=str(e),
                query=query,
            )
            return query

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    async def _call_llm(self, query: str) -> str:
        log.info("QueryRewriter: rewriting query", query=query)
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _REWRITER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,  # deterministic – we want consistent legal terminology
            max_tokens=256,
        )
        return response.choices[0].message.content
