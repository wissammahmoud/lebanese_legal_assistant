import sys
import os
import asyncio
import structlog
import logging
import io

# Force UTF-8 output so Arabic / French text prints correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import connections, Collection
from app.core.config import settings
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
log = structlog.get_logger()

# ──────────────────────────────────────────────
# Sample scenarios for rewriter testing
SAMPLE_SCENARIOS = [
    {
        "desc": "Appeals court – judicial recusal / nullity (Arabic)",
        "query": (
            "أصدرت محكمة الاستئناف حكماً في قضية تجارية كبرى، وكانت الهيئة التي نظرت في الدعوى مؤلفة من ثلاثة قضاة. "
            "بعد صدور الحكم، اكتشف الطرف الخاسر في الدعوى أن رئيس الهيئة هو في الحقيقة ابن عم أحد القضاة الأعضاء في "
            "نفس الهيئة التي أصدرت القرار. ما هو المصير القانوني لهذا الحكم؟"
        ),
    },
    {
        "desc": "Possession – less than 1 year / استرداد حيازة (Arabic)",
        "query": (
            "وضع سمير يده على قطعة أرض وزرعها لمدة ٤ أشهر فقط. قام أحد الجيران فجأة بتسييج الأرض ومنع سمير من الدخول إليها. "
            "عندما أراد سمير رفع دعوى «استرداد حيازة»، أخبره المحامي أن القانون عادة يشترط مرور سنة كاملة على الحيازة "
            "لرفع دعاوى الحماية. بناءً على نص المادة أعلاه، هل يحق لسمير رفع هذه الدعوى تحديداً رغم أنه لم يمر على "
            "حيازته سوى ٤ أشهر؟ ولماذا؟"
        ),
    },
    {
        "desc": "Legal title vs physical possession / استرداد حيازة (Arabic)",
        "query": (
            "يملك «منير» سند تمليك (أوراق رسمية) لشقة في بيروت، لكنه لم يزرها أبداً ولم يضع فيها أي أثاث. "
            "اكتشف لاحقاً أن شخصاً غريباً قد كسر القفل وسكن فيها. قرر منير رفع دعوى استرداد حيازة مستنداً إلى أنه "
            "صاحب الحق القانوني. هل تتوفر في وضع «منير» شروط هذه الدعوى؟"
        ),
    },
    {
        "desc": "Arbitration clause – enforceability & court jurisdiction (English)",
        "query": (
            "A Lebanese company signed a commercial distribution contract that included an arbitration clause specifying "
            "that all disputes must be resolved by the ICC in Paris. A dispute arose and the Lebanese party filed a lawsuit "
            "directly before the Lebanese commercial court, ignoring the arbitration clause. Can the Lebanese court accept "
            "jurisdiction, or is it obliged to refer the parties to arbitration? What happens to the lawsuit?"
        ),
    },
    {
        "desc": "Employer non-compete clause – validity under Lebanese labor law (English)",
        "query": (
            "An employee signed a non-compete agreement when joining a Beirut-based tech firm, prohibiting him from "
            "working in the same industry for 3 years anywhere in Lebanon after leaving the company. He resigned and "
            "immediately joined a competitor. The original employer is seeking an injunction and damages. "
            "Under Lebanese labor law, is such a non-compete clause enforceable, and what conditions must be met?"
        ),
    },
    {
        "desc": "Criminal liability – employer workplace accident (English)",
        "query": (
            "A construction worker fell from scaffolding on a Beirut building site and died. Investigators found that "
            "the employer had not provided any safety equipment and had ignored repeated written complaints from workers "
            "about dangerous conditions. The victim's family wants to pursue both criminal and civil claims. "
            "What criminal charges can be brought against the employer under Lebanese law, and who bears civil liability?"
        ),
    },
]



# ──────────────────────────────────────────────
# Synchronous Milvus search (used by both modes)
# ──────────────────────────────────────────────
def search(query: str, top_k: int = 5):
    """Embed *query* and return top-k Milvus results."""
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    query_vector = client.embeddings.create(
        input=[query], model="text-embedding-3-small"
    ).data[0].embedding

    connections.connect(uri=settings.MILVUS_URI)
    collection = Collection(settings.MILVUS_COLLECTION_NAME)
    collection.load()

    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": 200}},
        limit=top_k,
        output_fields=["text_content", "source_type", "metadata"],
    )
    return results


# ──────────────────────────────────────────────
# Rewriter test harness
# ──────────────────────────────────────────────
async def _rewrite_all(queries: list[str]) -> list[str]:
    """Rewrite all queries in a single event loop run to avoid lifecycle noise."""
    from app.services.query_rewriter_service import QueryRewriterService
    rewriter = QueryRewriterService()
    results = []
    for q in queries:
        results.append(await rewriter.rewrite(q))
    return results


def test_rewriter(run_search: bool = True):
    """Run all SAMPLE_SCENARIOS through the QueryRewriter and optionally search."""
    print("\n" + "=" * 70)
    print("   QUERY REWRITER TEST – Lebanese Legal Assistant")
    print("=" * 70)

    # Batch all rewrites in a single event loop run (avoids Windows asyncio noise)
    original_queries = [s["query"] for s in SAMPLE_SCENARIOS]
    rewritten_queries = asyncio.run(_rewrite_all(original_queries))

    for i, (scenario, rewritten) in enumerate(zip(SAMPLE_SCENARIOS, rewritten_queries), 1):
        print(f"\n[{i}] {scenario['desc']}")
        print(f"  ORIGINAL  : {scenario['query']}")
        print(f"  REWRITTEN : {rewritten}")

        if run_search:
            print("\n  ── Top Milvus results (rewritten query) ──")
            try:
                results = search(rewritten, top_k=5)
                for hits in results:
                    for hit in hits:
                        print(
                            f"    Score: {hit.score:.4f} | "
                            f"Source: {hit.entity.get('source_type')} | "
                            f"{hit.entity.get('text_content')[:120]}..."
                        )
            except Exception as e:
                print(f"    [Search error: {e}]")

        print("-" * 70)


# ──────────────────────────────────────────────
# Basic search (original behaviour, kept intact)
# ──────────────────────────────────────────────
def search_and_print(query: str):
    log.info("Embedding query...", query=query)
    results = search(query, top_k=20)
    for hits in results:
        log.info(f"Results for query: {query}")
        for hit in hits:
            print(f"ID: {hit.id}, Score: {hit.score:.4f}, Source: {hit.entity.get('source_type')}")
            print(f"Content: {hit.entity.get('text_content')[:200]}...")
            print(f"Metadata: {hit.entity.get('metadata')}")
            print("-" * 40)


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────
if __name__ == "__main__":
    if "--test-rewriter" in sys.argv:
        # Run the rewriter test harness (with Milvus search unless --no-search)
        run_search = "--no-search" not in sys.argv
        test_rewriter(run_search=run_search)
    else:
        q = "What are the landlord's obligations?"
        if len(sys.argv) > 1:
            q = " ".join(sys.argv[1:])
        search_and_print(q)
