import asyncio
import structlog
from app.services.rag_service import RAGService
from app.models.schemas import ChatRequest

# Configure logging to be less verbose for testing
structlog.configure(
    processors=[
        structlog.processors.JSONRenderer()
    ]
)

async def test_drafting_flow():
    rag_service = RAGService()
    
    print("\n--- Testing Proactive Prompting (Missing Info) ---")
    query1 = "أريد كتابة عقد إيجار لشقة في بيروت"
    request1 = ChatRequest(query=query1, history=[])
    response1 = await rag_service.process_query(request1)
    print(f"Query: {query1}")
    print(f"Response:\n{response1.response}")
    
    print("\n--- Testing Drafting (Provided Info) ---")
    query2 = "المؤجر هو سمير، والمستأجر هو رامي. الشقة في الأشرفية، الإيجار 500 دولار شهرياً لمدة سنة."
    # Simulate history from previous turn
    history2 = [
        {"role": "user", "content": query1},
        {"role": "assistant", "content": response1.response}
    ]
    request2 = ChatRequest(query=query2, history=history2)
    response2 = await rag_service.process_query(request2)
    print(f"Query: {query2}")
    print(f"Response:\n{response2.response}")

    print("\n--- Testing Normal RAG Query (No Drafting) ---")
    query3 = "ما هي عقوبة الشيك بدون رصيد في لبنان؟"
    request3 = ChatRequest(query=query3, history=[])
    response3 = await rag_service.process_query(request3)
    print(f"Query: {query3}")
    print(f"Response:\n{response3.response}")

if __name__ == "__main__":
    asyncio.run(test_drafting_flow())
