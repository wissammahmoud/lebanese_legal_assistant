from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
import os

# Set dummy env vars for testing if not set
os.environ["SERVICE_API_KEY"] = "test-key"
os.environ["OPENAI_API_KEY"] = "sk-test"

# Make sure settings are reloaded/initialized or just patch them. 
# Since pydantic settings read env at import, we might need reload or patch.
# For simplicity in this script, we assume .env might not exist or we mock settings.

client = TestClient(app)

def run_tests():
    print("Running Tests...")
    
    # Test Health
    try:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        print("✅ Health Check Passed")
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")

    # Test Auth Failure
    try:
        response = client.post(f"{settings.API_V1_STR}/chat", json={"query": "hi"})
        assert response.status_code == 403
        print("✅ Auth Check Passed (403 expected)")
    except Exception as e:
        print(f"❌ Auth Check Failed: {e}")

    print("Tests Completed.")

if __name__ == "__main__":
    run_tests()
