from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_ok():
    resp = client.get("/")
    assert resp.status_code == 200

def test_docs_ok():
    resp = client.get("/docs")
    assert resp.status_code == 200

def test_post_without_file_returns_422():
    # If POST requires a file without the file will be 422 error
    resp = client.post("/")
    assert resp.status_code in (400, 422)
