from fastapi.testclient import TestClient

import app


def test_health_endpoint():
    client = TestClient(app.app)
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_ready_endpoint_is_graceful_without_model():
    client = TestClient(app.app)
    response = client.get('/ready')
    assert response.status_code == 200
    body = response.json()
    assert 'model_present' in body
    assert 'feedback_db' in body
