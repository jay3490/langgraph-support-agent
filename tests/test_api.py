from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from support_agent.api import create_app
from support_agent.config import Settings
from support_agent.demo_policy import DemoPolicy
from support_agent.llm import FakeChatModel
from tests.conftest import TODAY


def parse_sse(body: str) -> list[tuple[str, Any]]:
    events: list[tuple[str, Any]] = []
    for block in body.replace("\r\n", "\n").strip().split("\n\n"):
        name, data = "message", []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data.append(line.removeprefix("data:").strip())
        if data:
            events.append((name, json.loads("\n".join(data))))
    return events


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings, model=FakeChatModel(responder=DemoPolicy()), today=TODAY)
    with TestClient(app) as test_client:
        yield test_client


def new_thread(client: TestClient, customer_id: str | None = "CUST-1001") -> str:
    response = client.post("/threads", json={"customer_id": customer_id})
    assert response.status_code == 201
    return response.json()["thread_id"]


def send(client: TestClient, thread_id: str, content: str) -> list[tuple[str, Any]]:
    response = client.post(f"/threads/{thread_id}/messages", json={"content": content})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


def test_health(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"


def test_create_thread(client: TestClient) -> None:
    response = client.post("/threads", json={"customer_id": "CUST-1002"})
    assert response.status_code == 201
    assert response.json()["customer_id"] == "CUST-1002"
    anonymous = client.post("/threads")
    assert anonymous.status_code == 201
    assert anonymous.json()["customer_id"] is None


def test_create_thread_validation(client: TestClient) -> None:
    assert client.post("/threads", json={"customer_id": "CUST-9999"}).status_code == 404
    assert client.post("/threads", json={"customer_id": "alice"}).status_code == 422


def test_message_streams_node_updates_tokens_and_done(client: TestClient) -> None:
    thread = new_thread(client)
    events = send(client, thread, "Where is ORD-10003?")
    kinds = [name for name, _ in events]
    assert kinds[0] == "node"
    assert "token" in kinds
    assert kinds[-1] == "done"
    nodes = [data["node"] for name, data in events if name == "node"]
    assert nodes == [
        "router",
        "order_status",
        "supervisor",
        "tools",
        "order_status",
        "supervisor",
        "respond",
    ]
    tokens = "".join(data["content"] for name, data in events if name == "token")
    done = events[-1][1]
    assert done["status"] == "completed"
    assert done["reply"] == tokens
    assert "1Z999AA10000000003" in done["reply"]


def test_get_thread_state(client: TestClient) -> None:
    thread = new_thread(client)
    send(client, thread, "Where is ORD-10003?")
    state = client.get(f"/threads/{thread}").json()
    assert state["customer_id"] == "CUST-1001"
    assert state["route"] == "order_status"
    assert state["pending_approval"] is None
    types = [m["type"] for m in state["messages"]]
    assert types == ["human", "ai", "tool", "ai"]
    assert state["messages"][1]["tool_calls"][0]["name"] == "lookup_order"
    assert state["final_response"] == state["messages"][-1]["content"]


def test_unknown_thread_is_404(client: TestClient) -> None:
    assert client.get("/threads/nope").status_code == 404
    response = client.post("/threads/nope/messages", json={"content": "hi"})
    assert response.status_code == 404
    assert client.post("/threads/nope/approve", json={"reviewer": "x"}).status_code == 404


def test_message_validation(client: TestClient) -> None:
    thread = new_thread(client)
    assert client.post(f"/threads/{thread}/messages", json={"content": ""}).status_code == 422
    assert client.post(f"/threads/{thread}/messages", json={}).status_code == 422


def test_refund_approval_flow(client: TestClient) -> None:
    thread = new_thread(client)
    events = send(client, thread, "Please refund ORD-10002, the desk wobbles")
    [(_, request)] = [(n, d) for n, d in events if n == "approval_required"]
    assert request["refunds"][0]["order_id"] == "ORD-10002"
    assert events[-1][1]["status"] == "awaiting_approval"

    state = client.get(f"/threads/{thread}").json()
    assert state["pending_approval"]["refunds"][0]["amount"] == "349.00"

    blocked = client.post(f"/threads/{thread}/messages", json={"content": "hello?"})
    assert blocked.status_code == 409

    approved = client.post(f"/threads/{thread}/approve", json={"reviewer": "sam", "note": "ok"})
    assert approved.status_code == 200
    result = approved.json()
    assert result["status"] == "completed"
    assert "refunded 349.00 USD" in result["reply"]

    again = client.post(f"/threads/{thread}/approve", json={"reviewer": "sam"})
    assert again.status_code == 409


def test_refund_rejection(client: TestClient) -> None:
    thread = new_thread(client)
    send(client, thread, "Refund ORD-10002")
    rejected = client.post(
        f"/threads/{thread}/reject", json={"reviewer": "sam", "note": "Need photos"}
    )
    assert rejected.status_code == 200
    assert "Need photos" in rejected.json()["reply"]
    assert client.post(f"/threads/{thread}/reject", json={"reviewer": ""}).status_code == 422


def test_thread_survives_app_restart(settings: Settings) -> None:
    def app() -> TestClient:
        return TestClient(
            create_app(settings, model=FakeChatModel(responder=DemoPolicy()), today=TODAY)
        )

    with app() as first:
        thread = new_thread(first)
        send(first, thread, "Refund ORD-10002")
    with app() as second:
        assert second.get(f"/threads/{thread}").json()["pending_approval"] is not None
        result = second.post(f"/threads/{thread}/approve", json={"reviewer": "sam"}).json()
        assert result["status"] == "completed"
