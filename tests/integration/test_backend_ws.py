import os
import json
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
import concurrent.futures
from dataclasses import dataclass

from os_ai_backend.app import create_app
from os_ai_llm.interfaces import LLMClient
from os_ai_llm.types import LLMResponse, Message, TextPart, ToolCall, Usage
from os_ai_core.tools.registry import ToolRegistry


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("OS_AI_BACKEND_TOKEN", "secret")

    class DummyLLM(LLMClient):  # type: ignore[abstract-method]
        def generate(self, *, messages, tools, system, **kwargs):  # type: ignore[override]
            return LLMResponse(messages=[Message(role="assistant", content=[TextPart(text="ok")])], tool_calls=[], usage=Usage())

        def format_tool_result(self, result):  # type: ignore[override]
            return Message(role="user", content=[TextPart(text="tool_result")])

    def fake_container(_provider=None, **kw):
        class _Inj:
            def get(self, cls):
                if cls.__name__ == "LLMClient":
                    return DummyLLM()
                if cls.__name__ == "ToolRegistry":
                    return ToolRegistry()
                raise KeyError(cls)
        return _Inj()

    # Patch backend DI wrapper to avoid real provider calls (no import of os_ai_core.di)
    import os_ai_backend.ws as backend_ws
    monkeypatch.setattr(backend_ws, "_create_container", fake_container)

    app = create_app()
    return TestClient(app)


def _recv_json(ws, timeout: float = 5.0):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(ws.receive_text)
        raw = fut.result(timeout=timeout)
    return json.loads(raw)


def test_ws_auth_required(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws"):
            pass


def test_ws_flow_session_and_run(client):
    with client.websocket_connect("/ws?token=secret") as ws:
        ws.send_text(json.dumps({"jsonrpc": "2.0", "id": "1", "method": "session.create", "params": {"provider": "anthropic"}}))
        resp = _recv_json(ws)
        assert resp["id"] == "1"
        assert resp["result"]["sessionId"]

        ws.send_text(json.dumps({"jsonrpc": "2.0", "id": "2", "method": "agent.run", "params": {"task": "echo", "maxIterations": 1}}))
        r2 = _recv_json(ws)
        assert r2["id"] == "2"
        assert r2["result"]["jobId"]

        # We expect some event notifications, eventually final
        final = None
        for _ in range(50):
            ev = _recv_json(ws)
            if ev.get("method") == "event.final":
                final = ev
                break
        assert final is not None
        assert final["params"]["status"] in ("ok", "fail")


def test_ws_mcp_approval_denied_returns_tool_error_and_final(monkeypatch):
    from os_ai_core.adapters.tools.composite_tool_gateway import CompositeToolGateway
    from os_ai_core.domain.tools.policies import ToolTrustLevel
    from os_ai_mcp.client.tool_provider import McpToolProvider

    monkeypatch.setenv("OS_AI_BACKEND_TOKEN", "secret")

    @dataclass
    class FakeMcpTool:
        name: str
        description: str
        inputSchema: dict

    class FakeMcpSession:
        def __init__(self) -> None:
            self.calls = []

        def initialize(self) -> None:
            return

        def list_tools(self):
            return [FakeMcpTool("write_file", "Write file", {"type": "object", "properties": {}})]

        def call_tool(self, name, arguments, timeout_seconds, cancel_token):
            self.calls.append((name, arguments))
            return None

        def close(self) -> None:
            return

    class DeniedMcpLLM(LLMClient):  # type: ignore[abstract-method]
        def __init__(self) -> None:
            self.calls = 0

        def generate(  # type: ignore[override]
            self,
            messages,
            tools,
            system=None,
            tool_choice="auto",
            max_tokens=1024,
            allow_parallel_tools=True,
            provider_context=None,
        ):
            self.calls += 1
            if self.calls == 1:
                tool = next(item for item in tools if item.name.startswith("mcp__local__write_file"))
                return LLMResponse(
                    messages=[Message(role="assistant", content=[TextPart(text="trying write")])],
                    tool_calls=[ToolCall(id="mcp-1", name=tool.name, args={"path": "x.txt", "text": "hello"})],
                    usage=Usage(input_tokens=1, output_tokens=1),
                )
            return LLMResponse(
                messages=[Message(role="assistant", content=[TextPart(text="handled denial")])],
                tool_calls=[],
                usage=Usage(input_tokens=1, output_tokens=1),
            )

        def format_tool_result(self, result):  # type: ignore[override]
            return Message(role="user", content=[TextPart(text=result.content[0].text)])

    session = FakeMcpSession()
    llm = DeniedMcpLLM()
    gateway = CompositeToolGateway([McpToolProvider("local", session, trust=ToolTrustLevel.LOCAL_UNTRUSTED)])

    def fake_container(_provider=None, **kw):
        class _Inj:
            def get(self, cls):
                if cls.__name__ == "LLMClient":
                    return llm
                if cls.__name__ == "ToolRegistry":
                    return ToolRegistry()
                if cls.__name__ == "CompositeToolGateway":
                    return gateway
                raise KeyError(cls)
        return _Inj()

    import os_ai_backend.ws as backend_ws
    monkeypatch.setattr(backend_ws, "_create_container", fake_container)

    app = create_app()
    with TestClient(app).websocket_connect("/ws?token=secret") as ws:
        ws.send_text(json.dumps({"jsonrpc": "2.0", "id": "1", "method": "agent.run", "params": {"task": "write", "maxIterations": 2}}))
        accepted = _recv_json(ws)
        job_id = accepted["result"]["jobId"]

        final = None
        tool_error = None
        approval_seen = False
        for _ in range(80):
            event = _recv_json(ws)
            if event.get("method") == "event.approval":
                approval_seen = True
                params = event["params"]
                ws.send_text(json.dumps({
                    "jsonrpc": "2.0",
                    "id": "approve-deny",
                    "method": "approval.respond",
                    "params": {
                        "jobId": params["jobId"],
                        "approvalId": params["approvalId"],
                        "approved": "false",
                    },
                }))
            if event.get("method") == "event.action" and event["params"].get("name") == "tool_result":
                tool_error = event
            if event.get("method") == "event.final":
                final = event
                break

    assert session.calls == []
    assert llm.calls == 2
    assert approval_seen is True
    assert tool_error is not None
    assert "approval result: denied" in tool_error["params"]["meta"]["text"]
    assert final is not None
    assert final["params"]["jobId"] == job_id
    assert final["params"]["status"] == "ok"
    assert "handled denial" in final["params"]["text"]


def test_rest_auth_and_files(client):
    # missing token
    r = client.post("/v1/files", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 401

    r = client.post("/v1/files", headers={"Authorization": "Bearer secret"}, files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 200
    file_id = r.json()["fileId"]

    r2 = client.get(f"/v1/files/{file_id}", headers={"Authorization": "Bearer secret"})
    assert r2.status_code == 200
    assert r2.content == b"hello"
