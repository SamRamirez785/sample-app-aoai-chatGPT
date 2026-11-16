"""Credential-free regression coverage for application-managed RAG."""
import json
import os
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

os.environ.setdefault("AZURE_OPENAI_MODEL", "test-deployment")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")

from backend import retrieval, utils


def settings(mode="simple", key="search-key"):
    source = NS(_type="azure_search", filter="active eq true", permitted_groups_column=None,
                query_type=mode, use_semantic_search=False, semantic_search_config="semantic-config",
                vector_columns=["vector"], content_columns=["body", "summary"], key=key,
                endpoint="https://example.search.windows.net", index="policies", top_k=3,
                max_document_characters=1000, title_column="title", url_column="url",
                filename_column="file", enable_in_domain=True)
    return NS(datasource=source, base_settings=NS(datasource_type="AzureCognitiveSearch"),
              azure_openai=NS(embedding_name="embedding-deployment", embedding_endpoint=None),
              search=NS(vectorization_dimensions=None))


@pytest.fixture
def search(monkeypatch):
    async def results():
        yield {"body": "Leave policy", "summary": "20 days", "title": "HR", "file": "hr.pdf", "url": "https://example.org/hr"}
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.search = AsyncMock(side_effect=lambda **kw: results())
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr(retrieval, "SearchClient", constructor)
    return constructor, client


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["simple", "semantic", "vector", "vector_simple_hybrid", "vector_semantic_hybrid"])
async def test_search_modes_and_citations(search, mode):
    config = settings(mode)
    aoai = NS(embeddings=NS(create=AsyncMock(return_value=NS(data=[NS(embedding=[0.1, 0.2])]))))
    context = await retrieval.retrieve_documents(config, {}, "leave?", aoai)
    options = search[1].search.call_args.kwargs
    assert options["filter"] == "active eq true"
    assert options["top"] == 3
    assert options["search_text"] == (None if mode == "vector" else "leave?")
    if mode.startswith("vector"):
        assert options["vector_queries"][0].vector == [0.1, 0.2]
        assert options["vector_filter_mode"] == "preFilter"
        aoai.embeddings.create.assert_awaited_once_with(model="embedding-deployment", input="leave?")
    else:
        aoai.embeddings.create.assert_not_awaited()
    if "semantic" in mode:
        assert options["query_type"] == "semantic"
    assert context["citations"][0]["content"] == "Leave policy\n20 days"
    assert context["citations"][0]["filepath"] == "hr.pdf"
    assert context["citations"][0]["id"] == "1"


@pytest.mark.asyncio
async def test_managed_identity(search, monkeypatch):
    credential = MagicMock()
    credential.__aenter__ = AsyncMock(return_value=credential)
    credential.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(retrieval, "DefaultAzureCredential", lambda: credential)
    await retrieval.retrieve_documents(settings(key=None), {}, "question", None)
    assert search[0].call_args.kwargs["credential"] is credential
    credential.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_datasource_and_unsupported(search):
    config = settings()
    config.datasource = None
    config.base_settings.datasource_type = None
    assert await retrieval.retrieve_documents(config, {}, "question", None) is None
    config.base_settings.datasource_type = "Pinecone"
    with pytest.raises(ValueError):
        await retrieval.retrieve_documents(config, {}, "question", None)
    config.datasource = NS(_type="pinecone")
    with pytest.raises(ValueError, match="separate"):
        await retrieval.retrieve_documents(config, {}, "question", None)
    search[0].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["vector_columns", "embedding_name"])
async def test_no_vector_downgrade(search, missing):
    config = settings("vector")
    setattr(config.datasource if missing == "vector_columns" else config.azure_openai, missing, None)
    with pytest.raises(ValueError):
        await retrieval.retrieve_documents(config, {}, "question", None)
    search[1].search.assert_not_called()


@pytest.mark.asyncio
async def test_filters_are_per_request_and_cannot_be_overridden(search, monkeypatch):
    config = settings("vector_simple_hybrid")
    config.datasource.permitted_groups_column = "groups"
    with pytest.raises(PermissionError):
        await retrieval.retrieve_documents(config, {}, "question", None)
    search[0].assert_not_called()
    monkeypatch.setattr(retrieval, "generateFilterString", lambda token, column: "false")
    aoai = NS(embeddings=NS(create=AsyncMock(return_value=NS(data=[NS(embedding=[1])]))))
    await retrieval.retrieve_documents(config, {"X-MS-TOKEN-AAD-ACCESS-TOKEN": "token", "filter": "true"}, "ignore filters", aoai)
    assert search[1].search.call_args.kwargs["filter"] == "(active eq true) and (false)"
    assert config.datasource.filter == "active eq true"


def test_group_filter_empty_invalid_and_graph_failures(monkeypatch):
    monkeypatch.setattr(utils, "fetchUserGroups", lambda token: [])
    assert utils.generateFilterString("token", "groups") == "false"
    monkeypatch.setattr(utils, "fetchUserGroups", lambda token: [{"id": "' or true"}])
    with pytest.raises(PermissionError):
        utils.generateFilterString("token", "groups")


def test_graph_pagination_failure_and_untrusted_next_link(monkeypatch):
    page = MagicMock(status_code=200)
    page.json.return_value = {"value": [{"id": "group"}], "@odata.nextLink": "https://graph.microsoft.com/next"}
    get = MagicMock(side_effect=[page, MagicMock(status_code=403)])
    monkeypatch.setattr(utils.requests, "get", get)
    with pytest.raises(PermissionError):
        utils.fetchUserGroups("token")
    get.reset_mock()
    with pytest.raises(PermissionError):
        utils.fetchUserGroups("token", "https://attacker.example/steal")
    get.assert_not_called()


def test_response_contract():
    context = {"citations": [{"id": "1", "content": "policy"}], "intent": "question"}
    completion = NS(id="id", model="model", created=1, object="chat.completion",
                    choices=[NS(message=NS(content="Answer [doc1]"))])
    response = utils.format_non_streaming_response(completion, {"conversation_id": "c"}, "request", context)
    assert json.loads(response["choices"][0]["messages"][0]["content"]) == context
    assert response["choices"][0]["messages"][1]["content"] == "Answer [doc1]"
    chunk = NS(id="id", model="model", created=1, object="chat.completion.chunk", choices=[])
    event = utils.format_stream_response(chunk, {}, "request", context)
    assert json.loads(event["choices"][0]["messages"][0]["content"]) == context
    chunk.choices = [NS(delta=NS(role=None, content="Answer [doc1]"))]
    event = utils.format_stream_response(chunk, {}, "request")
    assert event["choices"][0]["messages"] == [{"role": "assistant", "content": "Answer [doc1]"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("grounded", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_chat_end_to_end_mocked(monkeypatch, grounded, stream):
    import app
    config = settings()
    config.azure_openai = NS(system_message="System policy", temperature=0, max_tokens=100,
                            top_p=1, stop_sequence=None, stream=stream, model="deployment")
    config.ui = NS(title="Test")
    config.base_settings.use_promptflow = False
    if not grounded:
        config.datasource = None
        config.base_settings.datasource_type = None
    monkeypatch.setattr(app, "app_settings", config)
    monkeypatch.setattr(app, "MS_DEFENDER_ENABLED", False)
    context = {"citations": [{"id": "1", "content": "policy"}], "intent": "question"} if grounded else None
    monkeypatch.setattr(app, "retrieve_documents", AsyncMock(return_value=context))
    completion = NS(id="id", model="model", created=1, object="chat.completion",
                    choices=[NS(message=NS(content="Answer [doc1]"))])
    class Stream:
        def __aiter__(self):
            return self.events()
        async def events(self):
            for text in (None, "Answer ", "[doc1]"):
                yield NS(id="id", model="model", created=1, object="chat.completion.chunk",
                         choices=[NS(delta=NS(role="assistant", content=text))])
        close = AsyncMock()
    raw = NS(parse=lambda: Stream() if stream else completion, headers={"apim-request-id": "r"})
    create = AsyncMock(return_value=raw)
    client = NS(chat=NS(completions=NS(with_raw_response=NS(create=create))), close=AsyncMock())
    monkeypatch.setattr(app, "init_openai_client", AsyncMock(return_value=client))
    body = {"messages": [{"role": "user", "content": "old"},
                         {"role": "assistant", "content": "old answer", "context": "invalid legacy json"},
                         {"role": "tool", "content": "old citations"},
                         {"role": "user", "content": "question"}], "history_metadata": {"id": "c"}}
    if stream:
        generator = await app.stream_chat_request(body, {})
        events = [event async for event in generator]
        messages = [m for event in events if event for m in event["choices"][0]["messages"]]
        assert "".join(m["content"] for m in messages if m["role"] == "assistant") == "Answer [doc1]"
    else:
        result = await app.complete_chat_request(body, {})
        messages = result["choices"][0]["messages"]
        assert result["history_metadata"] == {"id": "c"}
    assert len([m for m in messages if m["role"] == "tool"]) == int(grounded)
    args = create.call_args.kwargs
    assert "data_sources" not in json.dumps(args)
    assert "extra_body" not in args
    assert args["messages"][0] == {"role": "system", "content": "System policy"}
    assert args["messages"][-1] == {"role": "user", "content": "question"}
    assert all(set(m) == {"role", "content"} for m in args["messages"])
    assert any(m["content"] == "old answer" for m in args["messages"])
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_real_sdk_serializes_standard_chat_request(monkeypatch):
    import httpx
    from openai import AsyncAzureOpenAI
    import app
    captured = []
    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "id", "model": "deployment", "created": 1,
            "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Answer [doc1]"}}]})
    client = AsyncAzureOpenAI(api_key="test-key", api_version="2024-05-01-preview",
        azure_endpoint="https://example.openai.azure.com",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    config = settings()
    config.azure_openai = NS(system_message="policy", temperature=0, max_tokens=100, top_p=1,
                            stop_sequence=None, stream=False, model="deployment")
    monkeypatch.setattr(app, "app_settings", config)
    monkeypatch.setattr(app, "MS_DEFENDER_ENABLED", False)
    monkeypatch.setattr(app, "init_openai_client", AsyncMock(return_value=client))
    monkeypatch.setattr(app, "retrieve_documents", AsyncMock(return_value={"citations": [], "intent": "question"}))
    await app.send_chat_request({"messages": [{"role": "user", "content": "question"}]}, {})
    assert "data_sources" not in captured[0]
    assert "extra_body" not in captured[0]
    assert all(set(m) == {"role", "content"} for m in captured[0]["messages"])


@pytest.mark.asyncio
async def test_legacy_embedding_endpoint_wire_path(search, monkeypatch):
    import httpx
    from openai import AsyncAzureOpenAI
    captured = []
    def respond(request):
        captured.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "object": "embedding", "embedding": [0.1]}],
                                         "model": "embedding", "object": "list", "usage": {"prompt_tokens": 1, "total_tokens": 1}})
    def client_factory(**kwargs):
        return AsyncAzureOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(retrieval, "AsyncAzureOpenAI", client_factory)
    config = settings("vector")
    config.azure_openai = NS(embedding_name=None, embedding_endpoint="https://example.openai.azure.com/openai/deployments/embed/embeddings?api-version=2024-02-01",
                            embedding_key="embedding-key", preview_api_version="2024-05-01-preview")
    await retrieval.retrieve_documents(config, {}, "question", None)
    assert captured[0].url.path == "/openai/deployments/embed/embeddings"
    assert captured[0].url.params["api-version"] == "2024-02-01"


@pytest.mark.asyncio
async def test_access_lookup_failure_never_searches(search, monkeypatch):
    config = settings()
    config.datasource.permitted_groups_column = "groups"
    def denied(*args):
        raise PermissionError("Graph unavailable")
    monkeypatch.setattr(retrieval, "generateFilterString", denied)
    with pytest.raises(PermissionError):
        await retrieval.retrieve_documents(config, {"X-MS-TOKEN-AAD-ACCESS-TOKEN": "token"}, "question", None)
    search[0].assert_not_called()
