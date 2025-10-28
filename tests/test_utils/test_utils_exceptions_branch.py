import pytest

from gpt2giga.utils import exceptions_handler


@pytest.mark.asyncio
async def test_exceptions_handler_unexpected_structure(monkeypatch):
    import gigachat

    class FakeResponseError(gigachat.exceptions.ResponseError):
        pass

    err = FakeResponseError("only-one-arg")

    @exceptions_handler
    async def boom():
        raise err

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ex:
        await boom()
    assert ex.value.status_code == 500
    assert "Unexpected ResponseError structure" in str(ex.value.detail)
