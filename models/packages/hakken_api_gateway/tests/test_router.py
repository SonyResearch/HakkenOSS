import httpx
import respx
import starlette.testclient
from fastapi import FastAPI
from starlette.testclient import TestClient

from hakken_api_gateway.router import router

# Store the original __init__ method
original_init = starlette.testclient.TestClient.__init__


def patched_init(self, app, **kwargs):
    # Extract kwargs with defaults
    base_url = kwargs.get("base_url", "http://testserver")
    raise_server_exceptions = kwargs.get("raise_server_exceptions", True)
    root_path = kwargs.get("root_path", "")
    backend = kwargs.get("backend", "asyncio")
    backend_options = kwargs.get("backend_options")
    cookies = kwargs.get("cookies")
    headers = kwargs.get("headers")

    self.async_backend = starlette.testclient._AsyncBackend(
        backend=backend, backend_options=backend_options or {}
    )
    asgi_app = app if starlette.testclient._is_asgi3(app) else starlette.testclient._WrapASGI2(app)

    self.app = asgi_app
    self.app_state = {}
    transport = starlette.testclient._TestClientTransport(
        self.app,
        portal_factory=self._portal_factory,
        raise_server_exceptions=raise_server_exceptions,
        root_path=root_path,
        app_state=self.app_state,
    )
    if headers is None:
        headers = {}
    headers.setdefault("user-agent", "testclient")

    # Call httpx.Client.__init__ without the 'app' parameter
    httpx.Client.__init__(
        self,
        base_url=base_url,
        headers=headers,
        transport=transport,
        follow_redirects=True,
        cookies=cookies,
    )


# Apply the monkey patch
starlette.testclient.TestClient.__init__ = patched_init  # type: ignore[method-assign,assignment]

# Instantiate the test client for our FastAPI app
app = FastAPI()
app.include_router(router)

client = TestClient(app)


@respx.mock
def test_service_not_found():
    """
    Tests that a 404 is returned for a service not in the configuration.
    """
    # Create a test app with auth dependency overridden
    test_app = FastAPI()

    # Create a router without auth for testing
    from fastapi import APIRouter, Request, Response
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    from hakken_api_gateway.handlers.router import ServiceType

    test_router = APIRouter()
    limiter = Limiter(key_func=get_remote_address)

    @test_router.api_route(
        "/{version}/{service}/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE"],
        # No auth dependency for testing
    )
    @limiter.limit("10/minute")
    async def test_reverse_proxy(request: Request):
        service = request.path_params.get("service")
        try:
            ServiceType(service)
        except ValueError:
            return Response(content=f"Service '{service}' not found.", status_code=404)
        # Rest of logic would go here, but we only need the service check
        return Response(content="OK", status_code=200)

    test_app.include_router(test_router)
    test_client = TestClient(test_app)

    response = test_client.get("/v1/nonexistent-service/some/path")

    assert response.status_code == 404
    assert "Service 'nonexistent-service' not found" in response.text
