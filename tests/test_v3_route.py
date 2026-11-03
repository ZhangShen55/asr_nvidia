from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.asr_v3 import router


def test_v3_is_registered_and_v2_is_removed():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/v3/seacraft_asr" in paths
    assert "/v2/seacraft_asr" not in paths
    assert client.post("/v2/seacraft_asr").status_code == 404


def test_v3_openapi_keeps_false_enhancement_defaults():
    app = FastAPI()
    app.include_router(router)
    schemas = app.openapi()["components"]["schemas"].values()
    body = next(schema for schema in schemas if "showSpk" in schema.get("properties", {}))
    for key in ("showSpk", "showEmotion", "showRoleIdentify"):
        assert body["properties"][key]["default"] is False


def test_main_application_registers_only_v3_qwen_route():
    import main

    paths = {route.path for route in main.app.routes if hasattr(route, "path")}
    assert "/v3/seacraft_asr" in paths
    assert "/v2/seacraft_asr" not in paths
