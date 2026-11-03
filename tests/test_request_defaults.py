import inspect
import unittest
from itertools import product
from types import SimpleNamespace

from entity.data import AsrRequestParams, get_asr_params


class RequestDefaultsTests(unittest.TestCase):
    def test_capability_classification_for_all_switch_combinations(self):
        from api.routes.asr_qwen_common import resolve_v3_capabilities

        for spk, emotion, identify in product((False, True), repeat=3):
            with self.subTest(spk=spk, emotion=emotion, identify=identify):
                result = resolve_v3_capabilities(SimpleNamespace(
                    showSpk=spk, showEmotion=emotion, showRoleIdentify=identify,
                ))
                self.assertEqual(result.need_role, spk or identify)
                self.assertEqual(result.need_emotion, emotion)
                self.assertEqual(result.category, "B" if spk or emotion or identify else "A")

    def test_openapi_enhancement_defaults_are_false(self):
        from fastapi import FastAPI
        from api.routes.asr_v3 import router

        app = FastAPI()
        app.include_router(router)
        schemas = app.openapi()["components"]["schemas"].values()
        body = next(schema for schema in schemas if "showSpk" in schema.get("properties", {}))
        for key in ("showSpk", "showEmotion", "showRoleIdentify"):
            self.assertIs(body["properties"][key]["default"], False)

    def test_v118_form_defaults_match_deployment_contract(self) -> None:
        parameters = inspect.signature(get_asr_params).parameters

        # 12 服务器与 Mac 的差异仅限 showSpk/showEmotion 默认值。
        self.assertIs(parameters["showSpk"].default, False)
        self.assertIs(parameters["showEmotion"].default, False)
        self.assertIs(parameters["showRoleIdentify"].default, False)
        self.assertIs(parameters["wordTimestamps"].default, False)
        self.assertEqual(parameters["language"].default, "auto")

    def test_internal_request_defaults_match_v118_form_contract(self) -> None:
        parameters = inspect.signature(AsrRequestParams).parameters

        self.assertIs(parameters["showSpk"].default, False)
        self.assertIs(parameters["showEmotion"].default, False)
        self.assertIs(parameters["showRoleIdentify"].default, False)


if __name__ == "__main__":
    unittest.main()
