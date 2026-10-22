import inspect
import unittest

from entity.data import AsrRequestParams, get_asr_params


class RequestDefaultsTests(unittest.TestCase):
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
