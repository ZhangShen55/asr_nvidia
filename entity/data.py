from dataclasses import dataclass
from fastapi import UploadFile, Depends, Form, File, Request
from typing import List, Optional
from typing import Annotated


@dataclass
class AsrRequestParams:
    audioFile: UploadFile
    wordTimestamps: bool
    hotWords: List[str] = None
    # initialPrompt: Optional[str]
    # vadFilter: bool = False
    # output: Optional[str] = "json"
    language: Optional[str] = "auto"
    # task: Optional[str] = "transcribe"
    showSpk: bool = False
    openPanel: bool = False
    showEmotion: bool = False
    showSpeed: bool = False
    showRoleIdentify: bool = False
    # ws: Optional[websockets.WebSocketClientProtocol] = None  # 如果使用，需要取消注释

    # def hotwords_as_string(self) -> str:
    #     return ','.join(self.hotWords)


async def get_asr_params(
    request: Request,
    audioFile: Annotated[UploadFile, File(...)],
    language: Annotated[str, Form()] = "auto",
    wordTimestamps: Annotated[bool, Form()] = False,
    output: Annotated[Optional[str], Form()] = None,
    hotWords: Annotated[Optional[List[str]], Form()] = None,
    showSpk: Annotated[bool, Form()] = False,
    openPanel: Annotated[bool, Form()] = False,
    showEmotion: Annotated[bool, Form()] = False,
    showSpeed: Annotated[bool, Form()] = False,
    showRoleIdentify: Annotated[bool, Form()] = False,
) -> AsrRequestParams:
    # 显式提交空 language 与未提交 language 语义不同：
    # 前者按不支持语言处理，后者使用默认 auto。
    form = await request.form()
    if "language" in form:
        raw_language = form.get("language")
        if isinstance(raw_language, str):
            language = raw_language
    if hotWords is None:
        hotWords = []
    if output is None:
        output = "json"
    # if initialPrompt is None:
    #     initialPrompt = "请转录为中文简体。"
    return AsrRequestParams(
        audioFile=audioFile,
        # task=task,
        language=language,
        # initialPrompt=initialPrompt,
        # vadFilter=vadFilter,
        wordTimestamps=wordTimestamps,
        # output=output,
        hotWords=hotWords,
        showSpk=showSpk,
        openPanel=openPanel,
        showEmotion=showEmotion,
        showSpeed=showSpeed,
        showRoleIdentify=showRoleIdentify
        # ws=ws
    )
