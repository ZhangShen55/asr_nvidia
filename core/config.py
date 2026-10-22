import tomli
import os
from dataclasses import dataclass
from typing import Optional


def _load_toml(path: str) -> dict:
    with open(path, "rb") as f:
        return tomli.load(f)


@dataclass
class Settings:
    # 从环境变量读取 config.toml 路径
    config_path: str = os.getenv("CONFIG_PATH", "./config.toml")
    _cfg: dict = None  # 实际配置字典

    def __post_init__(self):
        self._cfg = _load_toml(self.config_path)

    # 基础配置
    @property
    def log_path(self) -> str:
        return self._cfg.get("log_path", "./asr_service.log")

    @property
    def id_engine(self) -> str:
        import uuid
        return self._cfg.get("id_engine", f"seacraft-asr-{uuid.uuid4()}")

    @property
    def version(self) -> str:
        return self._cfg.get("version", "seacraft-voice-analysis-app-pro")

    # 设备与并发
    @property
    def ngpu(self) -> int:
        return self._cfg.get("ngpu", 1)

    @property
    def device(self) -> str:
        return self._cfg.get("device", "cuda:0")

    @property
    def ncpu(self) -> int:
        return self._cfg.get("ncpu", 4)

    @property
    def concurrency(self) -> int:
        return self._cfg.get("concurrency", 5)

    @property
    def instance_count(self) -> int:
        return self._cfg.get("instance_count", 4)

    # 模型路径 - 从嵌套的 [model_paths] 部分读取
    @property
    def vad_model_dir(self) -> str:
        model_paths = self._cfg.get("model_paths", {})
        return model_paths.get("vad_model_dir", "/model/speech_fsmn_vad_zh-cn-16k-common-pytorch")

    @property
    def punc_model_dir(self) -> str:
        model_paths = self._cfg.get("model_paths", {})
        return model_paths.get("punc_model_dir", "/model/punc_ct-transformer_cn-en-common-vocab471067-large")

    @property
    def asr_model_dir(self) -> str:
        model_paths = self._cfg.get("model_paths", {})
        return model_paths.get("asr_model_dir", "/model/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")

    @property
    def spk_model_dir(self) -> Optional[str]:
        model_paths = self._cfg.get("model_paths", {})
        return model_paths.get("spk_model_dir", "/model/speech_campplus_sv_zh_en_16k-common_advanced")

    @property
    def emotion_model_dir(self) -> str:
        model_paths = self._cfg.get("model_paths", {})
        # 注意：原 JSON 中 key 为 emotion_modek_dir（拼写错误），这里修正为 emotion_model_dir
        return model_paths.get("emotion_model_dir", "/model/emotion2vec_plus_seed")

    @property
    def whisper_model_dir(self) -> str:
        model_paths = self._cfg.get("model_paths", {})
        return model_paths.get("whisper_model_dir", "/model/faster-whisper-large-v3")

    # 语速计算配置 - 从嵌套的 [speech_rate] 部分读取
    @property
    def speech_rate_factor(self) -> float:
        speech_rate = self._cfg.get("speech_rate", {})
        return float(speech_rate.get("rate_factor", 1.0))

    # 功能开关 - 从嵌套的 [features] 部分读取
    @property
    def open_spk(self) -> bool:
        features = self._cfg.get("features", {})
        return features.get("open_spk", False)

    @property
    def open_mul_lang(self) -> bool:
        features = self._cfg.get("features", {})
        return features.get("open_mul_lang", False)

    @property
    def open_emotion(self) -> bool:
        features = self._cfg.get("features", {})
        return features.get("open_emotion", False)

    @property
    def ban_hotword(self) -> bool:
        features = self._cfg.get("features", {})
        return features.get("ban_hotword", False)

    # 长音频分块配置
    @property
    def chunk_threshold_minutes(self) -> float:
        return float(self._cfg.get("audio_chunk", {}).get("threshold_minutes", 90))

    @property
    def chunk_minutes(self) -> float:
        return float(self._cfg.get("audio_chunk", {}).get("chunk_minutes", 60))

    @property
    def min_last_chunk_minutes(self) -> float:
        return float(self._cfg.get("audio_chunk", {}).get("min_last_chunk_minutes", 15))

    @property
    def chunk_overlap_seconds(self) -> float:
        return float(self._cfg.get("audio_chunk", {}).get("overlap_seconds", 15))

    @property
    def chunk_retry_count(self) -> int:
        return int(self._cfg.get("audio_chunk", {}).get("chunk_retry_count", 2))

    # 文本规范化（ITN）配置
    @property
    def itn_enabled(self) -> bool:
        return bool(self._cfg.get("itn", {}).get("enabled", True))

    @property
    def itn_enable_standalone_number(self) -> bool:
        return bool(self._cfg.get("itn", {}).get("enable_standalone_number", True))

    @property
    def itn_enable_0_to_9(self) -> bool:
        return bool(self._cfg.get("itn", {}).get("enable_0_to_9", False))

    @property
    def itn_remove_interjections(self) -> bool:
        return bool(self._cfg.get("itn", {}).get("remove_interjections", True))

    @property
    def itn_failure_mode(self) -> str:
        mode = str(self._cfg.get("itn", {}).get("failure_mode", "passthrough")).strip().lower()
        if mode not in {"passthrough", "raise"}:
            raise ValueError("[itn].failure_mode must be one of: passthrough, raise")
        return mode

    @property
    def itn_config(self):
        from utils.itn import ItnConfig

        return ItnConfig(
            enabled=self.itn_enabled,
            enable_standalone_number=self.itn_enable_standalone_number,
            enable_0_to_9=self.itn_enable_0_to_9,
            remove_interjections=self.itn_remove_interjections,
            failure_mode=self.itn_failure_mode,
        )

    # Qwen3-ASR 适配链路（/v2/seacraft_asr）配置 - 从嵌套的 [qwen_asr] 部分读取
    @property
    def qwen_asr_enabled(self) -> bool:
        return bool(self._cfg.get("qwen_asr", {}).get("enabled", True))

    @property
    def qwen_base_url(self) -> str:
        return str(self._cfg.get("qwen_asr", {}).get("base_url", "http://127.0.0.1:8806")).rstrip("/")

    @property
    def qwen_model(self) -> str:
        # 留空表示从 /v1/models 自动发现
        return str(self._cfg.get("qwen_asr", {}).get("model", "")).strip()

    @property
    def qwen_segment_concurrency(self) -> int:
        """单个音频内同时发往 Qwen 的 VAD 段请求数。"""
        return int(self._cfg.get("qwen_asr", {}).get("segment_concurrency", 4))

    @property
    def qwen_request_concurrency(self) -> int:
        """所有音频合计发往 Qwen 的请求数上限（全局共享）。"""
        return int(self._cfg.get("qwen_asr", {}).get("request_concurrency", 32))

    @property
    def qwen_audio_concurrency(self) -> int:
        """同时处理的音频文件数上限。"""
        return int(self._cfg.get("qwen_asr", {}).get("audio_concurrency", 8))

    @property
    def qwen_queue_max_size(self) -> int:
        return int(self._cfg.get("qwen_asr", {}).get("queue_max_size", 64))

    @property
    def qwen_queue_wait_timeout_s(self) -> float:
        """0 表示队列满时立即拒绝；大于 0 才排队等待。"""
        return float(self._cfg.get("qwen_asr", {}).get("queue_wait_timeout_s", 0))

    @property
    def qwen_request_timeout_s(self) -> float:
        return float(self._cfg.get("qwen_asr", {}).get("request_timeout_s", 120))

    @property
    def qwen_connect_timeout_s(self) -> float:
        return float(self._cfg.get("qwen_asr", {}).get("connect_timeout_s", 10))

    @property
    def qwen_retry_count(self) -> int:
        return int(self._cfg.get("qwen_asr", {}).get("retry_count", 1))

    @property
    def qwen_min_segment_duration_s(self) -> float:
        return float(self._cfg.get("qwen_asr", {}).get("min_segment_duration_s", 0.3))

    @property
    def qwen_vad_max_single_segment_time_ms(self) -> int:
        return int(self._cfg.get("qwen_asr", {}).get("vad_max_single_segment_time_ms", 30000))

    @property
    def qwen_vad_max_end_silence_time_ms(self) -> int:
        return int(self._cfg.get("qwen_asr", {}).get("vad_max_end_silence_time_ms", 800))

    @property
    def qwen_vad_concurrency(self) -> int:
        """VAD 模型实例池大小：FSMN-VAD 实例不可并发复用，故用池子隔离。"""
        return int(self._cfg.get("qwen_asr", {}).get("vad_concurrency", 4))

    @property
    def qwen_speaker_similarity_threshold(self) -> float:
        """CAM++ 说话人聚类的余弦相似度阈值。

        取值 0.30 来自实测 sweep（见 docs/qwen-asr-v2-test-report.md）：
        单说话人音频在该阈值下聚为 1 簇（BCubed F1 = 1.0），
        三说话人课堂录音聚为 3 簇（F1 = 0.816）；阈值升高到 0.6 以上会
        把同一说话人拆成 6~63 簇。
        """
        return float(self._cfg.get("qwen_asr", {}).get("speaker_similarity_threshold", 0.3))

    @property
    def qwen_local_gpu_concurrency(self) -> int:
        """本机 GPU（CAM++ / emotion2vec）推理的并发路数上限。

        限制并发而不是完全串行：完全串行时 8 路并发下 role/emotion
        会成为端到端时延的主瓶颈。
        """
        return int(self._cfg.get("qwen_asr", {}).get("local_gpu_concurrency", 4))

    # v2 链路模型落卡 - 从嵌套的 [model_device] 部分读取，三者绑定同一张卡
    @property
    def model_device_vad(self) -> int:
        return int(self._cfg.get("model_device", {}).get("vad", 0))

    @property
    def model_device_emotion2vec(self) -> int:
        return int(self._cfg.get("model_device", {}).get("emotion2vec", 0))

    @property
    def model_device_speech_campplus(self) -> int:
        return int(self._cfg.get("model_device", {}).get("speech_campplus", 0))


settings = Settings()
