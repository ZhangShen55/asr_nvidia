import tomli
import os
import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


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

    # 日志配置
    @property
    def log_path(self) -> str:
        log = self._cfg.get("log", {})
        if "log_path" in log:
            return str(log["log_path"])
        # Keep a short migration window for deployments that have not moved the key.
        if "log_path" in self._cfg:
            logger.warning(
                "顶层 log_path 已废弃，请迁移到 [log].log_path；当前仍兼容读取该配置"
            )
            return str(self._cfg["log_path"])
        return "./asr_service.log"

    @property
    def log_max_days(self) -> int:
        """日志保留天数。"""
        return int(self._cfg.get("log", {}).get("max_days", 7))

    @property
    def log_max_size_mb(self) -> int:
        """单个日志文件的大小上限（MB）。"""
        return int(self._cfg.get("log", {}).get("max_size_mb", 100))

    # 服务实例数量
    @property
    def instance_count(self) -> int:
        """容器内并行运行的 uvicorn 实例数量。

        每个实例各自持有一份本地模型权重，因此实例数即显存倍数；
        默认单实例，避免显存被无意放大。
        """
        return int(self._cfg.get("instance_count", 1))

    # 模型路径 - 从嵌套的 [model_paths] 部分读取
    @property
    def vad_model_dir(self) -> str:
        model_paths = self._cfg.get("model_paths", {})
        return model_paths.get("vad_model_dir", "/model/speech_fsmn_vad_zh-cn-16k-common-pytorch")

    @property
    def spk_model_dir(self) -> Optional[str]:
        model_paths = self._cfg.get("model_paths", {})
        return model_paths.get("spk_model_dir", "/model/speech_campplus_sv_zh_en_16k-common_advanced")

    @property
    def emotion_model_dir(self) -> str:
        model_paths = self._cfg.get("model_paths", {})
        # 注意：原 JSON 中 key 为 emotion_modek_dir（拼写错误），这里修正为 emotion_model_dir
        return model_paths.get("emotion_model_dir", "/model/emotion2vec_plus_seed")

    # 语速计算配置 - 从嵌套的 [speech_rate] 部分读取
    @property
    def speech_rate_factor(self) -> float:
        speech_rate = self._cfg.get("speech_rate", {})
        return float(speech_rate.get("rate_factor", 1.0))

    # 功能开关 - 从嵌套的 [features] 部分读取
    @property
    def open_v2_role(self) -> bool:
        """v2 链路的 CAM++ 说话人嵌入开关。"""
        features = self._cfg.get("features", {})
        return bool(features.get("open_v2_role", True))

    @property
    def open_v2_emotion(self) -> bool:
        """v2 链路的 emotion2vec 情绪开关。"""
        features = self._cfg.get("features", {})
        return bool(features.get("open_v2_emotion", True))

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
    def qwen_vad_speech_noise_thres(self) -> float:
        """FSMN-VAD 的语音/噪声判定阈值（越小越灵敏）。"""
        raw = self._cfg.get("qwen_asr", {}).get("vad_speech_noise_thres", 0.6)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("[qwen_asr].vad_speech_noise_thres 必须是 0.0~1.0 数值") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("[qwen_asr].vad_speech_noise_thres 必须在 0.0~1.0 范围内")
        return value

    @staticmethod
    def _validated_vad_float(section: dict, key: str, default: float, lower: float, upper: float) -> float:
        raw = section.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"[qwen_asr].{key} 必须是有限数值，范围 {lower}~{upper}") from exc
        if not math.isfinite(value) or value < lower or value > upper:
            raise ValueError(f"[qwen_asr].{key} 必须在 {lower}~{upper} 范围内")
        return value

    @property
    def qwen_vad_decibel_thres(self) -> float:
        """FunASR 帧级 log 能量阈值，单位不是 dBFS；-100 表示不额外限制。"""
        return self._validated_vad_float(self._cfg.get("qwen_asr", {}), "vad_decibel_thres", -100.0, -100.0, 0.0)

    @property
    def qwen_vad_snr_thres(self) -> float:
        """FunASR 帧级 SNR 阈值；-100 表示不额外限制。"""
        return self._validated_vad_float(self._cfg.get("qwen_asr", {}), "vad_snr_thres", -100.0, -100.0, 100.0)

    @property
    def qwen_vad_low_volume_rms_dbfs_thres(self) -> float:
        """段级 RMS dBFS 低音量门限，低于该值才会被过滤。"""
        return self._validated_vad_float(
            self._cfg.get("qwen_asr", {}), "vad_low_volume_rms_dbfs_thres", -100.0, -100.0, 0.0
        )

    @property
    def qwen_vad_low_volume_filter_enabled(self) -> bool:
        raw = self._cfg.get("qwen_asr", {}).get("vad_low_volume_filter_enabled", False)
        if not isinstance(raw, bool):
            raise ValueError("[qwen_asr].vad_low_volume_filter_enabled 必须是 true 或 false")
        return raw

    # 直名别名便于标定脚本和运维代码读取，同时保持 qwen_asr 配置命名兼容。
    @property
    def vad_decibel_thres(self) -> float:
        return self.qwen_vad_decibel_thres

    @property
    def vad_snr_thres(self) -> float:
        return self.qwen_vad_snr_thres

    @property
    def vad_low_volume_rms_dbfs_thres(self) -> float:
        return self.qwen_vad_low_volume_rms_dbfs_thres

    @property
    def vad_low_volume_filter_enabled(self) -> bool:
        return self.qwen_vad_low_volume_filter_enabled

    @property
    def vad_device(self) -> str:
        """VAD 模型的运行设备。

        默认 "cpu"：实测 CPU 与 GPU 的 FSMN-VAD 分段边界逐段完全一致，
        而 CPU 单实例速度与 GPU 相当（1.24s vs 1.29s / 300s 片段），
        且多进程 worker 放在 CPU 上不会各自建立 CUDA 上下文。
        取 "cuda:N" 形式时可切回 GPU。
        """
        return str(self._cfg.get("qwen_asr", {}).get("vad_device", "cpu")).strip()

    @property
    def qwen_vad_pool_size(self) -> int:
        """常驻 VAD worker 进程数（每个进程持有一份 FSMN-VAD 实例）。"""
        return int(self._cfg.get("qwen_asr", {}).get("vad_pool_size", 8))

    @property
    def qwen_vad_pool_max(self) -> int:
        """VAD worker 进程的扩容上限。"""
        return int(self._cfg.get("qwen_asr", {}).get("vad_pool_max", 16))

    @property
    def qwen_vad_queue_watermark(self) -> int:
        """等待队列深度超过该值时临时拉起额外 worker。"""
        return int(self._cfg.get("qwen_asr", {}).get("vad_queue_watermark", 4))

    @property
    def qwen_vad_collapse_timeout_s(self) -> float:
        """额外 worker 的空闲回收时间（秒）。"""
        return float(self._cfg.get("qwen_asr", {}).get("vad_collapse_timeout_s", 120))

    @property
    def qwen_vad_queue_timeout_s(self) -> float:
        """请求等待 VAD worker 的超时时间（秒）。"""
        return float(self._cfg.get("qwen_asr", {}).get("vad_queue_timeout_s", 120))

    @property
    def qwen_vad_call_timeout_s(self) -> float:
        """单个 VAD worker 执行一份音频的超时时间（秒）。"""
        return float(self._cfg.get("qwen_asr", {}).get("vad_call_timeout_s", 120))

    @property
    def qwen_speaker_similarity_threshold(self) -> float:
        """CAM++ 说话人聚类的余弦相似度阈值。

        取值 0.30 来自实测 sweep（见 docs/qwen-asr-v2-test-report.md）：
        单说话人音频在该阈值下聚为 1 簇（BCubed F1 = 1.0），
        三说话人课堂录音聚为 3 簇（F1 = 0.816）；阈值升高到 0.6 以上会
        把同一说话人拆成 6~63 簇。
        """
        return float(self._cfg.get("qwen_asr", {}).get("speaker_similarity_threshold", 0.3))

    # 复读（生成幻觉循环）后处理
    @property
    def qwen_repetition_guard_enabled(self) -> bool:
        """是否对复读分段做检测与折叠修复。"""
        return bool(self._cfg.get("qwen_asr", {}).get("repetition_guard_enabled", True))

    @property
    def qwen_repetition_max_chars_per_second(self) -> int:
        """按语音时长估算合理字符数的上限系数（字符/秒）。

        标定依据（68,018 段真实样本）：正常段 p99.9 = 20.3 字符/秒、
        观测到的最高值 26.2；复读段为 282~313。取 40 留约 1.5 倍余量，
        同时远低于复读的量级。
        """
        return int(self._cfg.get("qwen_asr", {}).get("repetition_max_chars_per_second", 40))

    @property
    def qwen_repetition_min_chars(self) -> int:
        """判定长度异常的绝对下限字符数。"""
        return int(self._cfg.get("qwen_asr", {}).get("repetition_min_chars", 120))

    @property
    def qwen_repetition_min_unit(self) -> int:
        """参与循环判定的重复子串最小长度（字符）。"""
        return int(self._cfg.get("qwen_asr", {}).get("repetition_min_unit", 4))

    @property
    def qwen_repetition_min_times(self) -> int:
        """判定为循环所需的最小连续重复次数。"""
        return int(self._cfg.get("qwen_asr", {}).get("repetition_min_times", 3))

    # v2 链路模型落卡 - 从嵌套的 [model_device] 部分读取
    @property
    def model_device_emotion2vec(self) -> str:
        return self.model_device_emotion2vec_devices[0]

    @property
    def model_device_speech_campplus(self) -> str:
        return self.model_device_speech_campplus_devices[0]

    @property
    def model_device_emotion2vec_devices(self) -> list[str]:
        """emotion2vec 模型池实例设备；数组长度即实例数。"""
        return self._model_pool_devices("emotion2vec")

    @property
    def model_device_speech_campplus_devices(self) -> list[str]:
        """CAM++ 模型池实例设备；数组长度即实例数。"""
        return self._model_pool_devices("speech_campplus")

    @property
    def local_model_pool_emotion2vec_devices(self) -> list[str]:
        return self.model_device_emotion2vec_devices

    @property
    def local_model_pool_speech_campplus_devices(self) -> list[str]:
        return self.model_device_speech_campplus_devices

    @property
    def local_model_pool_speech_campplus_instances(self) -> int:
        """CAM++ 模型池副本数；缺省保持单副本。"""
        return len(self.model_device_speech_campplus_devices)

    @property
    def local_model_pool_emotion2vec_instances(self) -> int:
        """emotion2vec 模型池副本数；缺省保持单副本。"""
        return len(self.model_device_emotion2vec_devices)

    # 兼容调用方更短的模型池属性命名。
    @property
    def model_pool_speech_campplus_instances(self) -> int:
        return self.local_model_pool_speech_campplus_instances

    @property
    def model_pool_emotion2vec_instances(self) -> int:
        return self.local_model_pool_emotion2vec_instances

    def _model_pool_instances(self, name: str) -> int:
        """读取旧版数量字段，供旧配置和外部调用兼容。"""
        pool = self._cfg.get("model_device", {}).get("local_model_pool", {})
        if not isinstance(pool, dict):
            raise ValueError("[model_device].local_model_pool 必须是表")
        value = pool.get(name, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"[model_device.local_model_pool].{name} 必须是大于等于 1 的整数"
            )
        return value

    def _model_pool_devices(self, name: str) -> list[str]:
        model_device = self._cfg.get("model_device", {})
        pool = model_device.get("local_model_pool", {})
        if not isinstance(pool, dict):
            raise ValueError("[model_device].local_model_pool 必须是表")

        array_key = f"{name}_devices"
        if array_key in pool:
            value = pool[array_key]
            if not isinstance(value, list):
                raise ValueError(
                    f"[model_device.local_model_pool].{array_key} 必须是 cuda:N 数组"
                )
            devices = []
            for index, item in enumerate(value):
                field_name = f"{array_key}[{index}]"
                if not isinstance(item, str):
                    raise ValueError(
                        f"[model_device.local_model_pool].{field_name} 必须使用 cuda:N 格式"
                    )
                devices.append(self._validate_cuda_device(item, field_name))
            return devices

        # 兼容旧版“单设备 + 副本数”配置；新配置应使用 *_devices 数组。
        device = self._normalize_model_device(name)
        count = self._model_pool_instances(name)
        return [device] * count

    def _normalize_model_device(self, name: str) -> str:
        value = self._cfg.get("model_device", {}).get(name, "cuda:0")
        if isinstance(value, bool):
            raise ValueError(f"[model_device].{name} 必须是 cuda:N 设备串")
        if isinstance(value, (list, tuple, dict)):
            raise ValueError(f"[model_device].{name} 必须是单一 cuda:N 设备串，不支持多卡列表")
        if isinstance(value, int):
            if value < 0:
                raise ValueError(f"[model_device].{name} 的 GPU 编号不能为负数")
            logger.warning(
                "[model_device].%s 使用旧整数索引 %s，请迁移为 cuda:%s",
                name,
                value,
                value,
            )
            return f"cuda:{value}"
        return self._validate_cuda_device(value, name)

    @staticmethod
    def _validate_cuda_device(value, field_name: str) -> str:
        if isinstance(value, bool):
            raise ValueError(f"[model_device.local_model_pool].{field_name} 必须使用 cuda:N 格式")
        if isinstance(value, int):
            if value < 0:
                raise ValueError(f"[model_device.local_model_pool].{field_name} 的 GPU 编号不能为负数")
            return f"cuda:{value}"
        text = str(value).strip().lower()
        if not text.startswith("cuda:"):
            raise ValueError(f"[model_device.local_model_pool].{field_name} 必须使用 cuda:N 格式")
        suffix = text.split(":", 1)[1]
        if not suffix.isdigit():
            raise ValueError(f"[model_device.local_model_pool].{field_name} 必须使用 cuda:N 格式")
        return f"cuda:{int(suffix)}"


settings = Settings()
