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

    @property
    def deployment_profile(self) -> str:
        """NPU 部署标识；设备实际选择仍由各模型配置决定。"""
        value = os.getenv("ASR_DEPLOYMENT_PROFILE")
        if value is None:
            # 兼容旧的外部配置，但新模板不再要求用户填写该内部标记。
            value = self._cfg.get("deployment_profile", "npu-910b")
        if not isinstance(value, str):
            raise ValueError("ASR_DEPLOYMENT_PROFILE 必须是字符串")
        value = value.strip().lower()
        if value and value != "npu-910b":
            raise ValueError("ASR_DEPLOYMENT_PROFILE 只支持 npu-910b；旧 CUDA/TensorRT profile 请迁移")
        return value

    @property
    def is_npu_profile(self) -> bool:
        return self.deployment_profile == "npu-910b"

    @property
    def resource_sample_interval_s(self) -> float:
        value = self._cfg.get("status", {}).get("resource_sample_interval_s", 1.0)
        if type(value) not in (int, float) or not math.isfinite(value) or not .1 <= value <= 60:
            raise ValueError("status.resource_sample_interval_s必须为0.1～60秒有限数")
        return float(value)

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
        return model_paths.get("vad_model_dir", "./model/speech_fsmn_vad_zh-cn-16k-common-pytorch")

    @property
    def spk_model_dir(self) -> Optional[str]:
        model_paths = self._cfg.get("model_paths", {})
        return model_paths.get("spk_model_dir", "./model/speech_campplus_sv_zh_en_16k-common_advanced")

    @property
    def emotion_model_dir(self) -> str:
        model_paths = self._cfg.get("model_paths", {})
        # 注意：原 JSON 中 key 为 emotion_modek_dir（拼写错误），这里修正为 emotion_model_dir
        return model_paths.get("emotion_model_dir", "./model/emotion2vec_plus_large")

    # 语速计算配置 - 从嵌套的 [speech_rate] 部分读取
    @property
    def speech_rate_factor(self) -> float:
        speech_rate = self._cfg.get("speech_rate", {})
        return float(speech_rate.get("rate_factor", 1.0))

    # 功能开关 - 从嵌套的 [features] 部分读取
    @property
    def open_v2_role(self) -> bool:
        """v3 链路的 CAM++ 说话人嵌入开关（配置键保持兼容）。"""
        features = self._cfg.get("features", {})
        return bool(features.get("open_v2_role", True))

    @property
    def open_v2_emotion(self) -> bool:
        """v3 链路的 emotion2vec 情绪开关（配置键保持兼容）。"""
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

    # Qwen3-ASR 适配链路（/v3/seacraft_asr）配置 - 从嵌套的 [qwen_asr] 部分读取
    @property
    def qwen_asr_enabled(self) -> bool:
        return bool(self._cfg.get("qwen_asr", {}).get("enabled", True))

    @property
    def qwen_base_url(self) -> str:
        return str(self._cfg.get("qwen_asr", {}).get("base_url", "")).rstrip("/")

    @property
    def qwen_model(self) -> str:
        # 可选本地模型显示名称；留空按模型目录标识，不发起 HTTP 发现
        return str(self._cfg.get("qwen_asr", {}).get("model", "")).strip()

    def _qwen_positive_int(self, key: str, default: int) -> int:
        value = self._cfg.get("qwen_asr", {}).get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"[qwen_asr].{key} 必须是正整数")
        return value

    @property
    def qwen_execution_mode(self) -> str:
        value = self._cfg.get("qwen_asr", {}).get("execution_mode", "thread")
        if not isinstance(value, str) or value not in {"thread", "process"}:
            raise ValueError("[qwen_asr].execution_mode 必须是 thread 或 process")
        return value

    @property
    def qwen_process_shutdown_grace_s(self) -> float:
        # 只约束显式关闭后的宽限期，不改变单次推理timeout及取消的收尾语义。
        value = self._cfg.get("qwen_asr", {}).get("process_shutdown_grace_s", 30)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 300:
            raise ValueError("[qwen_asr].process_shutdown_grace_s 必须是0~300秒内正有限数")
        return float(value)

    @property
    def qwen_local_model_dir(self) -> str:
        value = self._cfg.get("qwen_asr", {}).get("local_model_dir", "/home/model/Qwen3-ASR-1.7B-hf")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("[qwen_asr].local_model_dir 必须是非空路径")
        return value.strip()

    @property
    def qwen_local_device(self) -> str:
        value = self._cfg.get("qwen_asr", {}).get("local_device", "npu:7")
        return self._validate_device_spec(value, "[qwen_asr].local_device", allow_cpu=False)

    @property
    def qwen_local_dtype(self) -> str:
        value = self._cfg.get("qwen_asr", {}).get("local_dtype", "bfloat16")
        if not isinstance(value, str) or value not in {"bfloat16", "float16", "float32"}:
            raise ValueError("[qwen_asr].local_dtype 必须是 bfloat16、float16 或 float32")
        return value

    @property
    def qwen_local_instance_count(self) -> int:
        return self._qwen_positive_int("local_instance_count", 1)

    @property
    def qwen_max_new_tokens(self) -> int:
        return self._qwen_positive_int("max_new_tokens", 256)

    def _qwen_bounded_int(self, key: str, default: int, lower: int, upper: int) -> int:
        value = self._cfg.get("qwen_asr", {}).get(key, default)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"[qwen_asr].{key} 必须是 {lower}~{upper} 范围内的整数")
        return value

    @property
    def qwen_max_inference_batch_size(self) -> int:
        # 缺省先保留兼容单批；128由独立验证profile请求，交付值以实机结果定案。
        return self._qwen_bounded_int("max_inference_batch_size", 1, 1, 128)

    @property
    def qwen_batch_wait_timeout_ms(self) -> int:
        return self._qwen_bounded_int("batch_wait_timeout_ms", 10, 0, 1000)

    @property
    def qwen_batch_max_duration_ratio(self) -> float:
        value = self._cfg.get("qwen_asr", {}).get("batch_max_duration_ratio", 4.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 64:
            raise ValueError("[qwen_asr].batch_max_duration_ratio 必须是1~64范围内有限数")
        return float(value)

    @property
    def qwen_max_batch_audio_frames(self) -> int:
        """padding后100Hz音频特征帧预算，单位为批成员数乘最长帧数。"""
        return self._qwen_bounded_int("max_batch_audio_frames", 102400, 1, 10000000)

    @property
    def qwen_max_batch_tokens(self) -> int:
        """保守估计批token预算；实际processor输入形状另行观测。"""
        return self._qwen_bounded_int("max_batch_tokens", 78336, 1, 10000000)

    @property
    def qwen_pending_segment_limit(self) -> int:
        return self._qwen_bounded_int("pending_segment_limit", 256, 1, 4096)

    @property
    def qwen_pending_audio_bytes_limit(self) -> int:
        return self._qwen_bounded_int("pending_audio_bytes_limit", 268435456, 1048576, 4294967296)

    @property
    def qwen_segment_concurrency(self) -> int:
        """单音频Qwen逻辑在途窗口；不包含已完成Qwen的增强等待。"""
        return self._qwen_bounded_int("segment_concurrency", 1, 1, 128)

    @property
    def qwen_request_concurrency(self) -> int:
        """全局在途本地转写数；与模型实例执行容量分别计数。"""
        return self._qwen_bounded_int("request_concurrency", 1, 1, 4096)

    @property
    def qwen_audio_concurrency(self) -> int:
        """同时处理的 A+B 音频文件总数上限。"""
        return max(1, int(self._cfg.get("qwen_asr", {}).get("audio_concurrency", 8)))

    @property
    def qwen_audio_spk_emo_concurrency(self) -> int:
        """请求说话人或情绪能力的 B 类整音频并发上限。"""
        value = self._cfg.get("qwen_asr", {}).get("audio_spk_emo_concurrency", 4)
        if type(value) is not int or value <= 0:
            raise ValueError("[qwen_asr].audio_spk_emo_concurrency 必须是正整数")
        return value

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

    def _vad_section(self) -> dict:
        """Return the sole VAD configuration section and flag legacy keys."""
        section = self._cfg.get("vad", {})
        if not isinstance(section, dict):
            raise ValueError("[vad] 必须是 TOML 表")
        legacy = self._cfg.get("qwen_asr", {})
        if isinstance(legacy, dict):
            legacy_keys = {
                "vad_device", "min_segment_duration_s", "vad_max_single_segment_time_ms",
                "vad_max_end_silence_time_ms", "vad_speech_noise_thres", "vad_decibel_thres",
                "vad_snr_thres", "vad_low_volume_rms_dbfs_thres", "vad_low_volume_filter_enabled",
                "vad_decibel_thres_filter_enabled", "vad_snr_thres_filter_enabled",
                "vad_segment_decibel_thres", "vad_decibel_frame_sample_ratio",
                "vad_decibel_low_frame_ratio_thres", "after_start", "before_end",
            }
            found = sorted(legacy_keys.intersection(legacy))
            if found and not getattr(self, "_legacy_vad_warning_emitted", False):
                logger.warning(
                    "[配置迁移] [qwen_asr] 中的 VAD 配置已迁移到 [vad]，旧键仅告警且不会生效: %s",
                    ", ".join(found),
                )
                self._legacy_vad_warning_emitted = True
        return section

    @property
    def qwen_min_segment_duration_s(self) -> float:
        return self.vad_min_segment_duration_s

    @property
    def vad_min_segment_duration_s(self) -> float:
        value = float(self._vad_section().get("min_segment_duration_s", 0.3))
        if not math.isfinite(value) or value < 0:
            raise ValueError("[vad].min_segment_duration_s 必须是非负有限数值")
        return value

    @property
    def qwen_vad_max_single_segment_time_ms(self) -> int:
        return self.vad_max_single_segment_time_ms

    @property
    def vad_max_single_segment_time_ms(self) -> int:
        value = int(self._vad_section().get("vad_max_single_segment_time_ms", 12000))
        if value <= 0:
            raise ValueError("[vad].vad_max_single_segment_time_ms 必须是正整数")
        return value

    @property
    def qwen_vad_max_end_silence_time_ms(self) -> int:
        return self.vad_max_end_silence_time_ms

    @property
    def vad_max_end_silence_time_ms(self) -> int:
        value = int(self._vad_section().get("vad_max_end_silence_time_ms", 800))
        if value < 0:
            raise ValueError("[vad].vad_max_end_silence_time_ms 必须是非负整数")
        return value

    @property
    def qwen_vad_speech_noise_thres(self) -> float:
        return self.vad_speech_noise_thres

    @property
    def vad_speech_noise_thres(self) -> float:
        """FSMN-VAD 的语音/噪声判定阈值（越小越灵敏）。"""
        raw = self._vad_section().get("vad_speech_noise_thres", 0.8)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("[vad].vad_speech_noise_thres 必须是 0.0~1.0 数值") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("[vad].vad_speech_noise_thres 必须在 0.0~1.0 范围内")
        return value

    @staticmethod
    def _validated_vad_float(section: dict, key: str, default: float, lower: float, upper: float) -> float:
        raw = section.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"[vad].{key} 必须是有限数值，范围 {lower}~{upper}") from exc
        if not math.isfinite(value) or value < lower or value > upper:
            raise ValueError(f"[vad].{key} 必须在 {lower}~{upper} 范围内")
        return value

    @property
    def qwen_vad_decibel_thres(self) -> float:
        """FunASR 帧级 log 能量阈值，单位不是 dBFS；-100 表示不额外限制。"""
        return self.vad_decibel_thres

    @property
    def qwen_vad_snr_thres(self) -> float:
        """FunASR 帧级 SNR 阈值；-100 表示不额外限制。"""
        return self.vad_snr_thres

    @property
    def qwen_vad_low_volume_rms_dbfs_thres(self) -> float:
        """段级 RMS dBFS 低音量门限，低于该值才会被过滤。"""
        return self.vad_low_volume_rms_dbfs_thres

    @property
    def qwen_vad_low_volume_filter_enabled(self) -> bool:
        return self.vad_low_volume_filter_enabled

    @property
    def vad_decibel_thres(self) -> float:
        return self._validated_vad_float(self._vad_section(), "vad_decibel_thres", -40.0, -100.0, 0.0)

    @property
    def vad_snr_thres(self) -> float:
        return self._validated_vad_float(self._vad_section(), "vad_snr_thres", 18.0, -100.0, 100.0)

    @property
    def vad_low_volume_rms_dbfs_thres(self) -> float:
        return self._validated_vad_float(
            self._vad_section(), "vad_low_volume_rms_dbfs_thres", -40.0, -100.0, 0.0
        )

    @property
    def vad_low_volume_filter_enabled(self) -> bool:
        raw = self._vad_section().get("vad_low_volume_filter_enabled", True)
        if not isinstance(raw, bool):
            raise ValueError("[vad].vad_low_volume_filter_enabled 必须是 true 或 false")
        return raw

    @property
    def vad_decibel_thres_filter_enabled(self) -> bool:
        raw = self._vad_section().get("vad_decibel_thres_filter_enabled", True)
        if not isinstance(raw, bool):
            raise ValueError("[vad].vad_decibel_thres_filter_enabled 必须是 true 或 false")
        return raw

    @property
    def vad_snr_thres_filter_enabled(self) -> bool:
        raw = self._vad_section().get("vad_snr_thres_filter_enabled", True)
        if not isinstance(raw, bool):
            raise ValueError("[vad].vad_snr_thres_filter_enabled 必须是 true 或 false")
        return raw

    @property
    def vad_segment_decibel_thres(self) -> float:
        return self._validated_vad_float(self._vad_section(), "vad_segment_decibel_thres", -40.0, -120.0, 0.0)

    @property
    def vad_decibel_frame_sample_ratio(self) -> float:
        value = self._validated_vad_float(self._vad_section(), "vad_decibel_frame_sample_ratio", 0.5, 1e-9, 1.0)
        return value

    @property
    def vad_decibel_low_frame_ratio_thres(self) -> float:
        return self._validated_vad_float(self._vad_section(), "vad_decibel_low_frame_ratio_thres", 0.8, 0.0, 1.0)

    @staticmethod
    def _validated_vad_scope_minutes(section: dict, key: str, default: float = 3.0) -> float:
        raw = section.get(key, default)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"[vad].{key} 必须是大于等于 -1 的有限数值") from exc
        if not math.isfinite(value) or value < -1.0:
            raise ValueError(f"[vad].{key} 必须是大于等于 -1 的有限数值")
        return value

    @property
    def vad_after_start(self) -> float:
        return self._validated_vad_scope_minutes(self._vad_section(), "after_start")

    @property
    def vad_before_end(self) -> float:
        return self._validated_vad_scope_minutes(self._vad_section(), "before_end")

    # 直名别名便于标定脚本和运维代码读取，同时保持 qwen_asr 调用方的 Python API 兼容。
    @property
    def vad_device(self) -> list[str]:
        """启动时固定预热的 VAD worker 设备数组。

        每个元素对应一个独立 FSMN-VAD 进程；数组长度就是完整音频的
        VAD 有效并发。旧的 `[qwen_asr]` 配置不会作为回退来源。
        """
        raw = self._vad_section().get("vad_device", ["cpu", "cpu", "npu:7", "npu:7"])
        if not isinstance(raw, list):
            raise ValueError("[vad].vad_device 必须是 cpu/npu:N 设备数组")
        return [self._validate_device_spec(value, f"[vad].vad_device[{i}]")
                for i, value in enumerate(raw)]

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

    def validate_model_backends(self) -> None:
        """只校验配置，硬件及文件检查由启动预检处理，懒加载不会被触发。"""
        _ = self.deployment_profile
        _ = self.resource_sample_interval_s
        _ = self.vad_device
        for name in ("speech_campplus", "emotion2vec"):
            self._model_pool_devices(name)
            self._model_pool_queue_value(name, "prefetch_queue_depth", 0, 0, 64)
            self._model_pool_queue_value(name, "batch_size", 1, 1, 1)
        _ = (self.model_device_speech_campplus_execution_mode,
             self.model_device_speech_campplus_process_startup_timeout_s,
             self.model_device_speech_campplus_process_shutdown_grace_s)
        pool = self._cfg.get("model_device_pool", {})
        obsolete = [key for key in pool if key.endswith(("_use_tensorrt", "_tensorrt_precision"))]
        if obsolete and not getattr(self, "_trt_warning_emitted", False):
            logger.warning("[配置迁移] 已忽略废弃 TensorRT 参数: %s", ", ".join(sorted(obsolete)))
            self._trt_warning_emitted = True
        if self.qwen_asr_enabled:
            _ = (self.qwen_execution_mode, self.qwen_process_shutdown_grace_s,
                 self.qwen_local_model_dir, self.qwen_local_device, self.qwen_local_dtype,
                 self.qwen_local_instance_count, self.qwen_max_new_tokens,
                 self.qwen_segment_concurrency, self.qwen_request_concurrency,
                 self.qwen_max_inference_batch_size, self.qwen_batch_wait_timeout_ms,
                 self.qwen_batch_max_duration_ratio,
                 self.qwen_max_batch_audio_frames, self.qwen_max_batch_tokens,
                 self.qwen_pending_segment_limit, self.qwen_pending_audio_bytes_limit)
        legacy = self._cfg.get("qwen_asr", {})
        if any(legacy.get(k) for k in ("base_url", "connect_timeout_s")) and not getattr(self, "_qwen_http_warning", False):
            logger.warning("[配置迁移] 本地 Qwen 不再使用 base_url/connect_timeout_s")
            self._qwen_http_warning = True

    @property
    def model_device_emotion2vec(self) -> str:
        return self.model_device_emotion2vec_devices[0] if self.model_device_emotion2vec_devices else "cpu"

    @property
    def model_device_speech_campplus(self) -> str:
        return self.model_device_speech_campplus_devices[0] if self.model_device_speech_campplus_devices else "cpu"

    @property
    def model_device_emotion2vec_devices(self) -> list[str]:
        """emotion2vec 模型池实例设备；数组长度即实例数。"""
        return self._model_pool_devices("emotion2vec")

    @property
    def model_device_speech_campplus_devices(self) -> list[str]:
        """CAM++ 模型池实例设备；数组长度即实例数。"""
        return self._model_pool_devices("speech_campplus")

    @property
    def model_device_speech_campplus_execution_mode(self) -> str:
        value = self._cfg.get("model_device_pool", {}).get("speech_campplus_execution_mode", "thread")
        if not isinstance(value, str) or value not in {"thread", "process"}:
            raise ValueError("[model_device_pool].speech_campplus_execution_mode 必须是 thread 或 process")
        if value == "process" and any(
            not device.startswith("npu:") for device in self.model_device_speech_campplus_devices
        ):
            raise ValueError("[model_device_pool].speech_campplus_execution_mode=process 仅支持 NPU 设备数组")
        return value

    def _campplus_process_seconds(self, key: str, default: float) -> float:
        value = self._cfg.get("model_device_pool", {}).get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 300:
            raise ValueError(f"[model_device_pool].{key} 必须是0~300秒内正有限数")
        return float(value)

    @property
    def model_device_speech_campplus_process_startup_timeout_s(self) -> float:
        return self._campplus_process_seconds("speech_campplus_process_startup_timeout_s", 60)

    @property
    def model_device_speech_campplus_process_shutdown_grace_s(self) -> float:
        # 只用于显式关闭；请求取消仍等待已开始的真实模型工作完成。
        return self._campplus_process_seconds("speech_campplus_process_shutdown_grace_s", 30)

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

    def _model_pool_queue_value(self, name, key, default, lower, upper):
        field = f"{name}_{key}"
        value = self._cfg.get("model_device_pool", {}).get(field, default)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"[model_device_pool].{field} 必须是 {lower}~{upper} 范围内的整数")
        return value

    @property
    def speech_campplus_prefetch_queue_depth(self) -> int:
        return self._model_pool_queue_value("speech_campplus", "prefetch_queue_depth", 0, 0, 64)

    @property
    def emotion2vec_prefetch_queue_depth(self) -> int:
        return self._model_pool_queue_value("emotion2vec", "prefetch_queue_depth", 0, 0, 64)

    @property
    def speech_campplus_batch_size(self) -> int:
        return self._model_pool_queue_value("speech_campplus", "batch_size", 1, 1, 1)

    @property
    def emotion2vec_batch_size(self) -> int:
        return self._model_pool_queue_value("emotion2vec", "batch_size", 1, 1, 1)

    def _model_pool_devices(self, name: str) -> list[str]:
        pool = self._cfg.get("model_device_pool", {})
        if not isinstance(pool, dict):
            raise ValueError("[model_device_pool] 必须是表")
        array_key = f"{name}_devices"
        value = pool.get(array_key, [])
        if not isinstance(value, list):
            raise ValueError(f"[model_device_pool].{array_key} 必须是设备数组")
        devices = []
        for index, item in enumerate(value):
            field_name = f"{array_key}[{index}]"
            if not isinstance(item, str):
                raise ValueError(f"[model_device_pool].{field_name} 必须是 cpu 或 npu:N")
            devices.append(self._validate_model_device(item, field_name))
        if "model_device" in self._cfg:
            logger.warning("[配置迁移] 已忽略旧 [model_device]，请迁移到 [model_device_pool]")
        return devices

    @staticmethod
    def _validate_device_spec(value, field_path: str, *, allow_cpu: bool = True) -> str:
        allowed = "cpu 或 npu:N" if allow_cpu else "npu:N"
        if not isinstance(value, str):
            raise ValueError(f"{field_path} 必须使用 {allowed} 格式")
        text = value.strip().lower()
        if allow_cpu and text == "cpu":
            return text
        if text.startswith("cuda"):
            raise ValueError(f"{field_path} 不再支持 CUDA，请迁移为 {allowed}")
        if not text.startswith("npu:") or not text.split(":", 1)[1].isdigit():
            raise ValueError(f"{field_path} 必须使用 {allowed} 格式")
        return f"npu:{int(text.split(':', 1)[1])}"

    @staticmethod
    def _validate_model_device(value, field_name: str) -> str:
        return Settings._validate_device_spec(value, f"[model_device_pool].{field_name}")


settings = Settings()
