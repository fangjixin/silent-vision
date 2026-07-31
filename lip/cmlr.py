from backend.config import Settings
from lip.mpc001 import MPC001LipReader


class CMLRLipReader(MPC001LipReader):
    """Chinese visual speech reader backed by mpc001 CMLR visual-only assets."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings,
            model_name="cmlr",
            language="zh",
            config_path=settings.mpc001_chinese_config_path,
        )
