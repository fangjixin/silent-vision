from backend.config import Settings
from lip.mpc001 import MPC001LipReader


class AVHuBERTLipReader(MPC001LipReader):
    """English visual speech reader backed by mpc001 LRS3 visual-only Auto-AVSR assets.

    The public candidate label remains "avhubert" for compatibility with the
    existing WebSocket schema. The implementation no longer expects a local
    TorchScript AV-HuBERT file.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings,
            model_name="avhubert",
            language="en",
            config_path=settings.mpc001_english_config_path,
        )
