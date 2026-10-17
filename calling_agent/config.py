from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from calling_agent.speech import Tone, Voice


ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    openai_api_key: SecretStr = SecretStr("")
    public_origin: str = "http://localhost:8080"
    voice_model: str = "gpt-live-1"
    backend_model: str = "gpt-6-sol"
    backend_reasoning_effort: str = "none"
    greeting_model: str = "gpt-4o-mini-tts"
    voice: Voice = "coral"
    tone: Tone = "positive"
    greeting_speed: float = Field(default=1.15, ge=0.25, le=4.0)
    organization_name: str = "Ukázkový program prevence"
    data_dir: Path = ROOT / "data" / "calling_agent"
    enable_outbound_calls: bool = False
    outbound_allowlist: str = ""
    sip_provider_url: str = ""
    sip_username: str = ""
    sip_password: SecretStr = SecretStr("")
    sip_caller_number: str = ""
    max_call_seconds: int = 600

    @property
    def allowed_numbers(self) -> set[str]:
        return {
            number.strip()
            for number in self.outbound_allowlist.split(",")
            if number.strip()
        }
