from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from calling_agent.speech import Tone, Voice


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class NewSession(StrictModel):
    mode: Literal["browser", "phone", "demo"] = "browser"
    contact_label: str = Field(default="", max_length=80)
    voice: Voice | None = None
    tone: Tone | None = None


class Offer(StrictModel):
    # SDP uses CRLF record terminators, including the final one. Stripping the
    # offer corrupts it and makes the upstream parser fail with EOF.
    model_config = ConfigDict(str_strip_whitespace=False)
    sdp: str = Field(min_length=10, max_length=65000)


class Dial(StrictModel):
    phone: str = Field(pattern=r"^\+[1-9]\d{7,14}$")
    contact_permission_confirmed: Literal[True]


class ResolveCall(StrictModel):
    remote_ended_confirmed: Literal[True]


class Consent(StrictModel):
    consent: Literal["granted", "declined", "withdrawn"]
    user_quote: str = Field(min_length=1, max_length=500)


class Interview(StrictModel):
    attendance: Literal[
        "unknown", "not_attended", "already_attended", "declined_to_answer"
    ]
    reason_category: Literal[
        "not_asked",
        "time",
        "transport",
        "fear",
        "forgot",
        "no_invitation",
        "already_attended",
        "health",
        "other",
        "declined_to_answer",
    ]
    reason_quote: str = Field(max_length=1000)
    municipality: str = Field(max_length=120)
    travel_preferences: str = Field(max_length=500)


class SearchCenters(StrictModel):
    query: str = Field(min_length=2, max_length=150)


class SelectCenter(StrictModel):
    center_id: int
    user_quote: str = Field(min_length=1, max_length=500)


class Preferences(StrictModel):
    preferred_dates: str = Field(min_length=1, max_length=500)
    preferred_times: str = Field(min_length=1, max_length=500)
    accessibility_needs: str = Field(max_length=500)


class ConfirmRequest(StrictModel):
    user_confirmed: Literal[True]
    confirmation_quote: str = Field(min_length=1, max_length=500)


class SchedulingRequest(StrictModel):
    center_id: int
    preferred_dates: str = Field(min_length=1, max_length=500)
    preferred_times: str = Field(min_length=1, max_length=500)
    accessibility_needs: str = Field(max_length=500)
    permission_to_save: Literal[True]
    confirmation_quote: str = Field(min_length=1, max_length=500)


class DemoSlots(StrictModel):
    center_id: int
    day_period: Literal["morning", "afternoon", "either"]
    not_before: str = Field(
        description="Requested earliest date as YYYY-MM-DD, or empty if unspecified.",
        max_length=10,
    )
    on_date: str = Field(
        description="Exact requested date as YYYY-MM-DD, or empty for a flexible period.",
        max_length=10,
    )


class DemoBooking(StrictModel):
    slot_id: str = Field(min_length=1, max_length=100)
    user_confirmed: Literal[True]
    confirmation_quote: str = Field(min_length=1, max_length=500)


class Handoff(StrictModel):
    reason: Literal["symptoms", "human_requested", "uncertain_identity", "other"]
    note: str = Field(max_length=500)


class OptOut(StrictModel):
    user_quote: str = Field(min_length=1, max_length=500)


class Center(StrictModel):
    id: int
    name: str
    city: str
    address: str
    postal_code: str
    region: str
    phone: list[str]
    email: str
    website: str
    online_booking_url: str
    opening_hours: str
    reported_wait_days: int | None
    wait_updated: str | None
    latitude: float
    longitude: float
    source_url: str

    @model_validator(mode="after")
    def valid_coordinates(self):
        if not (48 <= self.latitude <= 52 and 12 <= self.longitude <= 19):
            raise ValueError("Center coordinates outside the Czech Republic")
        return self
