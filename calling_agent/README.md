# Czech mammography calling agent

A runnable outreach stack: **GPT-Live (`gpt-live-1`)** for Czech speech, a
**`gpt-6-sol` Responses backend** for center search and scheduling tools,
**FastAPI**, **SQLite**, **WebRTC** for browser voice testing, and **direct SIP**
for outbound phone calls. Model IDs and voice are configurable. This follows
the current official API documentation; Czech pronunciation and conversational
quality still require live evaluation on your account and telephone network.

The conversation starts with the missed mammography appointment and an offer to
help arrange it or answer questions. Klára speaks in short, everyday Czech,
leaves room for replies, and is instructed to stop immediately when interrupted.
She follows the caller's questions; there is no routine questionnaire about
non-attendance, residence, transport or medical history. She asks whether morning or afternoon suits the caller,
offers two dated **simulated** slots, reads back the chosen center and time, and
saves a **demo booking confirmation** after explicit approval. Real center
information comes from mamo.cz; slots are invented by a local demo calendar.
The spoken conversation is an immersive demonstration: opening, offers and
confirmation use ordinary appointment language without demo disclaimers. The
operator UI and saved records still clearly mark all bookings as simulated. A
direct question about a real reservation receives a truthful answer. Nothing contacts
centers or sends SMS/email.

The browser plays a cached Czech greeting automatically when WebRTC connects;
it does not depend on the live model deciding to speak first. A caller's first
transcript fragment interrupts the clip. The operator can select Coral, Marin, Shimmer, Sage, Alloy or Cedar and a
positive, gentle, calm or businesslike tone. The choice is remembered in the
browser and saved per call; both the opening and live conversation use it.
Selections lock during a call and apply to the next one. Defaults are
`VOICE=coral` and `TONE=positive`. Greeting clips are cached separately for each
voice, tone, text and speed, so a late preload cannot replace the selected voice. The
shortened opening is rendered at 1.15× speed (`GREETING_SPEED=1.15`) once with `gpt-4o-mini-tts` and cached locally, including
across server restarts. Preparation starts when the console opens; the first render
may take several seconds. Browser autoplay restrictions are surfaced as errors.

Backend reasoning defaults to `none` and answers are capped at 700 output tokens.
Simple questions and choices use the voice model directly rather than requiring
a backend call. These changes target latency; response time is not benchmarked.

## Run locally

Use Python 3.12 and the repository's single root `uv` project. These commands
install only the calling stack, without downloading the ML training stack:

```sh
uv sync --only-group calling
cp -n calling_agent/.env.example .env
```

The local console opens directly without an app token or login. Configure a genuine
`ORGANIZATION_NAME` before real calls; the default is explicitly a demonstration.
No credentials are bundled or required for the synthetic walkthrough.

```sh
uv run --only-group calling python -m calling_agent.centers
uv run --only-group calling uvicorn calling_agent.main:app --host 127.0.0.1 --port 8080 --workers 1 --no-access-log
```

Open **http://localhost:8080**, then select
**Spustit syntetickou ukázku**. This exercises center search, afternoon demo slots and a
simulated booking confirmation with clearly fictional answers.
It makes no model or telephone calls. Only the public center directory needs
an initial download; subsequent synthetic runs work offline.

For real Czech speech, set `OPENAI_API_KEY` in `.env`, restart the server, and
select **Vyzkoušet hlasem**. Allow microphone access. Microphone input and speaker
output go directly over WebRTC; API credentials stay on the server. The backend
attaches immediately to execute tools and observe session events. HTTPS or
localhost is required by browsers. An API key alone does not establish access
to GPT-Live or the configured backend model; inaccessible models produce an
error rather than silently switching models. Close other tabs to avoid echo.

## Outbound phone calling

For an existing FaceTime Audio call on your Mac, use the separate
[FaceTime audio-routing setup](FACETIME.md). It uses two virtual audio devices
and requires no SIP account. The browser now lets you select the agent's audio
input explicitly. FaceTime dialing and hangup remain manual.

Use a SIP trunk supporting TLS, Opus, and SDES-SRTP, and enable outbound SIP
calling for your OpenAI organization. This uses the **Live API**, not the
Realtime WebRTC call-creation endpoint. Enter in `.env`:

```dotenv
ENABLE_OUTBOUND_CALLS=true
OUTBOUND_ALLOWLIST=+420YOUR_TEST_NUMBER
SIP_PROVIDER_URL=sips:YOUR_PROVIDER:5061
SIP_USERNAME=YOUR_USERNAME
SIP_PASSWORD=YOUR_PASSWORD
SIP_CALLER_NUMBER=+420YOUR_AUTHORIZED_CALLER_ID
```

Replace every placeholder; the example is deliberately not a dialable number.
Restart, expand **Odchozí telefonní hovor**, enter the explicitly permitted test
number, confirm contact permission, and click **Zavolat**. Each click requires a
fresh local session. There is no campaign, bulk dialer, or autonomous retry.
Do-not-contact suppression is checked across sessions. The phone number is
passed to the provider and held transiently during dialing; the local database
stores a keyed hash rather than the destination number. Preserve `phone-hash.key`
with the database when moving or restoring data; it is independent of application credentials.

Session creation can succeed without the callee answering. The console follows
ringing, answer, failure, close, and tool events through the sideband socket.
An ambiguous timeout is marked `unknown` and is never automatically retried.
Check the provider's call state before starting another session. After verifying
that an unknown/interrupted call has ended, use the console's explicit operator
confirmation to resolve its local record. **Ukončit hovor**
requests remote hangup; a failed hangup stays unknown. Process restarts mark
previous active calls interrupted. They are not resumed or redialed. The call
has a server-enforced duration limit (default ten minutes). Use one application
worker; the sideband connections and captions belong to that process.

## Twilio setup for the existing direct-SIP adapter

Two account capabilities are prerequisites, not just configuration fields:
OpenAI must enable outbound SIP for your organization, and Twilio must enable
the **Opus Limited Availability codec** for Elastic SIP Trunking. Twilio's
default trunk codecs are PCMU/PCMA, whereas the current OpenAI outbound SIP
interface requires Opus. Ask Twilio Sales/account support for Opus access before
assuming that an ordinary trunk will work. See [Twilio codecs](https://www.twilio.com/docs/sip-trunking/codecs)
and [OpenAI outbound SIP](https://developers.openai.com/api/docs/guides/voice-sip).
Twilio currently documents SIP trunking as requiring an upgraded account; see
[account limits](https://www.twilio.com/docs/sip-trunking/scale-and-limits).

1. In Twilio Console, open **Elastic SIP Trunking → Manage → Trunks**, create
   a trunk, and give it a recognizable name such as `mamo-agent`.
2. In **Termination**, choose a unique Termination SIP URI such as
   `YOUR_UNIQUE_TRUNK.pstn.twilio.com`. Attach a **Credential List** with a
   dedicated SIP username and strong password. These are SIP digest credentials,
   not your Twilio Account SID/Auth Token. If an IP ACL is also configured, both
   authentication checks apply; the SIP source here is OpenAI, not your laptop.
3. Enable **Secure Trunking** (TLS signaling and SRTP media). Confirm with Twilio
   that Opus is enabled for this account and outbound termination path.
4. Use a Twilio-owned phone number or a Twilio-verified caller ID for the
   `SIP_CALLER_NUMBER`. Under **Voice → Settings → Geo permissions**, enable the
   Czech Republic destination range needed for your test number.
5. Merge the following into the repository-root `.env`, preserving your
   `OPENAI_API_KEY`:

   ```dotenv
   ENABLE_OUTBOUND_CALLS=true
   OUTBOUND_ALLOWLIST=+420YOUR_OWN_TEST_NUMBER
   SIP_PROVIDER_URL=sips:YOUR_UNIQUE_TRUNK.pstn.twilio.com:5061
   SIP_USERNAME=YOUR_CREDENTIAL_LIST_USERNAME
   SIP_PASSWORD=YOUR_CREDENTIAL_LIST_PASSWORD
   SIP_CALLER_NUMBER=+YOUR_TWILIO_OR_VERIFIED_NUMBER
   ```

   Replace all placeholders with real values. No `TWILIO_ACCOUNT_SID` or
   `TWILIO_AUTH_TOKEN` is used by this adapter. No incoming-call webhook, ngrok,
   or inbound origination URL is required for this outbound-only path.
6. Stop the existing preview process before starting the configured server.
   The preview was deliberately launched with an empty `OPENAI_API_KEY`
   environment variable, which overrides `.env`; do not keep that override.
   Run the normal local command above, open the console,
   expand **Odchozí telefonní hovor**, confirm contact permission, and call only
   the allowlisted test number. Creating/configuring the trunk alone makes no call.

Check the application's error and Twilio's trunk call logs for failures. A
`403 outbound_sip_not_enabled` is an OpenAI account capability issue. Codec
negotiation failures require checking Opus on the trunk. Also verify the digest
credentials, caller ID, and Czech destination permission. This integration has
not been tested with a live Twilio account.

If Opus cannot be enabled, the existing direct-SIP adapter cannot be used with
that trunk. Twilio Agent Connect or a Programmable Voice audio bridge are
alternative integration paths; either requires a transport adapter change,
not merely different environment variables. Keep the Czech interview and
booking-request tools unchanged when adding that adapter.

Sources: [Twilio trunk setup](https://www.twilio.com/docs/sip-trunking),
[geographic permissions](https://www.twilio.com/docs/sip-trunking/voice-dialing-geographic-permissions),
[OpenAI partner integration paths](https://developers.openai.com/api/docs/guides/live-partner-integrations).

## Directory and data

`python -m calling_agent.centers` imports the public embedded center records from
[mamo.cz](https://www.mamo.cz/centra/), using the same upstream format as
`features/spatial/01_scrape_centers.py`. It stores names, addresses, regions,
GPS, public contacts, opening hours, online-booking links when provided,
published wait estimates and their dates, source links, fetch time, and the
source HTML SHA-256. Refreshes validate the complete directory and replace the
snapshot atomically; a changed/incomplete source cannot overwrite a good one.
The initial live import contained **75 centers on 17 October 2026**.

The agent searches without accents by city, region or name and offers up to six
matches to the backend. The spoken prompt limits suggestions to two or three.
If the query resolves to one Czech locality, results rank by approximate
straight-line distance from that locality's representative point to each center;
three nearby centers are returned. This is not a home-address distance, route
length or travel time. Multiple places sharing a name require clarification by
district. Unknown locations and explicit center names retain text search and
never receive invented distances. A stated preferred center takes priority.

The local place index contains **16,625 Czech populated places** from
[GeoNames](https://www.geonames.org/) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
It works offline and sends no caller locations to a geocoder. To reproduce it:

```sh
curl --fail --location --output data/calling_agent/geonames-CZ.zip https://download.geonames.org/export/dump/CZ.zip
uv run --only-group calling python -m calling_agent.places data/calling_agent/geonames-CZ.zip
```

The importer validates the country extract and replaces the index atomically.
If the index is missing, only text search is available. Published waiting days
are not bookable availability. The directory's booking links are not followed.

Local files live in gitignored `data/calling_agent/`:

- `centers.json`: public directory snapshot, reproducible by the importer.
- `places.json`, `geonames-CZ.zip`: public offline place index and source archive.
- `greeting-*.mp3`: generated public opening, never caller audio.
- `calls.sqlite3`: consent, caller-reported answers, selection, demo booking, legacy preference request,
  tool deduplication results, and call lifecycle. Sensitive local data.
- `phone-hash.key`: private suppression hashing key; filesystem mode 0600.

No caller audio or full transcript is written by this application. Live captions are
bounded in memory; they disappear at server restart or record deletion. The
voice session sets `store: false`, which disables the optional Live session
recording; it is not a claim about all provider-side retention. Consent
withdrawal removes the local interview and prior tool-result cache. Record
deletion removes the interview and its cache; an opt-out suppression hash is
retained so deletion cannot cause another call.

The console needs no app token or login and binds to localhost by default.
API routes reject cross-site browser requests and unexpected Host headers. JSON exports contain interview answers, so
keep them private. It deliberately does not load claim records, patient IDs,
screening predictions, or a contact list. Before wider deployment, replace the
local access model with individual access controls and configure retention,
provider data handling, organizational contact permission, and a staffed
follow-up process. This local stack is not a clinically validated intervention.

## Conversation behavior and boundaries

Following the [GPT-Live guide](https://developers.openai.com/api/docs/guides/live)
and [prompting guidance](https://developers.openai.com/api/docs/guides/live-prompting),
the voice prompt contains conversation style, interruption handling and clear
backend delegation conditions. The backend prompt owns detailed booking rules,
examples and decision procedures. Verified general FAQ information stays with
the voice model so ordinary questions need no backend round trip.

The static conversation prompt was reduced from 1,228 to 498 words in this
revision (before shared style and FAQ expansion). Model-facing tool results
retain required IDs, sources, contacts, date/time labels and distance evidence,
while omitting duplicate slots and internal notices. On the local Brno example,
center-result JSON dropped from 2,637 to 2,034 UTF-8 bytes, and availability from
2,257 to 941 bytes. These are payload measurements, not audio latency or quality
benchmarks. The directory, SQLite records and public API retain their full data.

The voice continues to use native WebRTC/full duplex and can answer a known
question while backend work is pending. Listening is silent, including at pauses;
background sounds are not new requests. An interruption of speech does not by
itself cancel a booking operation. A changed center/date must be delegated again
and older results must not be presented as the revised selection.

A short [listening checklist](VOICE_EVAL.md) covers the user's reported examples.

The AI briefly identifies itself, then explains the reason for calling and
listens. It uses natural Czech, one question at a time and usually one or two
short sentences. Explicit GPT-Live interruption and backchannel policies ask
it to yield immediately, listen silently without spoken filler (including
acknowledgments that may be absent from captions), and answer the
caller's latest question after an interruption. These are model instructions;
interruption timing still needs live evaluation through the chosen audio route.

Public center searches and demo slot generation do not require introductory
consent tools. Klára resolves one center at a time, uses all details already
provided, and offers two times with the date stated once. A caller's explicit
choice of an offered center and slot is itself confirmation; no separate
permission-to-note or repeated yes is required. An ambiguous yes still needs
clarification. Confirmation runs before the one-sentence spoken summary, which
includes the center, exact date and time.

`get_demo_slots` supports both flexible earliest dates (`not_before`) and an
exact requested day (`on_date`). Past or weekend exact dates return no slots;
they are not silently changed to another day. Results include two recommended
choices and Czech spoken date labels, so the voice need not calculate weekdays.
`confirm_demo_booking` accepts only a future slot returned for an offered center.
The saved record uses `status: demo_confirmed`, `booking_status: simulated` and
`simulated: true`. Weekday filtering and time offsets use Europe/Prague. These
slots are invented; holidays, opening hours and real availability are not checked.

No reason or residence questionnaire is required. The older interview and
manual-request tools remain for compatibility but are not exposed to live voice.

If the caller is hesitant, Klára offers one brief explanation with permission,
matched to the stated concern. It covers early detection, symptoms, discomfort
and limitations without promises, guilt or pressure. A clear refusal or request
to end the conversation takes precedence. Mentioned locations trigger nearby
center suggestions; no extra residence questionnaire is required.

Brief answers about referral, preparation and the examination come from
[mamo.cz guidance](https://www.mamo.cz/cs/o-programu/jak-se-objednat/), checked on
17 October 2026, in `knowledge.py`, together with the official
[program description](https://www.mamo.cz/cs/o-programu/popis-programu/) and
[FAQ](https://www.mamo.cz/cs/o-programu/faq/). Klára does not diagnose or decide clinical
eligibility. New breast symptoms or a request for a person create a
human-follow-up flag and block a routine booking request. Live transfer and
follow-up dispatch are not implemented. Opt-out blocks further collection and
future phone calls. Quoted confirmation comes from the model; understanding
consent and identity still needs conversational evaluation.

Prompts cover wrong recipients, corrections such as already having attended,
and uncertainty. Answering-machine detection and formal patient identity
verification are not implemented. For the first trial, call only your own
allowlisted number. Request completion does not itself disconnect the phone:
the participant, operator, or maximum-duration watchdog ends it.

## Docker

With root `.env` populated:

```sh
docker compose -f calling_agent/compose.yaml build
docker compose -f calling_agent/compose.yaml run --rm mamo python -m calling_agent.centers
docker compose -f calling_agent/compose.yaml up -d
```

The service runs as a non-root user, publishes only to `127.0.0.1`, uses a
read-only container filesystem and a named volume for the directory and private
state. The root build context excludes research datasets and credentials.
Refresh the directory with the same `compose run` command. When putting it
behind HTTPS, set `PUBLIC_ORIGIN` to the exact operator-console origin.
Docker manifests are provided; a container build was not verified here.

## Validation

```sh
uv sync --only-group calling --only-group dev
uv run --only-group calling --only-group dev python -m unittest discover -s calling_agent/tests -v
uv run --only-group calling --only-group dev ruff check calling_agent
node --check calling_agent/static/app.js
node --test calling_agent/tests/test_greeting.mjs
```

Tests use synthetic patient answers and mock provider calls. They exercise
consent, withdrawal, suppression, clinical handoff, incomplete/changed requests,
center validation, tool replay, nested Live function-call continuation, cancelled
responses, ambiguous timeouts, duplicate dialing, token-free local access and origin
checks, deletion and the offline walkthrough. They do not establish live API
access, Czech audio quality, SIP interoperability, or patient outcomes.

## Current protocol references

- [Automatic greeting and playback](https://developers.openai.com/api/docs/guides/live-conversations#greet-before-the-caller-speaks)
- [GPT-Live architecture](https://developers.openai.com/api/docs/guides/live)
- [WebRTC session creation](https://developers.openai.com/api/docs/guides/voice-webrtc)
- [Responses delegation and tool results](https://developers.openai.com/api/docs/guides/live-delegation)
- [SIP outbound calls](https://developers.openai.com/api/docs/guides/voice-sip)
- [Sideband controls](https://developers.openai.com/api/docs/guides/voice-server-controls)

API integration points: `live.py` owns session creation and sideband lifecycle;
`workflow.py` owns booking-request rules; `prompts.py` holds Czech prompts;
`centers.py` owns directory provenance; `main.py` exposes the operator API.
