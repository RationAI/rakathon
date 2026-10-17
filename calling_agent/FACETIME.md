# FaceTime Audio on this Mac

The caller uses FaceTime Audio normally. The local browser agent uses OpenAI's
voice API. No SIP trunk or WhatsApp Business account is involved. Dialing and
ending the FaceTime call remain manual.

Two Loopback devices connect the apps:

| Device | Application source | Selected as microphone in |
| --- | --- | --- |
| FaceTime to Agent | Phone (the active FaceTime Audio call on this Mac) | Local agent browser page |
| Agent to FaceTime | ChatGPT (which hosts this task's in-app browser) | Phone's Audio menu |

On macOS versions where the FaceTime app owns the call audio, use FaceTime as
the first source instead of Phone. When opening the agent in Safari or Chrome,
change the second source to that browser. Do not include other apps or your
physical microphone in these routes. Each device has only its application
source, wired to output channels 1 and 2; the default Pass-Thru source is removed
to avoid an unintended feedback path. “Mute when capturing” is disabled so the
operator can monitor both apps. Headphones are useful if acoustic feedback occurs.

## Start a test

1. Start the FaceTime Audio call and select **Agent to FaceTime** as its
   microphone in Phone > Audio (or FaceTime > Video). A running call can cache the
   device's original name, **Loopback Audio**, until the call app refreshes its
   device list. That was the original name of the return device in this setup.
2. Open http://localhost:8080; the calling controls load automatically without a token.
3. Click **Načíst zvuková zařízení** and permit microphone access.
4. Select **FaceTime to Agent (Virtual)**, enable **Zvuk z FaceTime / virtuálního
   zařízení**, then click **Vyzkoušet hlasem**. The virtual-input mode disables
   microphone noise suppression, gain adjustment, and echo cancellation so they
   do not alter the already-routed call audio.
5. Confirm the remote participant hears the Czech greeting and that their reply
   appears under **Volající** in the transcript.

**Ukončit asistentku** stops only the AI connection, not the FaceTime call.
To speak directly to the participant again, restore the call's microphone to
**Use System Setting** or **MacBook Pro Microphone**. Loopback devices persist
until you disable/delete them; the permissions screen can be reopened in
Loopback. Loopback is installed in trial mode; no license was purchased.

The actual trial found and fixed an SDP validation bug: audio offers must retain
their final CRLF. The regression test protects that exact behavior. A successful
audio connection does not, by itself, establish that backend tools completed;
check the consent and saved-request fields as well as the transcript.

References: [FaceTime audio device selection](https://support.apple.com/en-gb/guide/facetime/fctm26739220/mac),
[Loopback voice-call routing](https://rogueamoeba.com/support/knowledgebase/?showArticle=LB-VoIP).
