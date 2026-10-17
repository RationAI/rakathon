# Czech voice listening checks

Use a fresh session after prompt changes, on your own test device. Keep the same
voice, tone, FaceTime routing and scenario when comparing revisions. These checks
are manual; automated tool tests do not establish speech quality.

| Scenario | Caller input | What to listen for |
| --- | --- | --- |
| Silent start | Say nothing after connecting. | One automatic Czech opening, selected voice/tone, no spoken demo notice. |
| Interruption | Interrupt the opening, then a later answer with “Moment, chci se zeptat…” | Stops speaking, listens to the new question and does not restart the introduction. |
| Thinking pause | “Chtěla bych…”, pause briefly, then finish the sentence. | Gives room to think; no automatic “Jasně”, “hm” or competing question. |
| Nearby center | “Jsem z Kuřimi.” | Two sourced nearby center suggestions, no repeated request for the town or invented driving time. |
| Precise choice | Choose a center and morning; select an offered time with “V půl jedenácté v tom prvním.” | One backend confirmation, one complete spoken center/date/time, no second permission-to-note question. |
| Date repair | Mention a past or ambiguous date, then clarify the month once. | One focused clarification, no lecture about the calendar or repeated correction. |
| Question during lookup | While centers are loading: “Potřebuji žádanku?” | Answers the known FAQ and continues with the lookup result when available. |
| Correction during lookup | Change the requested center or date while the backend works. | Uses the correction and does not treat an older result as the current choice. |
| Hesitation/refusal | “Nic mě nebolí”; then, separately, “Nechci to řešit.” | Offers one brief explanation with permission; respects a clear refusal without persuasion. |

Rate naturalness, Czech pronunciation, clarity of times, voice consistency and
unwanted filler. Measure start-to-first-audible-word and caller-end-to-useful-reply
with audio playback, separately from caption arrival and backend completion.
Also check that the saved slot matches the spoken confirmation. Note interruption
recovery separately: a completed tool action and an interrupted spoken result
are different outcomes.

References: [GPT-Live](https://developers.openai.com/api/docs/guides/live),
[prompting](https://developers.openai.com/api/docs/guides/live-prompting),
[delegation](https://developers.openai.com/api/docs/guides/live-delegation).
