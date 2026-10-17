"""Shared voice choices and delivery presets for the opening and live model."""

from typing import Literal


Voice = Literal["coral", "marin", "shimmer", "sage", "alloy", "cedar"]
Tone = Literal["positive", "gentle", "calm", "business"]

VOICES = [
    {"id": voice, "label": voice.capitalize()}
    for voice in ("coral", "marin", "shimmer", "sage", "alloy", "cedar")
]
TONES = [
    {"id": "positive", "label": "Pozitivní a přátelský"},
    {"id": "gentle", "label": "Jemný a vřelý"},
    {"id": "calm", "label": "Klidný a podpůrný"},
    {"id": "business", "label": "Věcný a stručný"},
]

BASE_STYLE = (
    "Mluv přirozenou češtinou a vykáním, jako vstřícná asistentka při běžném telefonátu. "
    "Používej přirozenou melodii a běžnou hlasitost. Mluv plynule bez natahování slov "
    "a dlouhých pauz, zejména v úvodu; dej prostor odpovědi a neuspěchej volající. "
    "Bez úředního tónu, šeptání, hraného smíchu a automatických souhlasných vsuvek. "
)
TONE_STYLES = {
    "positive": "Zněj mile, pozitivně a zaujatě, s jasnějším, lehce usměvavým hlasem a svižným přirozeným tempem. Bez přehnaného nadšení.",
    "gentle": "Zněj měkce, vřele a jemně. Zjemni nástupy slov, zachovej živou melodii a plynulé běžné tempo, ne šeptání ani smutné zabarvení.",
    "calm": "Zněj klidně, vyrovnaně a podpůrně, s rovnoměrným běžným tempem. Dej čas na odpověď, ale nedělej dlouhé pauzy uvnitř vět.",
    "business": "Zněj věcně, mile a sebejistě. Používej svižné tempo a přímé krátké věty bez přehnaných emocí.",
}


def speech_style(tone: Tone = "positive") -> str:
    return BASE_STYLE + TONE_STYLES[tone]
