"""Renders the video scenes: every frame is a screenshot of the scene's render(t), the soundtrack is synthesized
from the sound cues the scene lists (window.CUES), and ffmpeg (bundled with imageio-ffmpeg) joins them.

Run capture.py first (screens/, used by pozvanka.html). Then, from this folder:
    uv run --no-project --with playwright --with imageio-ffmpeg --with numpy --with scipy python render.py
Outputs: the VIDEOS below, 30 fps, Czech.
"""

import asyncio
import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from playwright.async_api import async_playwright
from scipy.signal import butter, sosfilt


VIDEOS = [  # scene -> output; comment out the ones that need no rebuild
    ("pozvanka.html", "pozvanka_mamografie.mp4"),  # 1920 x 1080: the planner, Jana and Marie
    # 1080 x 1350, shown in the webapp (with a poster image next to them): how the examination goes, survival by stage
    ("vysetreni.html", "../../webapp/video/jak_probiha_mamografie.mp4"),
    ("preziti.html", "../../webapp/video/cim_driv_tim_lip.mp4"),
]
FPS = 30
WORKERS = 6
SR = 48_000


# ---------------------------------------------------------------- frames


async def frames(scene: str, folder: Path) -> tuple[float, list, dict]:
    url = Path(scene).resolve().as_uri()
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome")
        pages = []
        for _ in range(WORKERS):
            page = await browser.new_page(viewport={"width": 1920, "height": 1080})
            await page.goto(url)
            await page.evaluate("window.ready")
            w, h = await page.evaluate("window.SIZE")
            await page.set_viewport_size({"width": w, "height": h})
            pages.append(page)
        duration = await pages[0].evaluate("window.DURATION")
        cues = await pages[0].evaluate("window.CUES")
        music_cfg = await pages[0].evaluate("window.MUSIC || {}")
        music_cfg["poster"] = await pages[0].evaluate("window.POSTER ?? null")
        n = round(duration * FPS)

        async def work(k: int) -> None:
            for i in range(k, n, WORKERS):
                await pages[k].evaluate(f"render({i / FPS})")
                await pages[k].screenshot(path=folder / f"{i:05d}.png")
                if i % 100 == 0:
                    print(f"frame {i}/{n}", flush=True)

        await asyncio.gather(*(work(k) for k in range(WORKERS)))
        await browser.close()
    return duration, cues, music_cfg


# ---------------------------------------------------------------- sound


def t_axis(dur: float) -> np.ndarray:
    return np.arange(int(dur * SR)) / SR


def bell(f: float, dur: float = 0.9, tau: float = 0.35, partials=((1, 1), (2.0, 0.35), (3.01, 0.12))) -> np.ndarray:
    t = t_axis(dur)
    attack = np.minimum(1, t / 0.004)
    return attack * np.exp(-t / tau) * sum(a * np.sin(2 * np.pi * f * m * t) for m, a in partials)


def noise(dur: float, lo: float, hi: float, seed: int = 0) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal(int(dur * SR))
    return sosfilt(butter(2, [lo, hi], btype="band", fs=SR, output="sos"), x)


def sweep(f0: float, f1: float, dur: float, tau: float) -> np.ndarray:
    t = t_axis(dur)
    f = np.geomspace(f0, f1, len(t))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / tau)


def whoosh(dur: float, seed: int = 0) -> np.ndarray:
    x = noise(dur, 250, 2500, seed)
    t = np.linspace(0, 1, len(x))
    return 0.6 * x / np.abs(x).max() * np.sin(np.pi * t) ** 2


def layer(*parts: np.ndarray) -> np.ndarray:
    """Sum sounds of different lengths."""
    out = np.zeros(max(len(p) for p in parts))
    for p in parts:
        out[: len(p)] += p
    return out


def marimba(f: float) -> np.ndarray:
    return bell(f, 0.35, 0.12, ((1, 1), (3.93, 0.18), (9.2, 0.04)))


def sound(name: str, dur: float | None) -> np.ndarray:
    if name == "click":
        return layer(0.5 * bell(2400, 0.04, 0.006), 0.3 * bell(1100, 0.06, 0.012), 0.06 * noise(0.01, 1000, 8000))
    if name == "tap":
        return layer(0.22 * bell(1500, 0.06, 0.012), 0.14 * bell(700, 0.08, 0.02))
    if name == "tick":
        return 0.16 * bell(1760, 0.08, 0.02)
    if name == "pop":
        return 0.3 * sweep(380, 900, 0.09, 0.03)
    if name == "notify":
        out, lag = 0.22 * bell(1318.5), int(0.12 * SR)  # two-note chime
        out[lag:] += 0.22 * bell(1975.5)[: len(out) - lag]
        return out
    if name == "success":
        out = np.zeros(int(1.2 * SR))
        for i, f in enumerate([1046.5, 1318.5, 1568.0, 2093.0]):
            s = int(i * 0.07 * SR)
            out[s:s + int(0.9 * SR)] += 0.17 * bell(f)
        return out
    if name == "send":
        return layer(0.35 * whoosh(0.4, 3), 0.12 * sweep(500, 1300, 0.18, 0.08))
    if name == "whoosh":
        return 0.22 * whoosh(dur or 0.6, 1)
    if name == "swipe":
        return 0.25 * whoosh(0.3, 2)
    if name == "clock":
        return layer(0.2 * bell(2900, 0.05, 0.005), 0.1 * bell(1450, 0.05, 0.01))
    if name == "ring":  # a marimba ringtone: phrases of six notes with a pause
        out, step = np.zeros(int((dur or 1.6) * SR) + SR), 0.12
        phrase = [659.3, 830.6, 987.8, 1318.5, 987.8, 830.6]
        t0 = 0.0
        while t0 < (dur or 1.6):
            for i, f in enumerate(phrase):
                s = int((t0 + i * step) * SR)
                if t0 + i * step < (dur or 1.6):
                    out[s:s + int(0.35 * SR)] += 0.2 * marimba(f)[: len(out) - s]
            t0 += len(phrase) * step + 0.25
        return out
    if name == "hum":  # the compression motor
        t = t_axis(dur or 0.5)
        env = np.minimum(1, t / 0.06) * np.clip(((dur or 0.5) - t) / 0.08, 0, 1)
        return 0.07 * env * (np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 300 * t) + 0.25 * np.sin(2 * np.pi * 452 * t))
    if name == "beep":  # the exposure
        t = t_axis(0.35)
        return 0.12 * np.minimum(1, t / 0.01) * np.clip((0.3 - t) / 0.05, 0, 1) * np.sin(2 * np.pi * 1568 * t)
    if name == "thump":  # a soft low pulse
        return layer(0.5 * bell(62, 0.4, 0.09, ((1, 1), (2, 0.2))), np.pad(0.35 * bell(58, 0.4, 0.09, ((1, 1),)), (int(0.22 * SR), 0)))
    if name == "chord":  # the closing chord, Cmaj9
        t = t_axis(2.2)
        env = np.minimum(1, t / 0.25) * np.exp(-t / 1.6)
        return env * sum(0.05 * np.sin(2 * np.pi * f * t) for f in [130.8, 196.0, 246.9, 329.6, 293.7, 523.3])
    raise ValueError(name)


def music(duration: float, cfg: dict) -> np.ndarray:
    """A soft pad with a plucked arpeggio, one chord per 2.5 s bar (96 bpm). The scene's window.MUSIC can set
    the chords (Hz, cycled; default Cmaj7 - G/B - Am7 - Fmaj7) and rests ([from, to] s without the arpeggio)."""
    chords = cfg.get("chords") or [[130.8, 164.8, 196.0, 246.9], [123.5, 196.0, 220.0, 293.7], [110.0, 164.8, 196.0, 261.6], [87.3, 220.0, 261.6, 329.6]]
    rests = cfg.get("rests", [])
    bar = 2.5
    n = int(duration * SR)
    left, right = np.zeros(n), np.zeros(n)
    t = t_axis(bar + 1.0)
    for b in range(int(np.ceil(duration / bar))):
        notes = chords[b % len(chords)]
        env = np.minimum(1, t / 0.5) * np.clip((bar + 1.0 - t) / 0.9, 0, 1)
        s = int(b * bar * SR)
        for f in notes:
            for side, det in [(left, 0.998), (right, 1.002)]:
                v = 0.03 * env * (np.sin(2 * np.pi * f * det * t) + 0.3 * np.sin(2 * np.pi * 2 * f * det * t))
                side[s:s + len(v)] += v[: n - s]
        # eighth-note arpeggio an octave up, from bar 2 to the closing chord
        for k in range(8):
            at = b * bar + k * bar / 8
            if at < bar or at >= duration - 2.2 or any(a <= at < z for a, z in rests):
                continue
            f = 2 * notes[[0, 1, 2, 3, 2, 1, 2, 3][k]]
            v = 0.045 * bell(f, 0.6, 0.22, ((1, 1), (2, 0.25)))
            i = int(at * SR)
            pan = 0.5 + 0.3 * np.sin(k)
            left[i:i + len(v)] += (1 - pan) * 2 * v[: n - i]
            right[i:i + len(v)] += pan * 2 * v[: n - i]
    return np.stack([left, right], axis=1)


def soundtrack(duration: float, cues: list, music_cfg: dict, path: Path) -> None:
    n = int(duration * SR)
    mix = 0.8 * music(duration, music_cfg)
    for cue in cues:
        name, at, dur = cue[0], cue[1], cue[2] if len(cue) > 2 else None
        s = sound(name, dur)
        i = int(at * SR)
        if i >= n:
            continue
        s = s[: n - i]
        mix[i:i + len(s)] += s[:, None]
    fade = np.minimum(1, np.minimum(np.arange(n) / (0.05 * SR), (n - np.arange(n)) / (0.5 * SR)))
    mix *= fade[:, None]
    mix *= 0.89 / np.abs(mix).max()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype("<i2").tobytes())


# ---------------------------------------------------------------- main


def main() -> None:
    for scene, out in VIDEOS:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            duration, cues, music_cfg = asyncio.run(frames(scene, folder))
            soundtrack(duration, cues, music_cfg, folder / "audio.wav")
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                            "-framerate", str(FPS), "-i", str(folder / "%05d.png"), "-i", str(folder / "audio.wav"),
                            "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], check=True)
        print(f"wrote {out} ({Path(out).stat().st_size / 1e6:.1f} MB)")
        if music_cfg["poster"] is not None:  # the still a web page shows before play (window.POSTER, seconds)
            poster(out, music_cfg["poster"])


def poster(video: str, at: float) -> None:
    out = Path(video).with_suffix(".jpg")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", str(at), "-i", video,
                    "-frames:v", "1", "-q:v", "3", str(out)], check=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
