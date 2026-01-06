"""
capcut_phrase_highlight_shorts.py

CapCut-like captions for PORTRAIT / YouTube Shorts (1080x1920):
- Full phrase stays on screen
- Only the currently spoken word is highlighted
- Stable (no vertical jumping):
  - fixed anchor + absolute position: \an2\pos(x,y)
  - no vertical scaling (fscy stays 100)

Workflow:
1) Extract audio:
   ffmpeg -i "video.mp4" -vn -ac 1 -ar 16000 -c:a pcm_s16le "audio.wav"

2) Generate ASS:
   python capcut_phrase_highlight_shorts.py audio.wav -o captions_phrase.ass

3) Burn in:
   ffmpeg -i "video.mp4" -vf "ass=captions_phrase.ass" -c:a copy "video_captioned.mp4"

Install:
  pip install faster-whisper
"""

from __future__ import annotations

from typing import List, Dict, Optional
import argparse
from faster_whisper import WhisperModel


# -----------------------
# ASS helpers
# -----------------------
def ass_time(t: float) -> str:
    """Convert seconds -> ASS time h:mm:ss.cs (centiseconds)."""
    cs = int(round(t * 100))
    s, cs = divmod(cs, 100)
    m, s = divmod(s, 60)
    h, m = divmod(h := (m // 60), 60)  # safe-ish, but we can do the classic:
    # (Above line is awkward; using classic below)
    # return classic time
    cs = int(round(t * 100))
    s, cs = divmod(cs, 100)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def make_ass_header(
    play_res_x: int,
    play_res_y: int,
    font: str,
    size: int,
) -> str:
    """
    Bold caption style. Positioning handled per-line with \an2\pos(x,y),
    so margins/alignment in the style are fallback.
    """
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Phrase,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,2,0,1,6,2,2,60,60,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


# -----------------------
# Caption building
# -----------------------
def clean_word(w: str) -> str:
    return (w or "").strip()


def merge_phrases(words: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    """
    Merge common multi-token brand phrases and remove leading articles.
    Example:
      "a" + "cap" + "cut" -> "CapCut"
    Timing preserved from first start to last end.
    """
    out: List[Dict[str, float | str]] = []
    i = 0
    while i < len(words):
        w = str(words[i]["word"]).lower()

        # a + cap + cut -> CapCut
        if (
            i + 2 < len(words)
            and w == "a"
            and str(wor
