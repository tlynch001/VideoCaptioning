"""
capcut_phrase_highlight_stable.py

CapCut-like captions:
- Full phrase stays on screen
- Only the currently spoken word is highlighted (magenta by default)
- NO vertical "jumping" between word changes:
  - highlight pop is horizontal-only (fscy stays 100)
  - phrase is pinned to a fixed screen position with \an2\pos(x,y)

Workflow:
1) Extract audio (recommended):
   ffmpeg -i "caption1.mp4" -vn -ac 1 -ar 16000 -c:a pcm_s16le "audio.wav"

2) Generate ASS:
   python capcut_phrase_highlight_stable.py

3) Burn in:
   ffmpeg -i "caption1.mp4" -vf "ass=captions_phrase.ass" -c:a copy "caption1_phrase.mp4"

Install:
  pip install faster-whisper
"""

from __future__ import annotations

from typing import List, Dict, Optional
from faster_whisper import WhisperModel


# -----------------------
# ASS helpers
# -----------------------
def ass_time(t: float) -> str:
    """Convert seconds -> ASS time h:mm:ss.cs (centiseconds)."""
    cs = int(round(t * 100))
    s, cs = divmod(cs, 100)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def make_ass_header(
    play_res_x: int = 1920,
    play_res_y: int = 1080,
    font: str = "Arial Black",
    size: int = 78,
) -> str:
    """
    Style tuned for bold phrase captions.
    Note: Positioning is handled per-line via \an2\pos(x,y),
    so margins/alignment here are mostly a fallback.
    """
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Phrase,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,2,0,1,6,2,2,80,80,120,1

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
    Timing is preserved from first start to last end.
    """
    out: List[Dict[str, float | str]] = []
    i = 0

    while i < len(words):
        w = str(words[i]["word"]).lower()

        # a + cap + cut -> CapCut
        if (
            i + 2 < len(words)
            and w == "a"
            and str(words[i + 1]["word"]).lower() == "cap"
            and str(words[i + 2]["word"]).lower() == "cut"
        ):
            out.append({
                "word": "CapCut",
                "start": float(words[i]["start"]),
                "end": float(words[i + 2]["end"]),
            })
            i += 3
            continue

        # cap + cut -> CapCut (fallback)
        if (
            i + 1 < len(words)
            and w == "cap"
            and str(words[i + 1]["word"]).lower() == "cut"
        ):
            out.append({
                "word": "CapCut",
                "start": float(words[i]["start"]),
                "end": float(words[i + 1]["end"]),
            })
            i += 2
            continue

        out.append(words[i])
        i += 1

    return out

def group_words(
    words: List[Dict[str, float | str]],
    max_words: int = 7,
    max_chars: int = 28,
    max_gap: float = 0.65,
) -> List[List[Dict[str, float | str]]]:
    """
    Groups words into short phrases similar to modern short-form captions.
    Splits when:
      - pause between words exceeds max_gap
      - phrase word count exceeds max_words
      - phrase character length exceeds max_chars
    """
    groups: List[List[Dict[str, float | str]]] = []
    cur: List[Dict[str, float | str]] = []
    cur_len = 0
    last_end: Optional[float] = None

    for w in words:
        txt = str(w["word"])
        start = float(w["start"])
        end = float(w["end"])

        gap = (start - last_end) if last_end is not None else 0.0
        would_len = cur_len + (len(txt) + (1 if cur else 0))

        if cur and (gap > max_gap or len(cur) >= max_words or would_len > max_chars):
            groups.append(cur)
            cur = []
            cur_len = 0

        cur.append({"word": txt, "start": start, "end": end})
        cur_len = cur_len + len(txt) + (1 if cur_len else 0)
        last_end = end

    if cur:
        groups.append(cur)

    return groups


def choose_wrap_index(words: List[str]) -> Optional[int]:
    """
    Choose a wrap index for 2-line layout that balances line lengths.
    Returns an index i meaning: words[:i] on line1, words[i:] on line2.
    Returns None if no wrap needed.
    """
    if len(words) < 4:
        return None

    # Only wrap if the phrase is "long enough"
    total_len = len(" ".join(words))
    if total_len <= 18:
        return None

    best_i = 1
    best_score = 10**9
    for i in range(1, len(words)):
        left = " ".join(words[:i])
        right = " ".join(words[i:])
        score = abs(len(left) - len(right))
        if score < best_score:
            best_score = score
            best_i = i
    return best_i


def build_highlight_phrase(
    word_objs: List[Dict[str, float | str]],
    active_idx: int,
    highlight_color: str = "&H00FFFF00&",  # magenta-ish (ASS uses BGR)
    base_color: str = "&H00FFFFFF&",       # white
    pop_scale_x: int = 120,                # HORIZONTAL pop only (prevents vertical jumping)
    force_two_lines: bool = True,
) -> str:
    """
    Build phrase text where only active word is colored + slightly widened (x-scale).
    IMPORTANT: fscy stays 100 to prevent vertical bbox changes -> no jumping.
    """
    words = [str(w["word"]) for w in word_objs]
    wrap_i = choose_wrap_index(words) if force_two_lines else None

    parts: List[str] = []
    for i, w in enumerate(words):
        token = w
        if i == active_idx:
            token = (
                rf"{{\c{highlight_color}}}{w}"
                rf"{{\c{base_color}}}"
            )

        parts.append(token)

        # Insert ASS newline between words if wrapped
        if wrap_i is not None and i == wrap_i - 1:
            parts.append(r"\N")

    # Join with spaces, but avoid extra spaces around \N
    out: List[str] = []
    for t in parts:
        if t == r"\N":
            if out:
                out[-1] = out[-1].rstrip()
            out.append(r"\N")
        else:
            out.append(t + " ")

    return "".join(out).strip()


# -----------------------
# Main
# -----------------------
def main():
    # -------- Settings you may tweak --------
    audio_path = "audio.wav"
    out_ass = "captions_phrase.ass"

    # Whisper model/device
    model_name = "small"
    device = "cpu"          # change to "cuda" later
    compute_type = "int8"   # CPU; for CUDA: "float16"

    # Phrase grouping
    max_words_per_phrase = 7
    max_chars_per_phrase = 28
    max_gap_seconds = 0.65

    # Visual
    font_size = 84
    highlight_color = "&H00FFFF00&"  # magenta-ish
    pop_scale_x = 125                # widen active word slightly (no vertical scaling)
    force_two_lines = True

    # Pin position (prevents any reflow-based jumping)
    # Bottom-center anchor with absolute position:
    pos_x = 960
    pos_y = 830   # increase to move DOWN (try 780-880)

    # Timing
    min_word_dur = 0.10
    end_tail = 0.06
    fad_in_ms = 30
    fad_out_ms = 60
    # ---------------------------------------

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(audio_path, word_timestamps=True, vad_filter=True)

    # Collect words (flat list)
    all_words: List[Dict[str, float | str]] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            ww = clean_word(w.word)
            if ww:
                all_words.append({"word": ww, "start": float(w.start), "end": float(w.end)})
    
    all_words = merge_phrases(all_words)


    groups = group_words(
        all_words,
        max_words=max_words_per_phrase,
        max_chars=max_chars_per_phrase,
        max_gap=max_gap_seconds,
    )

    with open(out_ass, "w", encoding="utf-8") as f:
        f.write(make_ass_header(size=font_size))

        for g in groups:
            # One Dialogue per word: full phrase stays visible, only active word changes.
            for i in range(len(g)):
                start = float(g[i]["start"])

                # End at next word start for snappy highlight, else use word end
                if i < len(g) - 1:
                    end = float(g[i + 1]["start"])
                else:
                    end = float(g[i]["end"])

                if (end - start) < min_word_dur:
                    end = start + min_word_dur
                end += end_tail

                phrase = build_highlight_phrase(
                    g,
                    active_idx=i,
                    highlight_color=highlight_color,
                    base_color="&H00FFFFFF&",
                    pop_scale_x=pop_scale_x,
                    force_two_lines=force_two_lines,
                )

                # Absolute positioning + fade. \an2 anchors bottom-center at pos(x,y).
                text = rf"{{\an2\pos({pos_x},{pos_y})\fad({fad_in_ms},{fad_out_ms})}}{phrase}"

                f.write(
                    f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Phrase,,0,0,0,,{text}\n"
                )

    print(f"Wrote {out_ass}")


if __name__ == "__main__":
    main()
