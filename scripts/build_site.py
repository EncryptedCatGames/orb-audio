#!/usr/bin/env python3
"""Build the static GitHub Pages site in ``site/``.

Two jobs, both idempotent:

1. Bundle the numpy-only ORB sources into ``site/py/orb.zip`` so the page can
   run the real DSP renderer in the browser with Pyodide. This step needs only
   the standard library (``--py-only`` runs just this, as the Pages workflow
   does, so the in-browser engine can never drift from ``src/orb``).
2. Pre-render the sound library (LIBRARY + BLOCKS) and the hero playground
   presets with ``orb.api.render_spec`` into ``site/assets/audio/*.mp3`` plus
   ``site/assets/cues.json`` (title, use line, caption, Spec, audit and
   waveform peaks; the page draws its waveforms from the peaks). The
   playground presets use the DSP path only, the same one the browser runs;
   the library also runs the optional detail pass (``weights/detail.pt``,
   needs torch). Needs numpy and ffmpeg. If torch and ``weights/planner.pt``
   are available, every caption is also run through the planner and the
   result is recorded as ``planner_exact``. The build fails if any cue fails
   the audit.

Run:
    .venv-torch/bin/python scripts/build_site.py            # everything
    .venv-torch/bin/python scripts/build_site.py --force    # re-render audio
    python scripts/build_site.py --py-only                  # stdlib only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SITE = ROOT / "site"
PY_DIR = SITE / "py"
AUDIO_DIR = SITE / "assets" / "audio"
PLANNER_DIR = SITE / "assets" / "planner"
CUES_PATH = SITE / "assets" / "cues.json"

# The numpy-only slice of the package. Anything that imports torch
# (model/, train/, render/detail.py, eval/mel.py) stays out.
PY_FILES = [
    "orb/__init__.py",
    "orb/api.py",
    "orb/io.py",
    "orb/score.py",
    "orb/grammar.py",
    "orb/render/__init__.py",
    "orb/render/dsp.py",
    "orb/render/events.py",
    "orb/render/foley.py",
    "orb/render/natural.py",
    "orb/render/ambience.py",
    "orb/render/mix.py",
    "orb/render/music.py",
    "orb/render/texture.py",
    "orb/eval/__init__.py",
    "orb/eval/audit.py",
    "orb/eval/retrieval.py",
    "orb/model/spec_tokenizer.py",  # numpy-only; the in-browser planner decodes with it
]
# orb/model/__init__.py imports torch, so the bundle ships an empty one instead.
PY_STUBS = {"orb/model/__init__.py": b'"""Browser bundle: only the numpy-only spec tokenizer."""\n'}
SITE_MODULES = ["orb_web.py", "orb_planner.py"]

# Pan values are exact grammar.pan_value outputs: L3 -0.75, L2 -0.5, L1 -0.25,
# C 0, R1 0.25, R2 0.5, R3 0.75.
def ev(role, t, dur, gain_db, pan, seed):
    return {"role": role, "t": t, "dur": dur, "gain_db": gain_db, "pan": pan, "seed": seed}


# The sound library on the page: practical cues grouped by the job they do.
# Hand-written Specs (reviewed by ear), rendered with the optional detail pass
# (weights/detail.pt) and the playground's loudness cap, so "play all" does not
# jump in level. `use` is the one-line "where you would put it" on each card.
LIBRARY_GROUPS = ["App & UI", "Video editing", "Brand", "Games", "Nature & weather", "Places & scenes", "Action", "Everyday objects"]


def cue(group, name, title, use, spec):
    return {"group": group, "name": name, "title": title, "use": use, "spec": spec}


LIBRARY = [
    cue("App & UI", "notify", "Message received",
        "Chat apps, inbox, push notification.",
        {"duration": 2.0, "texture": "none", "mood": "bright",
         "events": [ev("ui_pop", 0.10, 0.12, -8, -0.25, 3), ev("bell", 0.22, 1.2, -12, 0.25, 7)]}),
    cue("App & UI", "tap", "Button tap + toggle",
        "Buttons, switches, checkboxes.",
        {"duration": 1.5, "texture": "none", "mood": "neutral",
         "events": [ev("click", 0.10, 0.05, -10, 0, 11), ev("ui_pop", 0.60, 0.10, -10, 0, 12)]}),
    cue("App & UI", "success", "Payment success",
        "Checkout complete, task done, upload finished.",
        {"duration": 2.5, "texture": "none", "mood": "hopeful",
         "events": [ev("ui_pop", 0.10, 0.10, -10, -0.25, 21), ev("ui_pop", 0.22, 0.10, -9, 0.25, 22),
                    ev("bell", 0.34, 1.6, -11, 0, 23)]}),
    cue("App & UI", "error", "Error / denied",
        "Wrong password, invalid input, blocked action.",
        {"duration": 1.5, "texture": "none", "mood": "tense",
         "events": [ev("glitch", 0.10, 0.18, -12, 0, 31), ev("click", 0.32, 0.05, -10, 0, 32),
                    ev("sub_drop", 0.32, 0.5, -14, 0, 33)]}),
    cue("App & UI", "typing", "Typing sequence",
        "Terminal UI, chat 'is typing', step-by-step reveal.",
        {"duration": 2.0, "texture": "none", "mood": "neutral",
         "events": [ev("click", 0.10 + 0.17 * i, 0.04, -12 + (i % 2), (-0.25, 0.25)[i % 2], 40 + i)
                    for i in range(7)] + [ev("ui_pop", 1.40, 0.10, -9, 0, 49)]}),
    cue("Video editing", "transition", "Slide transition",
        "Cuts between clips, swipe between slides.",
        {"duration": 2.0, "texture": "none", "mood": "bright",
         "events": [ev("whoosh", 0.20, 0.9, -6, 0, 51), ev("click", 1.05, 0.04, -14, 0, 52)]}),
    cue("Video editing", "title", "YouTube title hit",
        "Title cards, lower thirds, punch-ins.",
        {"duration": 3.0, "texture": "none", "mood": "intense",
         "events": [ev("whoosh", 0.20, 0.8, -8, -0.5, 61), ev("impact", 1.00, 0.9, -4, 0, 62),
                    ev("sub_drop", 1.00, 1.2, -10, 0, 63)]}),
    cue("Video editing", "podcast", "Podcast segment break",
        "Segment change, chapter marker, ad break.",
        {"duration": 4.0, "texture": "vinyl", "mood": "warm", "bed_db": -38,
         "events": [ev("noise_sweep", 0.30, 1.6, -16, 0, 71), ev("bell", 1.80, 1.8, -12, 0.25, 72)]}),
    cue("Brand", "logo", "Logo reveal sting",
        "App launch, intro bumper, brand outro.",
        {"duration": 6.0, "texture": "drone", "mood": "hopeful",
         "events": [ev("riser", 0.30, 2.2, -16, 0, 81), ev("whoosh", 2.10, 0.6, -10, 0, 82),
                    ev("impact", 2.60, 0.9, -5, 0, 83), ev("sub_drop", 2.60, 1.5, -9, 0, 84),
                    ev("bell", 2.62, 2.5, -11, 0, 85)]}),
    cue("Brand", "trailer", "Trailer hit",
        "Teasers, reveals, dramatic cuts.",
        {"duration": 8.0, "texture": "drone", "mood": "intense",
         "events": [ev("riser", 0.5, 2.5, -18, 0, 85), ev("whoosh", 2.6, 1.0, -10, -0.5, 119),
                    ev("impact", 3.3, 0.9, -4, 0, 102), ev("sub_drop", 3.3, 1.6, -8, 0, 136),
                    ev("noise_sweep", 4.6, 1.5, -16, 0.5, 151), ev("impact", 6.3, 0.9, -5, 0, 153),
                    ev("bell", 6.3, 1.6, -13, 0.25, 61)]}),
    cue("Games", "pickup", "Coin pickup",
        "Collectibles, XP gain, reward.",
        {"duration": 1.2, "texture": "none", "mood": "playful",
         "events": [ev("ui_pop", 0.05, 0.08, -9, 0, 91), ev("bell", 0.12, 0.8, -12, 0, 92)]}),
    cue("Games", "levelup", "Level up",
        "Level complete, achievement unlocked.",
        {"duration": 3.5, "texture": "none", "mood": "hopeful",
         "events": [ev("riser", 0.10, 1.0, -16, 0, 101), ev("ui_pop", 1.10, 0.1, -9, -0.5, 102),
                    ev("ui_pop", 1.22, 0.1, -9, 0.5, 103), ev("swell", 1.20, 1.6, -15, 0, 104),
                    ev("bell", 1.34, 2.0, -10, 0, 105)]}),
    cue("Games", "gameover", "Game over",
        "Fail state, defeat, out of lives.",
        {"duration": 3.5, "texture": "none", "mood": "dark",
         "events": [ev("glitch", 0.10, 0.3, -12, 0, 111), ev("impact", 0.45, 0.9, -6, 0, 112),
                    ev("sub_drop", 0.45, 2.2, -8, 0, 113)]}),
    cue("Games", "countdown", "Race countdown",
        "3-2-1-go, timers, round start.",
        {"duration": 4.0, "texture": "none", "mood": "tense",
         "events": [ev("click", 0.2, 0.05, -8, 0, 121), ev("click", 1.2, 0.05, -8, 0, 122),
                    ev("click", 2.2, 0.05, -8, 0, 123), ev("bell", 3.2, 0.8, -8, 0, 124),
                    ev("impact", 3.2, 0.6, -10, 0, 125)]}),
    cue("Places & scenes", "airlock", "Sci-fi airlock",
        "Game levels, sci-fi shorts, VR.",
        {"duration": 6.0, "texture": "sci_fi", "mood": "tense",
         "events": [ev("bell", 0.3, 1.0, -16, 0, 131), ev("click", 1.2, 0.05, -10, -0.5, 132),
                    ev("click", 1.35, 0.05, -10, 0.5, 133), ev("noise_sweep", 1.6, 2.0, -14, 0, 134),
                    ev("sub_drop", 1.6, 1.5, -14, 0, 135), ev("whoosh", 3.6, 1.0, -10, 0, 136),
                    ev("impact", 4.6, 0.8, -8, 0, 137)]}),
    cue("Places & scenes", "horror", "Horror jump scare",
        "Horror games, creepy reels.",
        {"duration": 5.0, "texture": "drone", "mood": "eerie",
         "events": [ev("swell", 0.3, 2.6, -18, 0, 141), ev("glitch", 2.9, 0.25, -10, 0.25, 142),
                    ev("impact", 3.0, 0.9, -3, 0, 143), ev("sub_drop", 3.0, 1.8, -8, 0, 144)]}),
    cue("Nature & weather", "storm", "Rain + distant thunder",
        "Ambience for focus videos, scene background.",
        {"duration": 8.0, "texture": "rain", "mood": "calm",
         # The real thunder role: seed % 4 = 3 is a far rumble, 2 a mid-distance roll.
         "events": [ev("thunder", 0.6, 6.0, -10, -0.25, 0x0f), ev("thunder", 4.4, 3.6, -16, 0.5, 0x0e)]}),
    # Physically modelled natural and foley roles (orb.render.foley). The seed
    # picks the variant: thunder seed % 4 is distance (0 close crack .. 3 far
    # rumble), footsteps seed % 3 the floor, explosion seed % 3 the distance.
    cue("Nature & weather", "thunderstorm", "Thunderstorm",
        "Storm scenes, weather in games, film ambience.",
        {"duration": 14.0, "texture": "rain", "mood": "dark",
         "events": [ev("thunder", 0.8, 7.0, -10, -0.5, 0x03), ev("thunder", 6.6, 6.5, -4, 0.25, 0x04)]}),
    cue("Places & scenes", "arrival", "Footsteps, door, knock",
        "Someone arrives: thrillers, games, radio drama.",
        {"duration": 8.0, "texture": "room_cold", "mood": "tense",
         "events": [ev("footsteps", 0.4, 3.2, -12, -0.5, 0x04), ev("door_slam", 3.9, 0.9, -6, 0, 0x01),
                    ev("knock", 5.6, 0.7, -10, 0.25, 0x03)]}),
    cue("Nature & weather", "cave", "Cave drips",
        "Dungeons, caves, quiet tension.",
        {"duration": 6.0, "texture": "drone", "mood": "eerie",
         "events": [ev("water_drop", 0.6, 0.35, -14, -0.5, 0x02), ev("water_drop", 1.9, 0.35, -18, 0.5, 0x05),
                    ev("water_drop", 3.1, 0.35, -12, 0, 0x08), ev("water_drop", 4.4, 0.35, -16, -0.25, 0x0b)]}),
    cue("Action", "explosion", "Explosion",
        "Game blasts, action cuts, destruction.",
        {"duration": 7.0, "texture": "wind", "mood": "intense",
         "events": [ev("explosion", 0.3, 3.2, -2, 0, 0x00), ev("explosion", 3.4, 3.0, -10, 0.5, 0x02)]}),
    # v0.3 scene beds (ocean, stream, forest, night, traffic) are built from the
    # same physical models, so a "beach" is waves breaking, not shaped noise.
    cue("Nature & weather", "beach", "Beach",
        "Travel videos, calm scenes, meditation apps.",
        {"duration": 12.0, "texture": "ocean", "mood": "calm",
         "events": [ev("wave_crash", 0.5, 6.0, -8, -0.25, 0x01), ev("bird_chirp", 5.0, 3.0, -18, 0.75, 0x03)]}),
    cue("Nature & weather", "forest_morning", "Forest morning",
        "Nature documentaries, game overworlds, wake-up screens.",
        {"duration": 12.0, "texture": "forest", "mood": "hopeful",
         "events": [ev("bird_chirp", 0.6, 3.0, -12, -0.5, 0x02), ev("bird_chirp", 4.2, 3.0, -16, 0.5, 0x01),
                    ev("wind_gust", 6.0, 4.0, -16, 0, 0x02)]}),
    cue("Nature & weather", "night_camp", "Night by the pond",
        "Night scenes, sleep sounds, quiet game levels.",
        {"duration": 12.0, "texture": "night", "mood": "calm",
         "events": [ev("frog", 1.0, 3.0, -14, -0.5, 0x00), ev("owl", 6.0, 3.0, -14, 0.5, 0x00)]}),
    cue("Nature & weather", "river", "Stream",
        "Relaxation, hiking footage, a garden.",
        {"duration": 10.0, "texture": "stream", "mood": "calm",
         "events": [ev("bird_chirp", 3.0, 3.0, -18, 0.5, 0x00)]}),
    cue("Places & scenes", "city_street", "City street",
        "Urban b-roll, establishing shots, open-world games.",
        {"duration": 12.0, "texture": "traffic", "mood": "neutral",
         "events": [ev("engine", 1.0, 4.0, -10, -0.5, 0x02), ev("car_horn", 5.5, 1.0, -14, 0.5, 0x01),
                    ev("siren", 6.5, 5.0, -18, -0.25, 0x00)]}),
    cue("Action", "sword_fight", "Sword fight",
        "Fantasy games, fight scenes, trailers.",
        {"duration": 6.0, "texture": "wind", "mood": "intense",
         "events": [ev("sword_clash", 0.3, 1.5, -8, 0, 0x01), ev("sword_clash", 1.4, 1.5, -6, -0.25, 0x00),
                    ev("sword_clash", 2.3, 1.5, -8, 0.25, 0x02), ev("punch", 3.6, 0.5, -8, 0, 0x02)]}),
    cue("Places & scenes", "haunted_house", "Haunted house",
        "Horror games, Halloween, creepy reveals.",
        {"duration": 10.0, "texture": "drone", "mood": "eerie",
         "events": [ev("wood_creak", 0.5, 2.5, -12, -0.5, 0x01), ev("clock_tick", 0.5, 8.0, -20, 0.5, 0x02),
                    ev("wood_creak", 4.0, 2.5, -14, 0.25, 0x00), ev("door_slam", 7.0, 0.9, -8, 0, 0x01)]}),
    # Everyday: one familiar sound per card, the kind people search for.
    cue("Places & scenes", "kitchen", "Kitchen",
        "Cooking videos, home scenes, ads.",
        {"duration": 8.0, "texture": "room_cold", "mood": "warm",
         "events": [ev("can_opening", 0.3, 1.0, -10, 0, 0x00), ev("pouring_water", 1.6, 3.0, -12, 0.25, 0x00),
                    ev("metal_clang", 5.2, 2.0, -16, -0.5, 0x01)]}),
    cue("Places & scenes", "office", "Office",
        "Work-from-home b-roll, productivity apps.",
        {"duration": 8.0, "texture": "room_cold", "mood": "neutral",
         "events": [ev("keyboard_typing", 0.3, 4.0, -14, -0.25, 0x00), ev("mouse_click", 4.6, 0.3, -12, 0.25, 0x01),
                    ev("phone_ring", 5.2, 2.6, -14, 0.5, 0x02)]}),
    cue("Everyday objects", "glass", "Glass breaking",
        "Accidents, break-ins, dramatic beats.",
        {"duration": 3.5, "texture": "none", "mood": "tense", "events": [ev("glass_break", 0.1, 1.6, -6, 0, 0x00)]}),
    cue("Everyday objects", "alarm", "Alarm clock",
        "Morning routines, wake-up moments.",
        {"duration": 4.0, "texture": "none", "mood": "tense", "events": [ev("clock_alarm", 0.1, 3.0, -10, 0, 0x00)]}),
    cue("Everyday objects", "phone", "Phone ringing",
        "Incoming calls, plot twists.",
        {"duration": 5.0, "texture": "none", "mood": "neutral", "events": [ev("phone_ring", 0.1, 4.0, -10, 0, 0x00)]}),
    cue("Everyday objects", "camera", "Camera shutter",
        "Screenshot, photo apps, paparazzi.",
        {"duration": 1.5, "texture": "none", "mood": "neutral", "events": [ev("camera_shutter", 0.1, 0.6, -10, 0, 0x00)]}),
    cue("Everyday objects", "applause", "Applause",
        "Wins, finales, end of a talk.",
        {"duration": 6.0, "texture": "none", "mood": "hopeful", "events": [ev("applause", 0.1, 5.0, -8, 0, 0x01)]}),
    cue("Everyday objects", "champagne", "Champagne pop",
        "Celebrations, launches, milestones.",
        {"duration": 2.5, "texture": "none", "mood": "playful", "events": [ev("cork_pop", 0.1, 1.5, -8, 0, 0x01)]}),
    cue("Everyday objects", "coins", "Coins in a dish",
        "Payments, tips, rewards.",
        {"duration": 2.5, "texture": "none", "mood": "bright", "events": [ev("coin_drop", 0.1, 1.5, -10, 0, 0x02)]}),
    cue("Everyday objects", "heartbeat", "Heartbeat",
        "Suspense, health apps, horror.",
        {"duration": 5.0, "texture": "none", "mood": "tense", "events": [ev("heartbeat", 0.1, 4.5, -6, 0, 0x01)]}),
    cue("Everyday objects", "church_bell", "Church bell",
        "Towns, weddings, the hour striking.",
        {"duration": 8.0, "texture": "none", "mood": "calm", "events": [ev("church_bell", 0.1, 7.0, -8, 0, 0x01)]}),
    cue("Action", "fireworks", "Fireworks",
        "New Year, celebrations, game wins.",
        {"duration": 6.0, "texture": "none", "mood": "playful",
         "events": [ev("fireworks", 0.1, 4.0, -8, -0.25, 0x00), ev("fireworks", 2.0, 4.0, -12, 0.5, 0x01)]}),
    cue("Action", "gunshot", "Gunshot",
        "Action games, westerns, starting pistol.",
        {"duration": 3.0, "texture": "none", "mood": "intense", "events": [ev("gunshot", 0.1, 1.5, -6, 0, 0x00)]}),
]

# Example chips under the hero prompt box: real grammar captions of library
# cues, built with orb.grammar.caption, so each one is exact. Cues with a bed
# go through the planner, which only knows even dB steps, so their gains are
# snapped to the grid first (at most 1 dB off the library version); "no bed"
# cues are read exactly, as orb.api.generate does.
PROMPT_CHIPS = [
    ("success", "Payment success"),
    ("levelup", "Level up"),
    ("trailer", "Trailer hit"),
    ("horror", "Horror scare"),
    ("airlock", "Airlock"),
    ("storm", "Rain + thunder"),
    ("thunderstorm", "Thunderstorm"),
    ("arrival", "Footsteps, door, knock"),
    ("notify", "Notification"),
    ("gameover", "Game over"),
    ("beach", "Beach"),
    ("forest_morning", "Forest morning"),
    ("night_camp", "Night by the pond"),
    ("city_street", "City street"),
    ("sword_fight", "Sword fight"),
    ("kitchen", "Kitchen"),
    ("office", "Office"),
    ("glass", "Glass breaking"),
]

# Building blocks: single-event cues that show what one line of the Spec does.
# Rendered the same way as the library.
BLOCKS = [
    {"group": "Building blocks", "name": "whoosh_air", "title": "Whoosh",
     "blurb": "One pass, panned left, over a wind bed.",
     "spec": {"duration": 4.0, "texture": "wind", "mood": "bright",
              "events": [ev("whoosh", 0.6, 0.9, -6.0, -0.5, 0x2A)]}},
    {"group": "Building blocks", "name": "impact_deep", "title": "Impact",
     "blurb": "Short and deep, in a cold room.",
     "spec": {"duration": 4.0, "texture": "room_cold", "mood": "dark",
              "events": [ev("impact", 0.6, 0.8, -4.0, 0.0, 0x1F)]}},
    {"group": "Building blocks", "name": "riser_tension", "title": "Riser",
     "blurb": "Tension that peaks exactly on its mark.",
     "spec": {"duration": 4.0, "texture": "drone", "mood": "tense",
              "events": [ev("riser", 0.4, 2.4, -16.0, 0.0, 0x51)]}},
    {"group": "Building blocks", "name": "sub_drop_clean", "title": "Sub drop",
     "blurb": "A clean low-end fall over a sci-fi hum.",
     "spec": {"duration": 4.0, "texture": "sci_fi", "mood": "intense",
              "events": [ev("sub_drop", 0.6, 1.2, -8.0, 0.0, 0x66)]}},
]


# The hero playground: hand-written Specs, ordered wow -> basic, so the page
# opens on the strongest one. Each ships pre-rendered (audio + peaks), so
# selecting it is instant; every edit after that re-renders in the browser.
#
# `controls` are the touch controls the page builds for the preset (event
# markers are always draggable on the waveform). Kinds:
#   pan     {events, label}            one slider, L..R; only roles that pan
#                                      (impact, ui_pop, click, sub_drop: the
#                                      wide roles ignore `pan` in the renderer)
#   gain    {events, label, min, max}  dB slider; moves the events together
#   seed    {events, label}            "new take": fresh seeds for those events,
#                                      plus a small, bounded jitter of their
#                                      length and level (app.js `newTake`): the
#                                      modal/sine bodies (impact, sub_drop) do
#                                      not change with the seed alone
#   texture {label, options}           the bed ("none" = digital silence)
#   bed_db  {label, min, max}          bed level vs the foreground
# `marks` are the draggable markers on the waveform: each names a group of
# events that move together by the same delta, drawn at its first event (the
# anchor). Events not in any mark get their own marker.
#
# Beds sit at -40 dB (grammar.DEFAULT_BED_DB): every texture is shaped noise,
# and -30 left an audible hiss in the gaps. Only Storm runs its bed louder,
# because the rain is the scene. Every cue is also capped to the same
# momentary loudness (orb_web.level), so switching chips does not jump.
#
# Why each one sounds the way it does:
# - trailer: a 2.9 s riser (apex 2.97 s) climbs into impact + sub_drop +
#   a decaying whoosh of air at 3.2 s; a swell (apex 5.77 s) sucks into the
#   answering hit at 5.8 s. No bell: the answer stays cinematic.
# - launch: a three-click countdown, panned L R L at 1 s spacing, the riser
#   starts on "1" and peaks at 4.26 s, then impact + sub + a long whoosh
#   tail with a quiet bell shimmer. No bed: the countdown sits in silence.
# - storm: rain at -28 dB; a far thunder rumble on the right, then at 3.6 s
#   a close strike on the left (the thunder role's crack, rumble and long
#   low tail). Both are the physically modelled thunder, no synth stand-ins.
# - airlock: warning bell, two latch clicks (L then R), the vent alone
#   (noise_sweep), the door slides (whoosh), and the seal (impact + sub)
#   locks with a click. Eight events, the grammar's maximum.
# - glitch: short glitch steps and two hard-panned clicks stutter for 0.6 s,
#   a riser inhales (apex 1.19 s), then impact + sub_drop land. No bed: a
#   stinger must cut cleanly into whatever plays next.
# - notify: the only "basic" UI cue. No bed, so the gaps are digital zero;
#   the click leads the pops by 80 ms, the pops step left -> right, and the
#   bell's slow t60 carries the tail.
PLAYGROUND = [
    {
        "name": "pg_trailer",
        "title": "Trailer hit",
        "tag": "cinematic",
        "blurb": "A riser into a stacked hit, a breath of air, then the answer.",
        "spec": {"duration": 8.0, "texture": "drone", "mood": "intense", "bed_db": -40.0,
                 "events": [
                     ev("riser", 0.3, 2.9, -12.0, 0.0, 0x55),
                     ev("swell", 1.5, 1.6, -18.0, 0.0, 0x56),
                     ev("impact", 3.2, 1.4, -2.0, 0.0, 0x66),
                     ev("sub_drop", 3.2, 2.4, -4.0, 0.0, 0x88),
                     ev("whoosh", 3.2, 1.4, -14.0, 0.0, 0x77),
                     ev("swell", 4.6, 1.3, -16.0, 0.0, 0x57),
                     ev("impact", 5.8, 1.2, -5.0, 0.0, 0x99),
                     ev("sub_drop", 5.8, 1.8, -8.0, 0.0, 0x89),
                 ]},
        "marks": [
            {"events": [0], "label": "riser"},
            {"events": [1], "label": "swell"},
            {"events": [2, 3, 4], "label": "hit"},
            {"events": [5], "label": "inhale"},
            {"events": [6, 7], "label": "answer"},
        ],
        "controls": [
            {"kind": "gain", "events": [3], "label": "sub weight", "min": -20, "max": 0},
            {"kind": "texture", "label": "bed", "options": ["drone", "sci_fi", "city_hum", "none"]},
            {"kind": "seed", "events": [0, 1, 2, 3, 4, 5, 6, 7], "label": "new take"},
        ],
    },
    {
        "name": "pg_launch",
        "title": "Launch",
        "tag": "countdown",
        "blurb": "Three, two, one: the lift climbs from the last count into the launch.",
        "spec": {"duration": 7.0, "texture": "none", "mood": "intense",
                 "events": [
                     ev("click", 0.3, 0.06, -12.0, -0.5, 0x51),
                     ev("click", 1.3, 0.06, -12.0, 0.5, 0x52),
                     ev("click", 2.3, 0.06, -10.0, -0.5, 0x53),
                     ev("riser", 2.4, 1.8, -10.0, 0.0, 0x54),
                     ev("impact", 4.3, 1.2, -2.0, 0.0, 0x55),
                     ev("sub_drop", 4.3, 2.2, -4.0, 0.0, 0x56),
                     ev("whoosh", 4.3, 2.0, -10.0, 0.0, 0x57),
                     ev("bell", 4.3, 2.4, -18.0, 0.0, 0x58),
                 ]},
        "marks": [
            {"events": [0], "label": "3"},
            {"events": [1], "label": "2"},
            {"events": [2, 3], "label": "1 · lift"},
            {"events": [4, 5, 6, 7], "label": "launch"},
        ],
        "controls": [
            {"kind": "pan", "events": [1], "label": "count 2"},
            {"kind": "gain", "events": [5], "label": "sub weight", "min": -20, "max": 0},
            {"kind": "texture", "label": "bed", "options": ["none", "wind", "sci_fi", "city_hum"]},
        ],
    },
    {
        "name": "pg_storm",
        "title": "Storm",
        "tag": "scene",
        "blurb": "Rain and a far rumble, then a close strike cracks on the left and rolls away.",
        "spec": {"duration": 10.0, "texture": "rain", "mood": "dark", "bed_db": -28.0,
                 # All real thunder (orb.render.foley): seed % 4 is distance,
                 # 3 far rumble, 0 close crack, 2 mid.
                 "events": [
                     ev("thunder", 0.3, 4.5, -16.0, 0.5, 0x33),
                     ev("thunder", 3.6, 6.4, -4.0, -0.5, 0x30),
                 ]},
        "marks": [
            {"events": [0], "label": "far roll"},
            {"events": [1], "label": "close strike"},
        ],
        "controls": [
            {"kind": "pan", "events": [1], "label": "strike"},
            {"kind": "bed_db", "label": "rain", "min": -44, "max": -16},
            {"kind": "seed", "events": [0, 1], "label": "new thunder"},
        ],
    },
    {
        "name": "pg_airlock",
        "title": "Airlock",
        "tag": "scene",
        "blurb": "Warning, latch, the vent hisses, the door slides, seals and locks.",
        "spec": {"duration": 7.0, "texture": "sci_fi", "mood": "eerie", "bed_db": -40.0,
                 "events": [
                     ev("bell", 0.3, 1.0, -16.0, 0.0, 0x3D),
                     ev("click", 1.2, 0.08, -12.0, -0.5, 0x41),
                     ev("click", 1.35, 0.08, -14.0, 0.5, 0x42),
                     ev("noise_sweep", 1.6, 2.2, -14.0, 0.0, 0x97),
                     ev("whoosh", 3.8, 1.2, -8.0, 0.0, 0x2B),
                     ev("impact", 4.9, 0.9, -6.0, 0.0, 0x1F),
                     ev("sub_drop", 4.9, 1.4, -10.0, 0.0, 0x88),
                     ev("click", 5.3, 0.1, -14.0, 0.0, 0x43),
                 ]},
        "marks": [
            {"events": [0], "label": "warn"},
            {"events": [1, 2], "label": "latch"},
            {"events": [3], "label": "vent"},
            {"events": [4], "label": "slide"},
            {"events": [5, 6, 7], "label": "seal"},
        ],
        "controls": [
            {"kind": "gain", "events": [3], "label": "vent level", "min": -28, "max": -4},
            {"kind": "pan", "events": [5, 6], "label": "seal"},
            {"kind": "texture", "label": "bed", "options": ["sci_fi", "room_cold", "none"]},
        ],
    },
    {
        "name": "pg_glitch",
        "title": "Glitch",
        "tag": "stinger",
        "blurb": "A digital stutter, an inhale, a landing. Built to cut into anything.",
        "spec": {"duration": 3.0, "texture": "none", "mood": "intense",
                 "events": [
                     ev("glitch", 0.1, 0.05, -12.0, 0.0, 0xB4),
                     ev("glitch", 0.24, 0.04, -14.0, 0.0, 0x5C),
                     ev("click", 0.36, 0.05, -12.0, -1.0, 0x11),
                     ev("glitch", 0.46, 0.08, -12.0, 0.0, 0x3E),
                     ev("click", 0.66, 0.05, -12.0, 1.0, 0x12),
                     ev("riser", 0.75, 0.48, -12.0, 0.0, 0x77),
                     ev("impact", 1.2, 0.7, -4.0, 0.0, 0x66),
                     ev("sub_drop", 1.2, 1.4, -8.0, 0.0, 0x88),
                 ]},
        "marks": [
            {"events": [0, 1, 2, 3, 4], "label": "stutter"},
            {"events": [5], "label": "inhale"},
            {"events": [6, 7], "label": "land"},
        ],
        "controls": [
            {"kind": "seed", "events": [0, 1, 3], "label": "new stutter"},
            {"kind": "gain", "events": [7], "label": "sub weight", "min": -24, "max": -2},
        ],
    },
    {
        "name": "pg_notify",
        "title": "Notify",
        "tag": "basic",
        "blurb": "Click, two pops and a bell. No bed: the gaps are true silence.",
        "spec": {"duration": 2.5, "texture": "none", "mood": "bright",
                 "events": [
                     ev("click", 0.12, 0.06, -14.0, -0.25, 0x21),
                     ev("ui_pop", 0.2, 0.16, -10.0, -0.25, 0x22),
                     ev("ui_pop", 0.38, 0.16, -8.0, 0.25, 0x23),
                     ev("bell", 0.56, 1.4, -16.0, 0.0, 0x24),
                 ]},
        "marks": [
            {"events": [0, 1], "label": "tap"},
            {"events": [2], "label": "pop"},
            {"events": [3], "label": "bell"},
        ],
        "controls": [
            {"kind": "pan", "events": [2], "label": "second pop"},
            {"kind": "gain", "events": [3], "label": "bell level", "min": -30, "max": -4},
            {"kind": "texture", "label": "bed", "options": ["none", "vinyl", "room_cold"]},
        ],
    },
]

# Anything that changes how a Spec sounds. Its hash is part of every cache
# key, so a renderer change re-renders the whole site on the next build.
RENDER_SOURCES = sorted((SRC / "orb" / "render").glob("*.py")) + [SRC / "orb" / "grammar.py", SRC / "orb" / "score.py"]


def renderer_hash() -> str:
    h = hashlib.sha256()
    # orb_web.py carries the playground's loudness cap (orb_web.level).
    for p in RENDER_SOURCES + [PY_DIR / "orb_web.py"]:
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:12]


def sync_python() -> Path:
    """Write site/py/orb.zip with fixed timestamps, so it is byte-stable."""
    PY_DIR.mkdir(parents=True, exist_ok=True)
    entries = [(rel, (SRC / rel).read_bytes()) for rel in PY_FILES]
    entries += sorted(PY_STUBS.items())
    entries += [(name, (PY_DIR / name).read_bytes()) for name in SITE_MODULES]
    target = PY_DIR / "orb.zip"
    tmp = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, data in entries:
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    if target.exists() and target.read_bytes() == tmp.read_bytes():
        tmp.unlink()
    else:
        tmp.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()[:12]
    manifest = {"files": [rel for rel, _ in entries], "sha256_12": digest}
    _write_if_changed(PY_DIR / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"py: {len(entries)} files -> {target.relative_to(ROOT)} ({target.stat().st_size} B, {digest})")
    return target


PLANNER_BINS = ("planner.q8.bin", "planner.f16.bin")


def planner_blob_name() -> str | None:
    """The weight file the current planner.json points at."""
    index_path = PLANNER_DIR / "planner.json"
    if not index_path.exists():
        return None
    return json.loads(index_path.read_text(encoding="utf-8")).get("file", "planner.f16.bin")


def export_planner(force: bool = False, dtype: str = "int8") -> None:
    """Write the planner for the browser: site/assets/planner/planner.json
    (config + tensor index) and one weight file, the tied head stored once.

    ``dtype="int8"`` (default since v0.4) stores every matrix as int8 with one
    fp16 scale per row (symmetric, round-to-nearest) and vectors as fp16:
    planner.q8.bin, about half the fp16 size. ``dtype="float16"`` writes
    planner.f16.bin as before. ``site/py/orb_planner.py`` dequantises on load
    and computes in float32; scripts/check_web_planner.py checks it decodes
    the same Specs as torch. Needs torch; skipped when up to date."""
    src = ROOT / "weights" / "planner.pt"
    index_path = PLANNER_DIR / "planner.json"
    fname = "planner.q8.bin" if dtype == "int8" else "planner.f16.bin"
    blob_path = PLANNER_DIR / fname
    if not src.exists():
        print("planner: weights/planner.pt missing; keeping the existing export")
        return
    sha = _file_sha(src)
    if not force and index_path.exists() and blob_path.exists():
        old = json.loads(index_path.read_text(encoding="utf-8"))
        if old.get("source_sha256") == sha and old.get("file", "planner.f16.bin") == fname:
            print(f"planner: export up to date ({blob_path.stat().st_size / 1e6:.1f} MB)")
            return
    import numpy as np
    import torch

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["config"]["model"])
    cfg.pop("dropout", None)
    sd = ckpt["model"]
    tensors, chunks, offset, params = [], [], 0, 0
    for name, value in sd.items():
        if name == "head.weight":  # tied to tok_emb.weight
            continue
        arr = value.detach().float().cpu().numpy()
        params += arr.size
        if dtype == "int8" and arr.ndim == 2:
            scale = np.abs(arr).max(axis=1) / 127.0
            scale = np.where(scale == 0, 1.0, scale).astype(np.float16)
            q = np.clip(np.rint(arr / scale.astype(np.float32)[:, None]), -127, 127).astype(np.int8)
            data = q.tobytes() + scale.astype("<f2").tobytes()
            tensors.append([name, list(arr.shape), offset, "q8"])
        else:
            data = arr.astype("<f2").tobytes()
            tensors.append([name, list(arr.shape), offset, "f16"] if dtype == "int8" else [name, list(arr.shape), offset // 2])
        chunks.append(data)
        offset += len(data)
        if dtype == "int8" and offset % 2:  # keep fp16 runs aligned
            chunks.append(b"\0")
            offset += 1
    PLANNER_DIR.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(b"".join(chunks))
    for other in PLANNER_BINS:
        if other != fname and (PLANNER_DIR / other).exists():
            (PLANNER_DIR / other).unlink()
    index = {
        "config": {**cfg, "vocab_size": int(sd["tok_emb.weight"].shape[0])},
        "dtype": "int8-rows" if dtype == "int8" else "float16",
        "file": fname,
        "params": int(params),
        "step": ckpt.get("step"),
        "source_sha256": sha,
        "bin_sha256_12": _file_sha(blob_path)[:12],
        "tensors": tensors,
    }
    index_path.write_text(json.dumps(index) + "\n", encoding="utf-8")
    print(f"planner: exported {params:,} params ({index['dtype']}) -> {blob_path.relative_to(ROOT)} ({blob_path.stat().st_size / 1e6:.1f} MB)")


def sync_media_flags() -> None:
    """Record in cues.json whether the demo film and poster exist (stdlib only).

    The film is produced separately; the page shows a fallback when it is
    missing, without probing for it and logging a 404.
    """
    if not CUES_PATH.exists():
        return
    data = json.loads(CUES_PATH.read_text(encoding="utf-8"))
    planner_index = PLANNER_DIR / "planner.json"
    if planner_index.exists():
        pi = json.loads(planner_index.read_text(encoding="utf-8"))
        fname = pi.get("file", "planner.f16.bin")
        planner_meta = {"index": "assets/planner/planner.json", "bin": f"assets/planner/{fname}",
                        "v": pi["bin_sha256_12"], "params": pi["params"],
                        "bytes": (PLANNER_DIR / fname).stat().st_size}
    else:
        planner_meta = None
    media = {
        "video": (SITE / "assets" / "orb-demo.mp4").exists(),
        "poster": (SITE / "assets" / "orb-demo-poster.jpg").exists(),
    }
    # One cache-bust key for the in-browser engine: the Python bundle plus
    # worker.js itself, so a worker-only change is never served stale.
    bundle = json.loads((PY_DIR / "manifest.json").read_text(encoding="utf-8"))["sha256_12"]
    py_hash = hashlib.sha256(bundle.encode() + (SITE / "worker.js").read_bytes()).hexdigest()[:12]
    if data.get("media") != media or data.get("py") != py_hash or data.get("planner") != planner_meta:
        data["media"] = media
        data["planner"] = planner_meta
        # Cache-busts worker.js and py/orb.zip whenever the renderer changes.
        data["py"] = py_hash
        CUES_PATH.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"media: {media}")


def _write_if_changed(path: Path, text: str) -> None:
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def _spec_hash(spec: dict, renderer: str) -> str:
    key = json.dumps({"spec": spec, "renderer": renderer}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _encode_mp3(audio, path: Path, ffmpeg: str) -> None:
    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "s16le", "-ar", "48000", "-ac", "2", "-i", "pipe:0",
        "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact",
        "-codec:a", "libmp3lame", "-b:a", "128k", str(path),
    ]
    subprocess.run(cmd, input=pcm, check=True)


def _detail_key() -> str:
    """Identity of the detail stage: its weights plus the code that runs them."""
    h = hashlib.sha256()
    h.update(_file_sha(ROOT / "weights" / "detail.pt").encode())
    for p in (SRC / "orb" / "model" / "detail.py", SRC / "orb" / "render" / "detail.py"):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_planner():
    """(tokenizer, model) or None when torch or the weights are missing."""
    weights = ROOT / "weights" / "planner.pt"
    if not weights.exists():
        return None
    try:
        from orb.api import load_planner
        return load_planner(weights)
    except Exception as exc:  # torch missing or incompatible
        print(f"planner unavailable ({type(exc).__name__}: {exc}); skipping exact check")
        return None


PANNABLE = ("impact", "ui_pop", "click", "sub_drop", "thunder", "door_slam", "knock", "footsteps", "water_drop", "explosion")
CONTROL_KINDS = ("pan", "gain", "seed", "texture", "bed_db")


def _check_controls(item: dict, spec: dict) -> None:
    """Fail the build on a playground control that would do nothing."""
    n = len(spec["events"])
    for c in item.get("controls", []):
        kind = c["kind"]
        if kind not in CONTROL_KINDS:
            raise ValueError(f"{item['name']}: unknown control kind {kind!r}")
        idx = c.get("events", [])
        if any(not (0 <= i < n) for i in idx):
            raise ValueError(f"{item['name']}: {kind} control points past the events")
        if kind == "pan" and any(spec["events"][i]["role"] not in PANNABLE for i in idx):
            raise ValueError(f"{item['name']}: pan on a wide role has no effect")
        if kind == "bed_db" and spec["texture"] == "none":
            raise ValueError(f"{item['name']}: bed_db control on a cue without a bed")
    seen: set[int] = set()
    for m in item.get("marks", []):
        if any(not (0 <= i < n) for i in m["events"]):
            raise ValueError(f"{item['name']}: mark points past the events")
        if seen & set(m["events"]):
            raise ValueError(f"{item['name']}: an event is in two marks")
        seen |= set(m["events"])


def render_gallery(force: bool, ffmpeg: str) -> None:
    sys.dont_write_bytecode = True  # keep site/py free of __pycache__
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(PY_DIR))
    import numpy as np  # noqa: F401
    import orb_web
    from orb.api import dict_to_spec, render_spec, spec_to_dict
    from orb.eval.audit import audit_waveform, passes
    from orb.grammar import caption, score_from_spec

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    previous = {}
    if CUES_PATH.exists():
        try:
            data = json.loads(CUES_PATH.read_text(encoding="utf-8"))
            previous = {c["name"]: c for c in data.get("cues", []) + data.get("playground", []) if c}
        except (OSError, json.JSONDecodeError, KeyError):
            previous = {}

    planner = None
    planner_tried = False
    detail_model = None
    renderer = renderer_hash()
    detail_key = _detail_key() if (ROOT / "weights" / "detail.pt").exists() else None
    print(f"renderer: {renderer}  detail: {detail_key}")
    # Page-only metadata, copied as-is (never part of the cache key).
    meta_keys = ("title", "blurb", "tag", "controls", "marks", "group", "use")

    def build(item: dict, leveled: bool = False, detail: bool = False) -> dict:
        nonlocal planner, planner_tried, detail_model
        spec = spec_to_dict(dict_to_spec(item["spec"]))  # normalise through the real types
        _check_controls(item, spec)
        cap = caption(score_from_spec(dict_to_spec(spec), with_music=False))
        if detail and detail_key is None:
            raise SystemExit(f"{item['name']}: needs weights/detail.pt (the library renders with the detail pass)")
        h = _spec_hash(spec, renderer + (f"+detail:{detail_key}" if detail else "") + ("+level" if leveled else ""))
        mp3 = AUDIO_DIR / f"{item['name']}.mp3"
        old = previous.get(item["name"])
        if not force and old and old.get("hash") == h and mp3.exists():
            entry = dict(old)
            entry.pop("spectrogram", None)  # older manifests carried one; the page no longer uses it
            for k in meta_keys:
                entry.pop(k, None)
            entry.update({k: item[k] for k in meta_keys if k in item})
            print(f"  {item['name']:<18} cached")
            return entry

        if detail and detail_model is None:
            from orb.api import load_detail
            detail_model = load_detail(ROOT / "weights" / "detail.pt")
        audio = render_spec(spec, detail=detail_model if detail else None)
        if leveled:  # the same cap the in-browser render applies (orb_web.render)
            audio = orb_web.level(audio)
        audit = audit_waveform(audio, 48000)
        _encode_mp3(audio, mp3, ffmpeg)

        exact = None
        if not planner_tried:
            planner_tried = True
            planner = _load_planner()
        if planner is not None:
            from orb.api import plan
            planned = spec_to_dict(plan(cap, *planner))
            exact = planned == spec

        print(f"  {item['name']:<18} {mp3.stat().st_size / 1024:6.0f} KB  pass={passes(audit)}  planner_exact={exact}")
        return {
            "name": item["name"],
            **{k: item[k] for k in meta_keys if k in item},
            "caption": cap,
            "spec": spec,
            "hash": h,
            "audio": f"assets/audio/{mp3.name}",
            "duration": spec["duration"],
            "detail": bool(detail),
            "planner_exact": exact,
            "audit": {k: round(float(v), 6) for k, v in audit.items()},
            "passes": bool(passes(audit)),
            "peaks": orb_web.peaks(audio),
        }

    print("library:")
    cues = [build(item, leveled=True, detail=True) for item in LIBRARY + BLOCKS]
    print("playground:")
    playground = [build(item, leveled=True) for item in PLAYGROUND]
    from orb.grammar import NO_BED, quantize_gain

    by_name = {c["name"]: c for c in LIBRARY}
    prompts = []
    for name, label in PROMPT_CHIPS:
        spec = dict_to_spec(by_name[name]["spec"])
        if spec.texture != NO_BED:
            for e in spec.events:
                e.gain_db = quantize_gain(e.gain_db)
            spec.bed_db = None
        prompts.append({"name": name, "label": label,
                        "caption": caption(score_from_spec(spec, with_music=False)),
                        "planned": spec.texture != NO_BED})
    payload = {
        "version": "0.1.0",
        "sample_rate": 48000,
        "renderer": renderer,
        "detail": detail_key,
        "groups": LIBRARY_GROUPS + ["Building blocks"],
        "prompts": prompts,
        "cues": cues,
        "playground": playground,
        "media": {},
        "py": "",
        "planner": None,
    }
    _write_if_changed(CUES_PATH, json.dumps(payload, indent=1) + "\n")

    keep = {f"{c['name']}.mp3" for c in cues + playground}
    for stale in AUDIO_DIR.glob("*.mp3"):
        if stale.name not in keep:
            stale.unlink()
    total = sum(p.stat().st_size for p in AUDIO_DIR.glob("*.mp3"))
    print(f"wrote {CUES_PATH.relative_to(ROOT)}; audio total {total / 1024:.0f} KB")
    failed = [c["name"] for c in cues + playground if not c["passes"]]
    if failed:
        raise SystemExit(f"audit failed: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-render every cue")
    parser.add_argument("--py-only", action="store_true", help="only refresh site/py/orb.zip (stdlib only)")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    args = parser.parse_args()

    sync_python()
    if not args.py_only:
        export_planner(args.force)
        render_gallery(args.force, args.ffmpeg)
    sync_media_flags()
    if args.py_only:
        return
    nojekyll = SITE / ".nojekyll"
    if not nojekyll.exists():
        nojekyll.write_text("")


if __name__ == "__main__":
    main()
