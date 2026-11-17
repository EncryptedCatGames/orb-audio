"""Free-text prompts for planner training (v0.4).

The planner first learned one exact caption grammar. People type something
else: "rain on a tin roof, then thunder", "glas braking in a kitchen", "a soft
notification ding". This module makes (free-text prompt, Spec) pairs. It
samples an *intent* (which sounds, how many, soft or loud, short or long, in
what order, where, what mood, how long), says it in one of many ways (synonyms,
word orders, filler, counts, typos, unsupported distractors such as a dog), and
lays it out with one deterministic composer, `compose`. The planner learns
"prompt -> intent" from the variety and "intent -> Spec" from the composer.

Only training data comes from here; nothing renders differently. The held-out
evaluation set (`data/freetext_eval.jsonl`) was written by hand, separately,
without these templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grammar import EVENT_STYLE, MAX_EVENTS, NO_BED, Spec, SpecEvent, pan_value, quantize_gain, quantize_time

# -- the lexicon ---------------------------------------------------------------
# role -> phrases for one occurrence. "|" separates a plural form used for counts
# ("two knocks"); roles without one say counts as "... twice" / "x3".
ROLE_PHRASES: dict[str, list[str]] = {
    "whoosh": ["whoosh|whooshes", "swoosh|swooshes", "swish|swishes", "woosh|wooshes", "fast whoosh|fast whooshes",
               "air whoosh", "swipe|swipes", "pass-by whoosh", "whooshing sound", "zip|zips", "quick swoosh transition"],
    "impact": ["impact|impacts", "hit|hits", "cinematic hit|cinematic hits", "boom|booms", "trailer hit|trailer hits",
               "heavy impact|heavy impacts", "slam hit", "braam", "big hit|big hits", "thud|thuds", "low boom|low booms"],
    "ui_pop": ["pop|pops", "ui pop|ui pops", "bubble pop|bubble pops", "blip|blips", "bloop|bloops",
               "coin pickup|coin pickups", "item pickup", "like button pop", "soft pop|soft pops", "plop|plops",
               "button pop", "collect sound", "power-up pickup"],
    "riser": ["riser|risers", "tension riser", "build-up", "rising tension", "uplifter", "suspense build",
              "rising sweep", "tension build", "risers going up"],
    "swell": ["swell|swells", "shimmer swell", "shimmering swell", "magic swell", "sparkle swell", "reverse swell",
              "airy swell", "shimmer", "glowing swell"],
    "click": ["click|clicks", "tick|ticks", "ui click|ui clicks", "button click|button clicks", "tap|taps",
              "toggle click", "switch click", "menu click|menu clicks", "tiny tick"],
    "bell": ["notification ding|notification dings", "ding|dings", "chime|chimes", "bell|bells", "ping|pings",
             "message chime", "success chime", "alert chime", "level up chime", "doorbell-like ding", "ding sound",
             "new message sound", "achievement chime", "ding-dong"],
    "sub_drop": ["sub drop|sub drops", "bass drop|bass drops", "deep drop", "808 drop", "low end drop", "sub boom",
                 "bass hit", "rumbling drop", "deep bass drop"],
    "glitch": ["glitch|glitches", "digital glitch", "error glitch", "data glitch", "stutter", "error sound",
               "broken signal", "digital stutter", "bitcrush glitch", "malfunction sound"],
    "noise_sweep": ["noise sweep|noise sweeps", "white noise sweep", "filter sweep", "sweep|sweeps",
                    "transition sweep", "noise riser", "airy sweep", "wind-like sweep"],
    "thunder": ["thunder", "thunderclap|thunderclaps", "rumble of thunder", "rolling thunder", "thunder crack",
                "distant storm rumble", "lightning crack", "thunderstorm rumble", "thunder boom|thunder booms"],
    "door_slam": ["door slam|door slams", "slamming door", "door slamming shut", "door banging shut", "slammed door",
                  "heavy door closing", "door shutting hard", "car door slam", "door bang|door bangs"],
    "knock": ["knock|knocks", "knock on the door", "knocking", "door knock|door knocks", "rap on the door",
              "knuckles on wood", "someone knocking", "tap on a door", "knock knock"],
    "footsteps": ["footsteps", "steps", "someone walking", "walking", "foot steps", "boots walking",
                  "running footsteps", "feet on gravel", "person walking by", "heels clicking", "walking on a wood floor",
                  "pacing", "someone approaching"],
    "water_drop": ["water drop|water drops", "drip|drips", "droplet|droplets", "dripping", "drop of water",
                   "leaky tap drip", "cave drip|cave drips", "plink of water", "dripping faucet"],
    "explosion": ["explosion|explosions", "blast|blasts", "big explosion|big explosions", "kaboom", "bomb going off",
                  "detonation|detonations", "grenade blast", "huge boom", "explosive blast"],
    "bird_chirp": ["birds chirping", "bird chirping", "birdsong", "chirping birds", "birds singing", "songbirds",
                   "a bird singing", "tweeting birds", "seagulls", "gulls", "sparrows", "morning birds", "bird calls",
                   "a little bird", "warbling birds", "chirps"],
    "crickets": ["crickets", "chirping crickets", "cicadas", "insects at night", "cricket chirps", "bugs chirping",
                 "night insects", "grasshoppers"],
    "wave_crash": ["waves|waves", "wave crashing", "crashing waves", "surf", "breaking waves", "waves on the shore",
                   "ocean waves", "sea waves", "waves rolling in", "the tide", "waves hitting rocks"],
    "wind_gust": ["gust of wind|gusts of wind", "wind gust|wind gusts", "howling wind", "whistling wind",
                  "a strong gust", "windy blast", "wind blowing", "rush of wind", "wind howling"],
    "splash": ["splash|splashes", "water splash|water splashes", "something falling in water", "splashing",
               "jump into a pool", "plunge", "big splash", "stone hitting water", "kerplunk"],
    "frog": ["frog|frogs", "frogs croaking", "croaking", "toad|toads", "bullfrog|bullfrogs", "ribbit",
             "frog croak|frog croaks", "pond frogs"],
    "owl": ["owl|owls", "owl hooting", "hooting owl", "hoot|hoots", "owl hoots", "an owl calling", "night owl",
            "barn owl"],
    "glass_break": ["glass breaking", "breaking glass", "glass shattering", "shattering glass", "window smash",
                    "smashed glass", "glass crash", "a cup breaking", "bottle smashing", "plate shattering",
                    "window breaking", "glass smash|glass smashes"],
    "clock_tick": ["clock ticking", "ticking clock", "tick tock", "tick-tock", "clock tick", "wall clock",
                   "stopwatch ticking", "ticking", "grandfather clock ticking", "timer ticking"],
    "phone_ring": ["phone ringing", "ringing phone", "telephone ring", "phone ring", "old phone ringing",
                   "cell phone ringtone", "landline ringing", "ringtone", "incoming call", "office phone ringing"],
    "keyboard_typing": ["typing", "keyboard typing", "typing on a keyboard", "keyboard clacking", "keystrokes",
                        "someone typing", "mechanical keyboard", "tapping keys", "typing on a laptop",
                        "clicky keyboard"],
    "mouse_click": ["mouse click|mouse clicks", "computer mouse click|computer mouse clicks", "clicking a mouse",
                    "mouse button click", "double click", "mouse clicking"],
    "camera_shutter": ["camera shutter", "camera click|camera clicks", "shutter click", "taking a photo",
                       "photo snap", "camera snap", "shutter sound", "dslr shutter", "picture being taken",
                       "snapshot"],
    "paper_rustle": ["paper rustling", "rustling paper", "paper crumpling", "crumpling paper", "page turn",
                     "turning pages", "flipping through papers", "shuffling papers", "crinkling paper",
                     "newspaper rustle"],
    "cork_pop": ["cork pop|cork pops", "champagne pop", "popping a cork", "champagne cork", "bottle uncorking",
                 "cork popping", "wine cork", "celebration cork pop"],
    "coin_drop": ["coins dropping", "coin drop|coin drops", "coins falling", "jingling coins", "loose change",
                  "coins jingling", "money clinking", "coins on a table", "dropping change", "coin clink"],
    "metal_clang": ["metal clang|metal clangs", "clang|clangs", "pipe hit", "metal hit", "pan clanging",
                    "metal clank", "anvil strike", "dropping a wrench", "clanging metal", "steel clang"],
    "wood_creak": ["wood creak|wood creaks", "creaking", "creaky floorboard", "creaking door", "creaky door",
                   "floor creaking", "old door creaking", "creaking wood", "stairs creaking", "creak"],
    "heartbeat": ["heartbeat", "heart beating", "heart pounding", "pounding heart", "pulse", "beating heart",
                  "thumping heartbeat", "heart thumping"],
    "applause": ["applause", "clapping", "audience clapping", "crowd applause", "people clapping", "cheering applause",
                 "round of applause", "hand claps", "audience applause", "ovation"],
    "pouring_water": ["pouring water", "water pouring", "pouring a drink", "filling a glass", "pouring tea",
                      "pouring coffee", "glass being filled", "water poured into a cup", "pouring juice"],
    "can_opening": ["can opening", "soda can opening", "opening a soda", "can crack", "can of soda", "pop tab",
                    "fizzy can", "beer can opening", "opening a can"],
    "clock_alarm": ["alarm clock", "alarm going off", "alarm ringing", "wake up alarm", "beeping alarm",
                    "alarm beeping", "clock alarm", "timer beeping", "kitchen timer"],
    "punch": ["punch|punches", "hit to the face", "fist hit", "body blow|body blows", "punching", "smack|smacks",
              "fight punch|fight punches", "jab|jabs", "kick|kicks"],
    "sword_clash": ["sword clash|sword clashes", "swords clashing", "blades clashing", "clashing swords",
                    "sword hit|sword hits", "steel on steel", "metal blades", "sword strike|sword strikes",
                    "blade clang"],
    "gunshot": ["gunshot|gunshots", "gun shot|gun shots", "pistol shot|pistol shots", "rifle shot|rifle shots",
                "shot|shots", "gunfire", "bang of a gun", "firing a gun", "handgun shot|handgun shots"],
    "car_horn": ["car horn|car horns", "honk|honks", "horn honking", "car honking", "beep of a car horn",
                 "taxi horn", "truck horn", "honking horn", "horn blast"],
    "siren": ["siren", "police siren", "ambulance siren", "fire truck siren", "sirens", "emergency siren",
              "wailing siren", "cop car siren"],
    "engine": ["engine", "car engine", "engine revving", "motor running", "engine idling", "truck engine",
               "motorcycle engine", "revving engine", "engine starting", "car driving by"],
    "church_bell": ["church bell|church bells", "church bells ringing", "bell tower", "tolling bell", "big bell tolling",
                    "cathedral bells", "chapel bell", "bell ringing in the distance", "temple bell"],
    "fireworks": ["fireworks", "firework|fireworks", "fireworks exploding", "firecrackers", "fireworks show",
                  "new year fireworks", "rockets bursting", "crackling fireworks"],
}

# Impulsive roles: a count means that many events in a row ("three gunshots").
COUNTABLE = {
    "whoosh", "impact", "ui_pop", "click", "bell", "sub_drop", "glitch", "thunder", "door_slam", "knock",
    "water_drop", "explosion", "splash", "glass_break", "mouse_click", "camera_shutter", "cork_pop", "coin_drop",
    "metal_clang", "punch", "sword_clash", "gunshot", "car_horn", "can_opening", "wave_crash", "wind_gust",
    "church_bell", "frog", "owl", "wood_creak", "noise_sweep", "swell",
}
# Gap from one event's start to the next of the same role in a counted run.
SPACING = {
    "whoosh": 0.6, "impact": 1.2, "ui_pop": 0.25, "click": 0.3, "bell": 0.8, "sub_drop": 1.5, "glitch": 0.35,
    "thunder": 4.5, "door_slam": 1.4, "knock": 0.9, "water_drop": 0.6, "explosion": 2.5, "splash": 1.2,
    "glass_break": 1.8, "mouse_click": 0.45, "camera_shutter": 0.7, "cork_pop": 1.0, "coin_drop": 1.2,
    "metal_clang": 1.2, "punch": 0.5, "sword_clash": 0.8, "gunshot": 0.7, "car_horn": 1.3, "can_opening": 1.2,
    "wave_crash": 3.5, "wind_gust": 3.0, "church_bell": 2.5, "frog": 2.0, "owl": 2.6, "wood_creak": 1.8,
    "noise_sweep": 1.6, "swell": 1.6,
}
# How long after an event starts the next thing in a sequence may begin.
HOLD = {
    "thunder": 3.0, "explosion": 1.8, "wave_crash": 3.0, "wind_gust": 2.5, "church_bell": 3.0, "riser": None,
    "swell": None, "fireworks": 3.0, "glass_break": 1.0, "metal_clang": 1.0, "sub_drop": 0.8, "impact": 0.6,
    "door_slam": 0.7, "splash": 0.8, "car_horn": 1.0, "coin_drop": 1.0, "cork_pop": 0.6,
}
# Long, continuous sources that layer under the rest instead of queueing.
CONTINUOUS = {
    "footsteps", "bird_chirp", "crickets", "keyboard_typing", "clock_tick", "phone_ring", "heartbeat", "applause",
    "pouring_water", "paper_rustle", "engine", "siren", "clock_alarm", "fireworks",
}
# Things a planner should leave out: a voice, music, a vocal animal.
UNSUPPORTED = [
    "a dog barking", "dogs barking", "a cat meowing", "people talking", "someone laughing", "a baby crying",
    "a rooster", "a man shouting", "kids playing", "background music", "a piano melody", "someone singing",
    "a crow cawing", "cows mooing", "a woman screaming", "a crowd chatting", "a guitar strum", "a voice saying hello",
    "sheep", "a horse neighing", "a parrot", "chickens clucking", "a lion roar", "a sneeze", "someone coughing",
]
UNSUPPORTED_FRAMES = ["{x}", "no {x}", "without {x}", "(no {x})", "but no {x}", "minus {x}", "{x} in the distance"]

# -- places, moods, modifiers ----------------------------------------------------

TEXTURE_PLACES: dict[str, list[str]] = {
    "room_cold": ["in a room", "indoors", "in an empty room", "in a hallway", "in a quiet house", "inside",
                  "in a basement", "in a warehouse", "in a corridor", "in an apartment", "in a big hall"],
    "wind": ["on a windy hill", "in the wind", "on a mountain", "in a desert", "on a cliff", "in a blizzard",
             "on a windy day", "outside in strong wind", "on a rooftop in the wind"],
    "city_hum": ["in the city", "in a city", "downtown", "in a busy city", "in an urban area", "in the subway",
                 "in a mall", "in a cafe", "in a restaurant", "in a train station", "in a busy office"],
    "vinyl": ["on vinyl", "lo-fi", "with vinyl crackle", "like an old record", "retro", "on an old record player",
              "nostalgic lo-fi", "old film style"],
    "drone": ["in a dark void", "in a cave", "underground", "in a dungeon", "in a haunted house", "in a crypt",
              "in an abandoned building", "with a dark drone", "in a horror scene", "in a graveyard"],
    "rain": ["in the rain", "during a rainstorm", "on a rainy day", "with rain falling", "in a downpour",
             "with heavy rain", "on a rainy night", "under the rain", "in a storm"],
    "fire": ["by a campfire", "by the fireplace", "near a crackling fire", "by a bonfire", "next to a fire",
             "around a campfire", "by the hearth", "with a fire crackling"],
    "sci_fi": ["on a spaceship", "in space", "in a sci-fi lab", "aboard a starship", "on a space station",
               "in a futuristic city", "in a cyberpunk alley", "in a reactor room", "in an alien base"],
    "ocean": ["at the beach", "on the beach", "by the sea", "at the seaside", "on the shore", "on a boat",
              "by the ocean", "at the coast", "on a pier", "at the harbor"],
    "stream": ["by a stream", "near a river", "by a brook", "next to a creek", "by a waterfall", "at the riverside",
               "near a babbling brook", "by running water"],
    "forest": ["in the forest", "in the woods", "in a park", "in a garden", "in the jungle", "in a meadow",
               "in the countryside", "in a forest clearing", "at dawn in the woods", "under the trees"],
    "night": ["at night", "on a summer night", "in a swamp", "by a pond at night", "at a campsite at night",
              "late at night outside", "on a warm night", "in the dark countryside"],
    "traffic": ["on a busy street", "in traffic", "at an intersection", "by the highway", "on a city street",
                "at a crosswalk", "by a busy road", "in rush hour traffic", "on the road"],
}
TEXTURE_BARE = {  # the place said as a bed on its own
    "room_cold": ["room tone", "quiet room tone"], "wind": ["wind", "howling wind ambience"],
    "city_hum": ["city ambience", "city hum"], "vinyl": ["vinyl crackle", "record crackle"],
    "drone": ["dark drone", "ominous drone"], "rain": ["rain", "rainfall", "rain ambience"],
    "fire": ["crackling fire", "fireplace crackle", "campfire"], "sci_fi": ["sci-fi ambience", "spaceship hum"],
    "ocean": ["ocean ambience", "the sea"], "stream": ["a babbling brook", "running stream"],
    "forest": ["forest ambience", "woodland ambience"], "night": ["night ambience", "night sounds"],
    "traffic": ["traffic", "street traffic", "traffic noise"],
}
MOOD_WORDS: dict[str, list[str]] = {
    "calm": ["calm", "relaxing", "peaceful", "gentle", "soothing", "quiet", "serene", "chill", "tranquil"],
    "warm": ["warm", "cozy", "cosy", "friendly", "homey", "comforting"],
    "playful": ["playful", "fun", "cartoony", "silly", "bouncy", "cute", "whimsical"],
    "hopeful": ["hopeful", "uplifting", "inspiring", "optimistic", "triumphant"],
    "bright": ["bright", "cheerful", "happy", "sunny", "sparkly", "upbeat"],
    "tense": ["tense", "suspenseful", "nervous", "anxious", "thriller", "urgent", "stressful"],
    "dark": ["dark", "ominous", "gloomy", "sad", "grim", "menacing", "brooding"],
    "eerie": ["eerie", "creepy", "spooky", "scary", "haunting", "unsettling", "mysterious", "ghostly"],
    "neutral": ["neutral", "plain", "simple", "realistic", "natural", "clean"],
    "intense": ["intense", "epic", "action-packed", "aggressive", "dramatic", "explosive", "chaotic"],
}
MODS = {
    "soft": (["soft", "quiet", "gentle", "faint", "subtle", "light", "muffled", "low-volume"], -8.0),
    "distant": (["distant", "far away", "faraway", "far off", "in the distance", "far-off"], -10.0),
    "loud": (["loud", "huge", "massive", "big", "powerful", "booming", "very loud", "heavy"], 4.0),
    "close": (["close", "up close", "close-up", "right next to you", "nearby"], 2.0),
}
LENGTH_MODS = {"quick": (["quick", "short", "brief", "fast", "snappy", "tiny"], 0.6),
               "long": (["long", "slow", "lingering", "extended", "drawn-out", "sustained"], 1.6)}
PAN_MODS = {"left": (["on the left", "from the left", "left side", "panned left"], -0.75),
            "right": (["on the right", "from the right", "right side", "panned right"], 0.75)}
NUMBER_WORDS = {2: ["two", "2", "a couple of", "double"], 3: ["three", "3", "a few"], 4: ["four", "4", "several"],
                5: ["five", "5", "lots of"]}
TIMES_WORDS = {2: ["twice", "two times", "x2", "2 times"], 3: ["three times", "x3", "3 times", "thrice"],
               4: ["four times", "x4", "4 times"], 5: ["five times", "x5", "5 times"]}

# Where a sound lives when the prompt names no place, and its usual mood.
HOME = {
    "wave_crash": "ocean", "splash": "ocean", "bird_chirp": "forest", "crickets": "night", "frog": "night",
    "owl": "night", "thunder": "rain", "wind_gust": "wind", "car_horn": "traffic", "siren": "traffic",
    "engine": "traffic", "church_bell": "city_hum", "fireworks": "city_hum", "applause": "room_cold",
    "keyboard_typing": "room_cold", "phone_ring": "room_cold", "water_drop": "drone", "explosion": "drone",
    "sword_clash": "wind", "gunshot": "drone",
}
DRY = {"whoosh", "impact", "ui_pop", "riser", "swell", "click", "bell", "sub_drop", "glitch", "noise_sweep",
       "camera_shutter", "mouse_click", "coin_drop"}
ROLE_MOOD = {
    "thunder": "dark", "explosion": "intense", "gunshot": "intense", "sword_clash": "intense", "punch": "intense",
    "siren": "tense", "heartbeat": "tense", "riser": "tense", "glitch": "tense", "clock_tick": "tense",
    "owl": "eerie", "wood_creak": "eerie", "bird_chirp": "calm", "crickets": "calm", "wave_crash": "calm",
    "frog": "calm", "fireworks": "bright", "applause": "hopeful", "cork_pop": "playful", "ui_pop": "playful",
    "bell": "bright", "swell": "hopeful", "coin_drop": "playful", "sub_drop": "dark", "impact": "intense",
    "church_bell": "calm", "door_slam": "tense", "knock": "tense",
}
TEXTURE_MOOD = {"rain": "calm", "fire": "warm", "drone": "dark", "forest": "calm", "ocean": "calm", "night": "calm",
                "stream": "calm", "sci_fi": "tense", "vinyl": "warm"}

# Whole places said without listing sounds: a canonical scene.
SCENE_PHRASES: dict[str, tuple[list[str], str, str, list[tuple[str, int, str | None]]]] = {
    # name -> (phrases, texture, mood, [(role, count, modifier)])
    "beach": (["a day at the beach", "beach ambience", "a sunny beach", "seaside scene", "beach with seagulls"],
              "ocean", "calm", [("wave_crash", 1, None), ("bird_chirp", 1, None)]),
    "forest_morning": (["a forest morning", "morning in the woods", "dawn chorus", "sunrise in the forest",
                        "spring morning in the forest"], "forest", "calm", [("bird_chirp", 2, None), ("wind_gust", 1, "soft")]),
    "night_camp": (["camping at night", "a night campsite", "night by the lake", "summer night outdoors",
                    "night in the countryside"], "night", "calm", [("crickets", 1, None), ("owl", 1, "distant")]),
    "kitchen": (["a busy kitchen", "kitchen sounds", "someone cooking", "a cafe kitchen", "making breakfast"],
                "room_cold", "warm", [("pouring_water", 1, None), ("metal_clang", 1, "soft"), ("can_opening", 1, None)]),
    "office": (["an office", "office ambience", "a busy office", "working at a desk", "an open-plan office"],
               "room_cold", "neutral", [("keyboard_typing", 1, None), ("mouse_click", 2, None), ("phone_ring", 1, "distant")]),
    "city_street": (["a city street", "a busy street", "downtown traffic", "rush hour", "a street corner"],
                    "traffic", "neutral", [("engine", 1, None), ("car_horn", 1, "distant"), ("footsteps", 1, None)]),
    "sword_fight": (["a sword fight", "a sword duel", "knights fighting", "a medieval duel", "a samurai duel"],
                    "wind", "intense", [("sword_clash", 3, None), ("whoosh", 1, None), ("punch", 1, None)]),
    "haunted_house": (["a haunted house", "a creepy old house", "a ghost story", "a horror hallway",
                       "an abandoned mansion"], "drone", "eerie", [("wood_creak", 1, None), ("door_slam", 1, "distant"), ("heartbeat", 1, None)]),
    "storm": (["a thunderstorm", "a storm", "a stormy night", "a rainstorm", "storm rolling in"],
              "rain", "dark", [("thunder", 2, None)]),
    "battle": (["a battle", "a war zone", "a battlefield", "a firefight", "an action scene"],
               "fire", "intense", [("explosion", 2, None), ("gunshot", 3, None)]),
    "celebration": (["a celebration", "a party", "new year's eve", "a birthday party", "a victory celebration"],
                    "city_hum", "bright", [("cork_pop", 1, None), ("applause", 1, None), ("fireworks", 1, None)]),
    "cave": (["a cave", "a dripping cave", "inside a cavern", "an underground cave", "a mine shaft"],
             "drone", "eerie", [("water_drop", 3, None)]),
    "game_win": (["a level up", "a game win", "victory jingle", "achievement unlocked", "level complete"],
                 NO_BED, "bright", [("riser", 1, "quick"), ("ui_pop", 2, None), ("bell", 1, None)]),
    "game_over": (["a game over", "losing a life", "game over sound", "you died", "mission failed"],
                  NO_BED, "dark", [("glitch", 1, None), ("impact", 1, None), ("sub_drop", 1, None)]),
    "trailer": (["a trailer hit", "a movie trailer intro", "a cinematic intro", "a trailer opener", "an epic intro"],
                NO_BED, "intense", [("riser", 1, None), ("impact", 1, "loud"), ("sub_drop", 1, None)]),
    "transition": (["a transition", "a scene transition", "a slide transition", "a swipe transition",
                    "a video transition"], NO_BED, "neutral", [("whoosh", 1, None), ("noise_sweep", 1, "soft")]),
    "notification": (["a notification", "a notification sound", "a message alert", "an app alert",
                      "new message"], NO_BED, "bright", [("bell", 1, None)]),
    "western": (["a western shootout", "a saloon shootout", "a wild west duel", "a cowboy standoff",
                 "high noon in a western town"], "wind", "tense", [("gunshot", 2, None), ("door_slam", 1, None), ("footsteps", 1, None)]),
    "construction": (["a construction site", "building work", "workers hammering", "a building site"],
                     "city_hum", "neutral", [("metal_clang", 3, None), ("engine", 1, "distant")]),
    "library": (["a quiet library", "a study room", "a reading room", "studying in a library"],
                "room_cold", "calm", [("paper_rustle", 1, "soft"), ("footsteps", 1, "distant"), ("clock_tick", 1, "soft")]),
    "forge": (["a blacksmith forge", "a blacksmith at work", "forging a sword", "an anvil workshop"],
              "fire", "warm", [("metal_clang", 4, None)]),
    "heist": (["a bank heist", "a heist getaway", "a getaway scene", "a robbery"],
              "traffic", "tense", [("glass_break", 1, None), ("siren", 1, None), ("engine", 1, None)]),
    "stadium": (["a stadium crowd", "a sports arena", "a goal scored", "a touchdown", "the winning point"],
                "city_hum", "intense", [("applause", 1, "loud"), ("whoosh", 1, None)]),
    "cabin_storm": (["a cabin in a storm", "a cozy cabin during a storm", "a stormy night in a cabin"],
                    "rain", "warm", [("thunder", 1, "distant"), ("wood_creak", 1, None)]),
    "harbor": (["a harbor", "a fishing port", "the docks", "a marina"],
               "ocean", "calm", [("wave_crash", 1, "soft"), ("bird_chirp", 1, None), ("metal_clang", 1, "distant")]),
    "campfire": (["a campfire night", "sitting by the campfire", "roasting marshmallows", "a bonfire at night"],
                 "fire", "warm", [("crickets", 1, None), ("owl", 1, "distant")]),
    "morning": (["a morning routine", "waking up", "a monday morning", "the start of the day"],
                "room_cold", "warm", [("clock_alarm", 1, None), ("footsteps", 1, None), ("pouring_water", 1, None)]),
    "boxing": (["a boxing match", "a fist fight", "a bar fight", "a brawl", "a kung fu fight"],
               "room_cold", "intense", [("punch", 4, None), ("applause", 1, "distant")]),
    "logo": (["a logo reveal", "a logo sting", "an intro sting", "a brand sting", "a youtube intro"],
             NO_BED, "hopeful", [("whoosh", 1, None), ("impact", 1, None), ("swell", 1, None)]),
    "email": (["an email arrived", "you've got mail", "an inbox notification", "a new email"],
              NO_BED, "bright", [("bell", 1, "soft")]),
    "treasure": (["a treasure chest opening", "opening a chest in a game", "loot drop", "finding treasure"],
                 NO_BED, "playful", [("wood_creak", 1, "quick"), ("swell", 1, None), ("ui_pop", 3, None)]),
    "pause": (["a pause menu", "a game pause", "opening the menu", "menu open"],
              NO_BED, "neutral", [("click", 1, None), ("noise_sweep", 1, "quick")]),
    "purchase": (["a purchase", "a payment success", "a cash register", "ka-ching", "a sale", "checkout complete"],
                 NO_BED, "bright", [("coin_drop", 1, None), ("bell", 1, None)]),
    "onboarding": (["an onboarding finish", "a task done", "a success state", "a confirmation", "a toast message"],
                   NO_BED, "bright", [("ui_pop", 1, None), ("bell", 1, "soft")]),
    "jump": (["a jump in a platformer", "a cartoon jump", "a character jumping", "a bounce"],
             NO_BED, "playful", [("ui_pop", 1, None), ("whoosh", 1, "quick")]),
    "error": (["an error sound", "an error", "a failed action", "wrong answer", "access denied"],
              NO_BED, "tense", [("glitch", 1, None), ("click", 1, None)]),
}

FILLERS = ["", "", "", "", "imagine ", "here's what i need: ", "we need ", "our app needs ", "sound of ", "the sound of ", "sfx: ", "sound effect of ", "i need ", "give me ",
           "make ", "can you make ", "audio of ", "a clip of ", "generate ", "create ", "i want ", "foley: ",
           "need a sound: ", "please make ", "something like ", "just "]
SUFFIXES = ["", "", "", "", "", " sound", " sfx", " sound effect", " for my game", " for a video",
            " for an app", " for a podcast intro", " please", " for a short film", " for my youtube video",
            " for a trailer", " ambience", " audio", "!", ".", "..."]
SEQ_JOIN = [" then ", ", then ", " followed by ", ", and then ", ". then ", " and after that ", " -> ", ", after that ",
            " then later ", ", next "]
PAR_JOIN = [" and ", ", ", " with ", " plus ", " & ", " along with ", " over ", " + ", " while ", " as ", " when "]


# Sentence forms ("a door slams", "the phone rings"), so prompts written as a
# little story ("when the character walks in, a door slams") read the same.
_VERB_FORMS = {
    "whoosh": ["something whooshes past", "a car zooming past", "an arrow flying by"],
    "impact": ["something hits hard", "a heavy object lands", "the logo slams in"],
    "ui_pop": ["a bubble pops", "an item gets picked up", "the player picks up a coin"],
    "riser": ["the tension builds", "it builds up", "the music-free build rises"],
    "swell": ["a shimmer swells up", "a magical sparkle swells"],
    "click": ["a button gets clicked", "a switch flips", "a toggle clicks"],
    "bell": ["a bell rings", "a chime plays", "an email arrives", "a message comes in", "a task completes"],
    "sub_drop": ["the bass drops", "a deep boom hits"],
    "glitch": ["the screen glitches", "the signal breaks up", "a robot malfunctions", "a glitchy buzz"],
    "noise_sweep": ["a noise sweeps across", "a filter sweeps up"],
    "thunder": ["thunder rolls", "thunder rumbles overhead", "lightning strikes", "a storm rumbles overhead"],
    "door_slam": ["a door slams", "someone slams the door", "the door bangs shut", "a wooden door slams"],
    "knock": ["someone knocks", "someone knocks on the door", "a knock at the door", "the mailman knocks"],
    "footsteps": ["someone walks in", "a character walks", "someone runs down the hall", "footsteps approach",
                  "someone walks down the stairs"],
    "water_drop": ["water drips", "a tap drips", "drops fall from the ceiling"],
    "explosion": ["something explodes", "a bomb goes off", "a cannon fires", "a cannon blast", "a car explodes"],
    "bird_chirp": ["birds sing", "a bird sings", "birds are chirping", "the birds wake up"],
    "crickets": ["crickets chirp", "the crickets are out"],
    "wave_crash": ["waves crash", "the waves roll in", "waves break on the rocks"],
    "wind_gust": ["the wind howls", "the wind picks up", "a gust blows through"],
    "splash": ["something splashes", "a rock is thrown into the water", "someone dives in", "a fish jumps"],
    "frog": ["frogs croak", "a frog croaks"],
    "owl": ["an owl hoots", "an owl calls out"],
    "glass_break": ["glass shatters", "a window shatters", "a glass falls and breaks", "the vase shatters",
                    "a mug breaks on the floor"],
    "clock_tick": ["a clock ticks", "the clock is ticking", "time ticks away"],
    "phone_ring": ["the phone rings", "a phone starts ringing", "phones ringing", "the phone rings and rings"],
    "keyboard_typing": ["someone types", "a hacker types fast", "someone types an email", "the password is typed"],
    "mouse_click": ["someone clicks the mouse", "a mouse clicks", "the dev clicks"],
    "camera_shutter": ["a photo is taken", "someone snaps a picture", "a camera snaps"],
    "paper_rustle": ["pages turn", "someone flips a page", "papers rustle", "a letter is unfolded"],
    "cork_pop": ["a cork pops", "the champagne pops", "someone pops the champagne"],
    "coin_drop": ["coins drop", "a coin falls", "a cash register opens", "change spills on the counter"],
    "metal_clang": ["metal clangs", "a hammer hits an anvil", "a pipe falls", "a pot drops"],
    "wood_creak": ["a floorboard creaks", "the door creaks open", "a squeaky door opens", "the stairs creak",
                   "a rocking chair creaks"],
    "heartbeat": ["a heart pounds", "the heart races", "my heart beats fast"],
    "applause": ["the crowd claps", "everyone claps", "the audience applauds", "the stadium cheers"],
    "pouring_water": ["water pours", "someone pours a drink", "a glass is filled", "coffee is poured into a mug"],
    "can_opening": ["a can cracks open", "someone opens a soda", "a soda can pops open"],
    "clock_alarm": ["the alarm goes off", "an alarm clock rings", "the timer goes off"],
    "punch": ["someone gets punched", "a fist connects", "a punch lands", "a guy gets hit"],
    "sword_clash": ["swords clash", "blades meet", "a sword is drawn and clashes"],
    "gunshot": ["a gun fires", "someone fires a pistol", "a shot rings out", "shots ring out"],
    "car_horn": ["a car honks", "a driver honks", "a taxi honks"],
    "siren": ["a siren wails", "sirens wail", "the police arrive", "an ambulance speeds by"],
    "engine": ["an engine starts", "a car speeds by", "a car pulls away", "an engine roars", "a car zooms off"],
    "church_bell": ["a church bell rings", "the bells toll", "the bell tower chimes"],
    "fireworks": ["fireworks go off", "fireworks light up the sky", "rockets burst overhead"],
}
for _r, _v in _VERB_FORMS.items():
    ROLE_PHRASES[_r] = ROLE_PHRASES[_r] + _v

# Places that say a bed in other words.
_MORE_PLACES = {
    "room_cold": ["in a library", "in a cabin", "in a classroom", "in a lobby", "in a bank", "in an elevator",
                  "in a garage", "in a studio", "at home"],
    "wind": ["in a canyon", "on a mountain top", "on the tundra", "in a western town", "on an open plain"],
    "city_hum": ["in a stadium", "at a construction site", "in a saloon", "at a market", "in a hotel lobby",
                 "at an arcade", "at a festival"],
    "fire": ["in a blacksmith forge", "by a wood stove", "in a cabin by the fire"],
    "drone": ["in a bunker", "in a dark corridor", "in a sewer", "in a tomb"],
    "night": ["at dusk", "at midnight outside", "on a dark road at night"],
    "forest": ["at a picnic", "in a backyard", "on a hiking trail", "in a rainforest"],
    "ocean": ["on a ship deck", "at a lighthouse", "on a sailboat"],
    "rain": ["in a cabin during a storm", "on a tin roof in the rain", "at a bus stop in the rain"],
    "sci_fi": ["in a cockpit", "inside a robot", "in a hacker den"],
}
for _t, _v in _MORE_PLACES.items():
    TEXTURE_PLACES[_t] = TEXTURE_PLACES[_t] + _v

# -- the intent and the composer -------------------------------------------------


@dataclass
class Item:
    role: str
    count: int = 1
    level: str | None = None  # soft / distant / loud / close
    length: str | None = None  # quick / long
    side: str | None = None  # left / right


@dataclass
class Intent:
    steps: list[list[Item]]  # sequential steps; items inside a step overlap
    texture: str  # one of TEXTURES or NO_BED
    mood: str
    duration: float | None = None
    words: dict = field(default_factory=dict)  # bookkeeping for the phrasing


def _cont(role: str) -> bool:
    return role in CONTINUOUS


def compose(intent: Intent) -> Spec:
    """Lay an intent out as a Spec, deterministically.

    Continuous sources (typing, birds, a siren) start together and run under
    everything else; impulsive ones (a knock, a gunshot) queue one after the
    other with a natural gap. A counted run spaces its hits by `SPACING`. A
    later step ("... then ...") starts when the step before has had its say.
    """
    start0 = 0.5 if intent.texture != NO_BED else 0.1
    placed: list[tuple[str, float, float, float, float]] = []  # role, t, dur, gain, pan
    cursor = start0
    for step in intent.steps:
        conts = [it for it in step if _cont(it.role)]
        hits = [it for it in step if not _cont(it.role)]
        step_end = cursor
        for j, it in enumerate(conts):
            dur = _dur(it)
            t = cursor + 0.3 * j
            for k in range(it.count if it.role in COUNTABLE else 1):
                pan = _pan(it, k, it.count if it.role in COUNTABLE else 1)
                placed.append((it.role, t, dur, _gain(it, k), pan))
            step_end = max(step_end, t + min(dur, 2.5 if hits else dur))
        hit_t = cursor + (0.5 if conts else 0.0)
        for it in hits:
            dur = _dur(it)
            n = it.count if it.role in COUNTABLE else 1
            for k in range(n):
                placed.append((it.role, hit_t, dur, _gain(it, k), _pan(it, k, n)))
                if k < n - 1:
                    hit_t += SPACING.get(it.role, 1.0) * (0.7 if it.length == "quick" else 1.0)
            hold = HOLD.get(it.role, min(dur, 0.6))
            hit_t += dur if hold is None else min(hold, dur)
            hit_t += 0.25
            step_end = max(step_end, hit_t)
        cursor = step_end + 0.2
    natural_end = max([t + d for _, t, d, _, _ in placed], default=start0 + 2.0)
    if intent.duration is not None:
        duration = float(intent.duration)
    else:
        duration = min(30.0, max(2.0, np.ceil((natural_end + (0.6 if intent.texture != NO_BED else 0.3)) * 2) / 2))
    scale = 1.0
    if intent.duration is not None and natural_end > duration - 0.2:
        scale = max(0.05, (duration - 0.2) / max(natural_end, 1e-6))
    events: list[SpecEvent] = []
    for role, t, dur, gain, pan in placed[:MAX_EVENTS]:
        t = quantize_time(min(t * scale, duration - 0.1))
        d = dur * scale if scale < 1.0 else dur
        if intent.duration is not None and _cont(role) and scale == 1.0:
            d = max(d, duration - t - 0.5)  # a stated length: continuous sources fill it
        d = quantize_time(float(np.clip(d, 0.05, min(9.99, duration - t))))
        if d < 0.05:
            continue
        events.append(SpecEvent(role, t, d, quantize_gain(gain), pan, 0))
    events.sort(key=lambda e: (e.t, e.role))
    texture = intent.texture
    if texture == NO_BED and not events:
        texture = "room_cold"
    return Spec(float(duration), texture, intent.mood, events)


def _dur(it: Item) -> float:
    d = EVENT_STYLE[it.role][3]
    if it.length:
        d *= LENGTH_MODS[it.length][1]
    return d


def _gain(it: Item, k: int) -> float:
    g = EVENT_STYLE[it.role][2] + (MODS[it.level][1] if it.level else 0.0)
    if it.role == "bird_chirp" and k % 2 == 1:
        g -= 4.0
    return float(np.clip(g, -30.0, 0.0))


def _pan(it: Item, k: int, n: int) -> float:
    if it.side:
        return pan_value(1 if it.side == "left" else 7)
    if it.role == "bird_chirp" and n > 1:
        return pan_value(2 if k % 2 == 0 else 6)
    if n > 1 and it.role in ("sword_clash", "punch", "wave_crash", "frog", "owl"):
        return pan_value(3 if k % 2 == 0 else 5)
    return 0.0


# -- sampling an intent and saying it ---------------------------------------------

_ROLES = list(ROLE_PHRASES)
# Sounds that go together, so multi-sound prompts read like real requests.
_AFFINITY = {
    "ocean": ["wave_crash", "bird_chirp", "wind_gust", "splash", "footsteps"],
    "forest": ["bird_chirp", "wind_gust", "wood_creak", "footsteps", "owl", "water_drop"],
    "night": ["crickets", "owl", "frog", "wood_creak", "footsteps"],
    "stream": ["splash", "bird_chirp", "frog", "water_drop", "footsteps"],
    "traffic": ["car_horn", "siren", "engine", "footsteps", "church_bell", "door_slam"],
    "city_hum": ["footsteps", "church_bell", "car_horn", "siren", "applause", "coin_drop", "camera_shutter", "fireworks"],
    "room_cold": ["door_slam", "knock", "footsteps", "glass_break", "phone_ring", "keyboard_typing", "mouse_click",
                  "clock_tick", "paper_rustle", "pouring_water", "can_opening", "cork_pop", "metal_clang",
                  "clock_alarm", "heartbeat", "applause", "coin_drop", "camera_shutter", "punch"],
    "rain": ["thunder", "footsteps", "door_slam", "car_horn", "water_drop", "wind_gust"],
    "wind": ["wind_gust", "wood_creak", "door_slam", "footsteps", "sword_clash", "metal_clang"],
    "fire": ["wood_creak", "footsteps", "owl", "crickets", "explosion", "sword_clash"],
    "drone": ["heartbeat", "wood_creak", "door_slam", "water_drop", "footsteps", "riser", "sub_drop", "impact",
              "explosion", "gunshot", "clock_tick", "glitch"],
    "sci_fi": ["glitch", "whoosh", "riser", "sub_drop", "impact", "noise_sweep", "click", "bell", "explosion"],
    "vinyl": ["clock_tick", "paper_rustle", "pouring_water", "footsteps", "bell", "knock"],
    NO_BED: ["whoosh", "impact", "ui_pop", "riser", "swell", "click", "bell", "sub_drop", "glitch", "noise_sweep",
             "camera_shutter", "mouse_click", "coin_drop", "cork_pop", "punch", "gunshot", "glass_break"],
}
TEXTURES_ALL = list(TEXTURE_PLACES)


def sample_intent(rng: np.random.Generator) -> Intent:
    r = rng.random()
    if r < 0.12:  # a whole scene, named
        name = str(rng.choice(list(SCENE_PHRASES)))
        phrases, tex, mood, parts = SCENE_PHRASES[name]
        steps = [[Item(role, c, level=lv if lv in MODS else None, length=lv if lv in LENGTH_MODS else None)
                  for role, c, lv in parts]]
        intent = Intent(steps, tex, mood)
        intent.words["scene"] = str(rng.choice(phrases))
        if rng.random() < 0.35:
            m = str(rng.choice(list(MOOD_WORDS)))
            intent.mood = m
            intent.words["mood"] = m
        _maybe_duration(rng, intent)
        return intent
    # Pick a place first (or none), then sounds that fit it, with some strays.
    place_mode = rng.random()
    if place_mode < 0.18:
        tex = NO_BED
    else:
        tex = str(rng.choice(TEXTURES_ALL))
    pool = _AFFINITY[tex]
    n_items = int(rng.choice([1, 1, 1, 2, 2, 2, 3, 3, 4]))
    if r < 0.17:
        n_items = 0  # just a bed: "rain on a window", "forest ambience"
        if tex == NO_BED:
            tex = str(rng.choice(TEXTURES_ALL))
    roles: list[str] = []
    for _ in range(n_items):
        src = pool if rng.random() < 0.8 else _ROLES
        cand = [x for x in src if x not in roles]
        if cand:
            roles.append(str(rng.choice(cand)))
    items = []
    for role in roles:
        it = Item(role)
        if role in COUNTABLE and rng.random() < 0.28:
            it.count = int(rng.choice([2, 2, 2, 3, 3, 4, 5]))
        if role == "bird_chirp" and rng.random() < 0.5:
            it.count = 2  # "birds" read as two singers, L and R
        if rng.random() < 0.22:
            it.level = str(rng.choice(list(MODS)))
        if rng.random() < 0.15:
            it.length = str(rng.choice(list(LENGTH_MODS)))
        if rng.random() < 0.05:
            it.side = str(rng.choice(list(PAN_MODS)))
        items.append(it)
    # Keep within MAX_EVENTS.
    while sum(it.count for it in items) > MAX_EVENTS:
        big = max(items, key=lambda x: x.count)
        big.count -= 1
    # Steps: all at once, all in sequence, or a mix.
    steps: list[list[Item]] = []
    seq = rng.random()
    if len(items) <= 1 or seq < 0.45:
        steps = [items] if items else []
    elif seq < 0.8:
        steps = [[it] for it in items]
    else:
        cut = int(rng.integers(1, len(items)))
        steps = [items[:cut], items[cut:]]
    texture_said = tex != NO_BED and rng.random() < 0.75
    if not texture_said:
        tex = _implied_texture(items)
    if not items and tex == NO_BED:
        tex = "room_cold"
    mood_said = rng.random() < 0.4
    mood = str(rng.choice(list(MOOD_WORDS))) if mood_said else _implied_mood(items, tex)
    if not texture_said and mood_said and tex == "room_cold" and mood in ("tense", "dark", "eerie", "intense"):
        tex = "drone"
    intent = Intent(steps, tex, mood)
    intent.words["texture_said"] = texture_said
    intent.words["mood_said"] = mood_said
    _maybe_duration(rng, intent)
    return intent


def _maybe_duration(rng, intent: Intent) -> None:
    if rng.random() < 0.22:
        intent.duration = float(rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 2.5, 7, 9]))


def _implied_texture(items: list[Item]) -> str:
    for it in items:
        if it.role in HOME:
            return HOME[it.role]
    if items and all(it.role in DRY for it in items):
        return NO_BED
    return "room_cold"


def _implied_mood(items: list[Item], tex: str) -> str:
    for it in items:
        if it.role in ROLE_MOOD:
            return ROLE_MOOD[it.role]
    return TEXTURE_MOOD.get(tex, "neutral")


def _phrase(rng, it: Item) -> str:
    raw = str(rng.choice(ROLE_PHRASES[it.role]))
    sing, _, plural = raw.partition("|")
    count = it.count if it.role in COUNTABLE or it.role == "bird_chirp" else 1
    if it.role == "bird_chirp":
        # "birds" already reads as the pair; a single bird is said as one.
        sing = sing if count > 1 or "a " in sing or sing in ("chirps",) else str(rng.choice(["a bird chirping", "a little bird", "a single bird", "one bird singing"]))
        count = 1
    text = sing
    if count > 1:
        if plural and rng.random() < 0.7:
            text = f"{rng.choice(NUMBER_WORDS[count])} {plural}"
        else:
            text = f"{sing} {rng.choice(TIMES_WORDS[count])}"
    elif rng.random() < 0.35 and not text.startswith(("a ", "an ", "the ", "someone", "some ")):
        text = ("an " if text[0] in "aeiou" else "a ") + text
    if it.level:
        w = str(rng.choice(MODS[it.level][0]))
        text = _modify(rng, text, w)
    if it.length:
        w = str(rng.choice(LENGTH_MODS[it.length][0]))
        text = _modify(rng, text, w)
    if it.side:
        text = f"{text} {rng.choice(PAN_MODS[it.side][0])}"
    return text


def _modify(rng, text: str, w: str) -> str:
    if w.startswith(("in the", "far", "up ", "right ", "on the")) or " " in w and rng.random() < 0.5:
        return f"{text} {w}"
    for art in ("a ", "an "):
        if text.startswith(art):
            rest = text[len(art):]
            return ("an " if w[0] in "aeiou" else "a ") + f"{w} {rest}"
    if rng.random() < 0.25:
        return f"{text}, {w}"
    return f"{w} {text}"


def say(rng: np.random.Generator, intent: Intent) -> str:
    """One free-text way to ask for `intent`."""
    if "scene" in intent.words:
        body = intent.words["scene"]
        if intent.words.get("mood"):
            body = _with_mood(rng, body, intent.mood)
    else:
        steps = [[_phrase(rng, it) for it in step] for step in intent.steps]
        groups = []
        for parts in steps:
            if len(parts) == 1:
                groups.append(parts[0])
            elif parts:
                j = str(rng.choice(PAR_JOIN))
                groups.append(j.join(parts[:-1]) + (" and " if rng.random() < 0.6 else j) + parts[-1])
        sounds = ""
        for i, g in enumerate(groups):
            sounds += g if i == 0 else str(rng.choice(SEQ_JOIN)) + g
        place = ""
        if intent.words.get("texture_said"):
            if sounds and rng.random() < 0.8:
                place = str(rng.choice(TEXTURE_PLACES[intent.texture]))
            else:
                place = str(rng.choice(TEXTURE_BARE[intent.texture] + TEXTURE_PLACES[intent.texture]))
        if not sounds:
            body = place or str(rng.choice(TEXTURE_BARE[intent.texture]))
        elif not place:
            body = sounds
        else:
            o = rng.random()
            if o < 0.55:
                body = f"{sounds} {place}"
            elif o < 0.75:
                body = f"{place}, {sounds}" if place.startswith(("in ", "on ", "at ", "by ", "near ", "next ", "under ", "around ", "aboard ", "during ", "with ", "late ", "outside ", "inside", "indoors")) else f"{place} with {sounds}"
            elif o < 0.9:
                bare = str(rng.choice(TEXTURE_BARE[intent.texture]))
                body = f"{sounds} {rng.choice(['over', 'with', 'and', 'in', 'under'])} {bare}"
            else:
                body = f"{sounds}. {place}"
        if intent.words.get("mood_said"):
            body = _with_mood(rng, body, intent.mood)
    if intent.duration is not None:
        body = _with_duration(rng, body, intent.duration)
    if rng.random() < 0.15:
        x = str(rng.choice(UNSUPPORTED))
        frame = str(rng.choice(UNSUPPORTED_FRAMES))
        if frame.startswith(("no ", "without ", "(no ", "but no ", "minus ")):
            x = x.split(" ", 1)[1] if x.startswith(("a ", "an ")) else x
        frame = frame.format(x=x)
        body = f"{body}, {frame}" if rng.random() < 0.7 else f"{frame}, {body}"
    text = str(rng.choice(FILLERS)) + body + str(rng.choice(SUFFIXES))
    text = _typos(rng, text)
    c = rng.random()
    if c < 0.08:
        text = text.upper()
    elif c < 0.3:
        text = text[:1].upper() + text[1:]
    return " ".join(text.split())


def _with_mood(rng, body: str, mood: str) -> str:
    w = str(rng.choice(MOOD_WORDS[mood]))
    o = rng.random()
    if o < 0.4:
        return f"{body}, {w}"
    if o < 0.6:
        return f"{w} {body}"
    if o < 0.8:
        return f"{body}, {w} mood"
    return f"{body} ({w})"


def _with_duration(rng, body: str, dur: float) -> str:
    n = f"{dur:g}"
    form = str(rng.choice([f"{n} second", f"{n}-second", f"{n}s", f"{n} sec", f"{n} seconds of"]))
    if form.endswith("of"):
        return f"{form} {body}"
    if rng.random() < 0.6:
        return f"{form} {body}" if not form.endswith("s") or form.endswith("ss") else f"{body}, {form}"
    return f"{body}, {n} seconds" if rng.random() < 0.5 else f"{body} ({n}s)"


_KEYS = "qwertyuiop asdfghjkl zxcvbnm"


def _neighbor(c: str, rng) -> str:
    rows = _KEYS.split()
    for row in rows:
        i = row.find(c)
        if i >= 0:
            j = min(len(row) - 1, max(0, i + int(rng.choice([-1, 1]))))
            return row[j]
    return c


def _typos(rng, text: str) -> str:
    rate = float(rng.choice([0.0, 0.0, 0.0, 0.03, 0.06, 0.1]))
    if rate == 0.0:
        return text
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) < 4 or not w.isalpha() or rng.random() > rate * 3:
            continue
        k = int(rng.integers(1, len(w) - 1))
        op = rng.random()
        if op < 0.3:
            w = w[:k] + w[k + 1:]  # drop
        elif op < 0.55:
            w = w[:k] + w[k + 1] + w[k] + w[k + 2:]  # swap
        elif op < 0.75:
            w = w[:k] + w[k] + w[k:]  # double
        else:
            w = w[:k] + _neighbor(w[k], rng) + w[k + 1:]  # fat finger
        words[i] = w
    return " ".join(words)


def sample_pair(rng: np.random.Generator) -> tuple[str, Spec]:
    intent = sample_intent(rng)
    return say(rng, intent), compose(intent)
