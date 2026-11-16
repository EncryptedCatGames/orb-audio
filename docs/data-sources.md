# Data sources

The detail model is only as good as its pairs. The rule is simple: **CC0 for anything that ships, licensed for anything that trains, provenance for everything.**

## Texture and ambience (abundant)

| source | license | notes |
|---|---|---|
| Freesound, CC0 filter | CC0 | the main source; 30k clips is realistic |
| Internet Archive | varies | check per item; many CC0 field recordings |
| Free Music Archive | CC0 subset | ambience and room tone, not songs |
| Pixabay audio | Pixabay license | check derivative-training terms |
| Zapsplat | Zapsplat license | free with attribution; check terms |

Good roles: room tone, wind, rain, fire, city hum, vinyl, drone, sci-fi hum.

## Designed SFX (scarce)

CC0 whooshes, impacts, risers, and UI sounds are rare. Options, in order:

1. **Teacher-generated** (Stable Audio Open or equivalent). Check the teacher's license before training on derivatives. Disclose it.
2. **Licensed library** (Artlist, Epidemic, Soundly). Read the training clause; most forbid it.
3. **CC0 only**. Expect a thin role distribution. If a role has fewer than 500 clean clips, it does not get a detail model.

## What to reject

- speech, music, or a strong tonal bed under the effect
- clips longer than 3 s for the detail pairs
- clipped, DC-offset, or heavily compressed clips
- anything with an unclear license

## Preparation

- 48 kHz mono WAV for the pair builder.
- Trim to 0.1–3 s around the onset.
- Normalize peak to 0.9 before the pair builder; it RMS-normalizes the pair.
- Keep the original file name and license in `data/provenance.jsonl`.

## Provenance record

```json
{"file": "data/cc0/whoosh_042.wav", "source": "freesound", "id": "123456", "license": "CC0", "url": "..."}
```

The license audit at ship time reads this file. If a clip has no provenance, it does not ship.
