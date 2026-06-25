"""
All LLM prompts live here. No prompt text lives in logic code.
Each function returns (system_message, user_message) tuple.
"""

import json
from config.axes import AXES


# ─────────────────────────────────────────────────────────────────
# ANNOTATION PROMPT (offline, one call per post)
# ─────────────────────────────────────────────────────────────────

_ANNOTATION_SYSTEM = """You are a precise annotation system. Your job is to label a short social-media post along delivery/style axes — measuring HOW something is said, not WHAT is said. Judge the delivery, not the topic and not whether you agree.

## AXES

### 1. reading_level (scalar 0.0–1.0)  [textual]
How advanced the vocabulary and sentence structure is. Folds in how much context the author assumes.
- 0.2 = simple everyday language anyone can follow ("My cat did the funniest thing today")
- 0.5 = average social media post, some topic-specific terms but accessible
- 0.8 = specialist vocabulary, complex syntax, assumes familiarity ("The tarsi of these arboreal predators rotate ~180°")
→ Return: {"score": <float>}

### 2. concreteness (scalar 0.0–1.0)  [textual]
Abstract/theoretical claims vs grounded in specifics — examples, numbers, named cases, analogies.
- 0.2 = pure abstract claim, no examples ("Capitalism incentivizes extraction over sustainability")
- 0.5 = mix of general points and some specifics
- 0.8 = rich in concrete details, specific named cases ("In July 1919 tabby kitten Wopsie became the first cat to fly across the Atlantic on airship R34")
→ Return: {"score": <float>}

### 3. narrativity (scalar 0.0–1.0 + subtype)  [textual]
How story-like the delivery is — characters, sequence, scene, or arc.
- 0.2 = pure assertion or argument, no narrative framing ("Dogs are not obligate carnivores")
- 0.5 = some sequential or situational framing ("During summer an epidemic garnered attention... now the culprit is identified")
- 0.8 = clear story with character/experience/arc ("My mom's beagle licks faces... anyway he now follows me everywhere")
→ Return: {"score": <float>, "subtype": "<personal anecdote|news narrative|historical|observational scene|null>"}

### 4. hedging (scalar 0.0–1.0)  [interpersonal]
How certain the delivery sounds. About epistemic stance, NOT about whether the claim is true.
- 0.2 = states things flatly as fact ("Declawing does not reduce euthanasia")
- 0.5 = some qualification ("declawing probably doesn't help much")
- 0.8 = heavily hedged, tentative ("it might be that declawing doesn't really help, though I could be wrong and the evidence seems mixed")
→ Return: {"score": <float>}

### 5. tone (scalar 0.0–1.0)  [interpersonal]
INTENSITY of feeling — how heated vs measured. NOT the opinion/stance itself (a calm post can hold a strong opinion).
- 0.2 = calm, measured, detached, clinical ("A study showed no marked increase in surrenders after declaw bans")
- 0.5 = moderately engaged, has feeling but controlled
- 0.8 = highly charged, urgent, outraged, intense ("this is the most batshit false equivalence I've ever seen")
→ Return: {"score": <float>}

### 6. warmth (scalar 0.0–1.0)  [interpersonal]
DIRECTION of affect toward the reader/subject — distinct from tone (intensity).
- 0.2 = cold, harsh, hostile, contemptuous ("anyone who believes this is an idiot")
- 0.5 = neutral, matter-of-fact
- 0.8 = warm, kind, affirming, generous ("I totally get why you'd feel that way, and honestly it's lovely")
→ Return: {"score": <float>}

### 7. self_disclosure (scalar 0.0–1.0)  [interpersonal]
How much the author opens up about their OWN self/experience/feelings.
- 0.2 = impersonal, detached, no author presence ("Prairie dogs are a keystone species")
- 0.5 = some personal framing ("I find prairie dogs fascinating")
- 0.8 = openly personal/confessional ("I burned out in year 3 and hid it from everyone; here's what I learned")
→ Return: {"score": <float>}

### 8. casualness (scalar 0.0–1.0)  [interpersonal]
Polished formal register vs casual internet register (slang, lowercase, fragments, emoji).
- 0.2 = polished, formal, fully punctuated written prose
- 0.5 = conversational but clean
- 0.8 = texty/slangy, lowercase, fragments, emoji ("ok this is actually insane 🔥 no notes")
→ Return: {"score": <float>}

### 9. humor (scalar 0.0–1.0 + subtype)  [poetic]
Presence and intensity of comedy, wit, or playful tone.
- 0.0 = entirely serious, no comedic element
- 0.3 = lightly playful, a wry aside ("typical star behavior")
- 0.7 = substantially humorous, comedy is a primary feature
→ Return: {"score": <float>, "subtype": "<dry|satirical|self-deprecating|absurdist|deadpan|witty|null>"}

## RULES
1. Judge ONLY from the text. Do not infer beyond what is written.
2. Every axis gets a score. No axis should be left out.
3. Do not conflate axes: reading_level is about WORDS, concreteness is about SPECIFICITY, tone is INTENSITY of feeling, warmth is DIRECTION of feeling, hedging is CERTAINTY, self_disclosure is AUTHOR PRESENCE.
4. Return ONLY a JSON object keyed by axis name. No prose, no markdown, no code fences."""


_ANNOTATION_FEW_SHOT = [
    {
        "post": "the undisputed best thing in the world is when dogs are dreaming, and you can tell because they're running, waggling their tail, eating or borking a little bit in their sleep",
        "annotation": {
            "reading_level": {"score": 0.2},
            "concreteness": {"score": 0.7},
            "narrativity": {"score": 0.4, "subtype": "observational scene"},
            "hedging": {"score": 0.1},
            "tone": {"score": 0.45},
            "warmth": {"score": 0.85},
            "self_disclosure": {"score": 0.3},
            "casualness": {"score": 0.6},
            "humor": {"score": 0.4, "subtype": "witty"},
        }
    },
    {
        "post": "one of the dominant ideologies among this country's elite is College Admissions Brain where you reorient your entire worldview around valorizing the effective strategies for getting into good colleges",
        "annotation": {
            "reading_level": {"score": 0.7},
            "concreteness": {"score": 0.25},
            "narrativity": {"score": 0.15, "subtype": None},
            "hedging": {"score": 0.25},
            "tone": {"score": 0.5},
            "warmth": {"score": 0.3},
            "self_disclosure": {"score": 0.1},
            "casualness": {"score": 0.35},
            "humor": {"score": 0.35, "subtype": "satirical"},
        }
    },
    {
        "post": "So the myth that declawing \"saves\" cats from euthanasia doesn't hold. U of Florida's shelter med program did a study showing no marked increase in owner surrenders/euth after declaw bans were introduced.",
        "annotation": {
            "reading_level": {"score": 0.55},
            "concreteness": {"score": 0.75},
            "narrativity": {"score": 0.3, "subtype": "news narrative"},
            "hedging": {"score": 0.15},
            "tone": {"score": 0.35},
            "warmth": {"score": 0.45},
            "self_disclosure": {"score": 0.05},
            "casualness": {"score": 0.35},
            "humor": {"score": 0.0, "subtype": None},
        }
    },
]


def annotation_system() -> str:
    """System prompt for annotating a post on all style axes (includes few-shot)."""
    examples = "\n\n## CALIBRATION EXAMPLES\n"
    for i, ex in enumerate(_ANNOTATION_FEW_SHOT, 1):
        examples += f'\nExample {i}:\nPOST: "{ex["post"]}"\nANNOTATION:\n{json.dumps(ex["annotation"], indent=2)}\n'
    return _ANNOTATION_SYSTEM + examples


def annotation_user(post_text: str) -> str:
    """User message for annotation."""
    return f'POST:\n"""\n{post_text}\n"""'


# ─────────────────────────────────────────────────────────────────
# PROFILE UPDATE PROMPT (online, per interaction)
# ─────────────────────────────────────────────────────────────────

def profile_update_system() -> str:
    """System prompt for updating the user profile."""
    axes_list = "\n".join(f'  - {ax["name"]}: {ax["definition"]}' for ax in AXES)
    axis_names = ", ".join(ax["name"] for ax in AXES)

    return f"""You maintain a running profile of one user's content preferences based on their interactions.

The profile has two parts:
1. TOPICS — which topics they are interested in, and how much background they appear to have in each.
2. STYLE — their delivery preferences along fixed axes, stated in plain language. Emergent subtypes are allowed (e.g. "prefers dry, understated humor over slapstick").

STYLE AXES:
{axes_list}

RULES:
- Update BOTH topics and style from each interaction.
- Topic interest and style are learned TOGETHER from every interaction. A post choice reveals both.
- When two shown posts differ on multiple axes simultaneously, attribute cautiously — lower confidence on those axes rather than guessing which axis drove the choice.
- Do NOT attribute a topical preference to a style axis. If the difference between two posts is clearly topical (different subject matter), note that and do not update style.
- A style preference reaches high confidence (> 0.7) only after CONSISTENT evidence across 2+ interactions.
- Be conservative with early interactions. First interaction should yield low confidence everywhere.
- IMPORTANT: In the confidence object, use EXACTLY the topic names as they appear in the post data (e.g. "Books & Reading", not "books_reading"). Use EXACTLY these axis names: {axis_names}.

RETURN JSON ONLY:
{{
  "topics_prose": "<2-4 sentences: what topics interest this user, what they seem to know>",
  "style_prose": "<2-4 sentences: delivery preferences observed so far, with caveats for uncertainty>",
  "confidence": {{
    "topics": {{"<exact topic name from posts>": <0.0-1.0>, ...}},
    "axes": {{"<exact axis name>": <0.0-1.0>, ...}}
  }}
}}"""


def profile_update_user(current_profile: dict, shown_posts: list[dict],
                        user_choices: list[str]) -> str:
    """User message for profile update."""
    posts_desc = []
    for p in shown_posts:
        status = "✓ CHOSEN" if p["post_id"] in user_choices else "✗ SKIPPED"
        axes_str = json.dumps(p.get("axes", {}), indent=None)
        posts_desc.append(
            f'[{status}] id={p["post_id"]}, topic={p["topic_name"]}\n'
            f'  Axes: {axes_str}\n'
            f'  Text: "{p["text"][:350]}"'
        )

    return (
        f"CURRENT PROFILE:\n"
        f"Topics: {current_profile.get('topics_prose') or '(first interaction — no data yet)'}\n"
        f"Style: {current_profile.get('style_prose') or '(first interaction — no data yet)'}\n\n"
        f"THIS INTERACTION:\n" + "\n\n".join(posts_desc)
    )
