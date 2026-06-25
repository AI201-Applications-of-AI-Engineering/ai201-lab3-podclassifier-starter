import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    Structure (per specs/classifier-spec.md):
      1. Task instruction + the four valid label definitions
      2. The labeled examples, each as a "Title / Description / Label" block
         (zero-shot: skipped entirely when no examples are provided)
      3. The new episode in the same block shape, then an instruction to classify
         and the exact output format to return ("Label: X / Reasoning: Y")
    """
    # --- 1. Task instruction --------------------------------------------------
    # Tells the LLM what it's doing and defines each label so it can classify
    # even with zero examples. Mirrors the taxonomy but kept concise.
    task_instruction = (
        "You are classifying podcast episodes by their format. Classify the "
        "episode into exactly one of these four labels:\n\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion — "
        "no guests, no assembled external sources\n"
        "- panel: multiple guests with roughly equal speaking time, often "
        "debating or discussing a topic together\n"
        "- narrative: a story assembled from external sources — interviews, "
        "archival audio, reporting — with a clear narrative arc\n"
    )

    # --- 2. Example format ----------------------------------------------------
    # Each labeled example is rendered as a Title / Description / Label block,
    # blocks separated by a "---" delimiter (spec: "example format").
    # When labeled_examples is empty we skip this whole section so the prompt
    # degrades cleanly to a zero-shot prompt — no dangling "Examples:" header.
    example_blocks = []
    for ex in labeled_examples:
        example_blocks.append(
            f"Title: {ex['title']}\n"
            f"Description: {ex['description']}\n"
            f"Label: {ex['label']}"
        )

    examples_section = ""
    if example_blocks:
        examples_section = (
            "Here are labeled examples:\n\n"
            + "\n\n---\n\n".join(example_blocks)
            + "\n\n---\n\n"
        )

    # --- 3. New episode presentation + output format request ------------------
    # Same block shape as the examples, but the Label line is replaced with "?"
    # so the model fills it in (spec: "how the new episode is presented").
    # Then we request the exact "Label: X / Reasoning: Y" format that
    # classify_episode() parses (spec: "output format request").
    new_episode = (
        "Now classify this episode:\n\n"
        f"Description: {description}\n"
        "Label: ?\n\n"
        "Classify the episode above. If detail is thin, still pick your best "
        "guess from the four labels — do not refuse. Return your answer in "
        "exactly this format:\n\n"
        "Label: <one of: interview, solo, panel, narrative>\n"
        "Reasoning: <one or two sentences explaining your choice>"
    )

    return f"{task_instruction}\n{examples_section}{new_episode}"


def _parse_response(text: str) -> dict:
    """
    Pull a label and reasoning out of the LLM's "Label: X / Reasoning: Y" text.

    Per the spec, we don't assume fixed line numbers (a zero-shot prompt can
    drift) — we scan for the "Label:" and "Reasoning:" tokens instead. The
    label is then normalized (strip whitespace + markdown/punctuation, lower)
    so it can be compared against VALID_LABELS reliably.
    """
    label = ""
    reasoning = ""

    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("label:") and not label:
            # Take everything after the first colon as the raw label.
            raw = stripped.split(":", 1)[1]
            # Normalize: drop markdown/punctuation noise, collapse to lowercase.
            label = raw.strip().strip("*`_.\"' ").lower()
        elif lowered.startswith("reasoning:") and not reasoning:
            reasoning = stripped.split(":", 1)[1].strip()

    return {"label": label, "reasoning": reasoning}


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    Returns a dict with:
      - "label"     : one of VALID_LABELS, or "unknown" if invalid/unparseable/error
      - "reasoning" : the model's brief explanation (or an error message on failure)
    """
    # Step 1 — Build the few-shot prompt from the labeled examples + this episode.
    prompt = build_few_shot_prompt(labeled_examples, description)

    try:
        # Step 2 — Send the prompt to the LLM. One user message; cap the tokens
        # since we only need a label line plus a sentence or two of reasoning.
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
        )
        text = response.choices[0].message.content or ""

        # Step 3 — Parse the "Label: X / Reasoning: Y" response.
        parsed = _parse_response(text)
        label = parsed["label"]
        reasoning = parsed["reasoning"]

        # Step 4 — Validate. Anything not in the taxonomy collapses to "unknown"
        # (counts as incorrect in evaluation, but is still returned, not raised).
        if label not in VALID_LABELS:
            label = "unknown"

        return {"label": label, "reasoning": reasoning}

    except Exception as e:
        # Step 5 — Never let one bad call crash the 20-episode evaluation loop.
        # Mark the result and carry the error text forward for later review.
        return {"label": "unknown", "reasoning": f"error: {e}"}
