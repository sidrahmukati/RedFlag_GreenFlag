import re
from typing import Dict, List
from schema import BehavioralRubric, PILLAR_COLUMNS
from transformers import pipeline

# Lightweight, fast local NLI model (~280 MB, downloads once and caches locally)
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-small-mnli-fever-docnli-ling-2c"

# Positive and Negative behavioral descriptions for each of the 9 pillars
PILLAR_HYPOTHESES: Dict[str, Dict[str, str]] = {
    "public_respect": {
        "pos": (
            "The partner treats them with dignity, warmth, and respect in front"
            " of others."
        ),
        "neg": (
            "The partner humiliates, mocks, or disrespects them in public."
        ),
    },
    "public_private_consistency": {
        "pos": (
            "The partner behaves consistently and stays in regular contact both"
            " in person and over text."
        ),
        "neg": (
            "The partner acts caring in person, but disappears, ghosts, or"
            " stops texting for days without explanation."
        ),
    },
    "social_integrity": {
        "pos": (
            "The partner is honest, loyal, and does not gossip or put others"
            " down."
        ),
        "neg": (
            "The partner talks badly behind people's backs or acts two-faced."
        ),
    },
    "honesty_and_transparency": {
        "pos": (
            "The partner communicates openly, truthfully, and transparently"
            " without hiding things."
        ),
        "neg": (
            "The partner lies, makes excuses, contradicts their actions, or"
            " hides the truth."
        ),
    },
    "emotional_attentiveness": {
        "pos": (
            "The partner is attentive, caring, and actively checks in on their"
            " feelings."
        ),
        "neg": (
            "The partner ignores their messages, neglects their needs, or acts"
            " emotionally unavailable."
        ),
    },
    "empathy_and_validation": {
        "pos": (
            "The partner listens with empathy and validates their feelings."
        ),
        "neg": "The partner dismisses, invalidates, or mocks their feelings.",
    },
    "accountability": {
        "pos": (
            "The partner admits mistakes sincerely and takes responsibility for"
            " their actions."
        ),
        "neg": (
            "The partner deflects blame, calls the other person too sensitive,"
            " or refuses to apologize."
        ),
    },
    "boundary_respect": {
        "pos": "The partner respects personal boundaries, space, and consent.",
        "neg": (
            "The partner pressures them, crosses clear boundaries, or ignores"
            " consent."
        ),
    },
    "safety_catchall": {
        "pos": (
            "The partner creates a safe, non-controlling, and trustworthy"
            " environment."
        ),
        "neg": (
            "The partner is controlling, possessive, manipulative, or threatens"
            " safety."
        ),
    },
}


class BehaviorExtractor:

    def __init__(self, relevance_threshold: float = 0.60):
        self.relevance_threshold = relevance_threshold
        self.classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=-1,  # Runs cleanly on CPU; change to 0 if using CUDA GPU
        )

    def _split_sentences(self, text: str) -> List[str]:
        # Split on sentence endings AND contrast conjunctions (but, however, yet, although)
        raw_parts = re.split(
            r"[.!?\n]+|\b(?:however|but|yet|although)\b,?",
            text,
            flags=re.IGNORECASE,
        )
        sentences = [
            s.strip(" ,;:") for s in raw_parts if len(s.strip(" ,;:")) > 4
        ]
        return sentences if sentences else [text.strip()]

    def extract(self, story: str) -> BehavioralRubric:
        sentences = self._split_sentences(story)
        pillar_hits: Dict[str, List[float]] = {
            col: [] for col in PILLAR_COLUMNS
        }

        candidate_labels = []
        label_map = {}
        for col, pair in PILLAR_HYPOTHESES.items():
            candidate_labels.append(pair["pos"])
            label_map[pair["pos"]] = (col, "pos")
            candidate_labels.append(pair["neg"])
            label_map[pair["neg"]] = (col, "neg")

        for sentence in sentences:
            result = self.classifier(
                sentence,
                candidate_labels=candidate_labels,
                hypothesis_template="In this relationship situation, {}",
                multi_label=True,
            )
            clause_scores: Dict[str, Dict[str, float]] = {
                col: {"pos": 0.0, "neg": 0.0} for col in PILLAR_COLUMNS
            }
            for label, score in zip(result["labels"], result["scores"]):
                col, polarity = label_map[label]
                clause_scores[col][polarity] = score

            # Rank pillars by how strongly they fired on this specific clause
            clause_winners = []
            for col, pair_scores in clause_scores.items():
                pos_s, neg_s = pair_scores["pos"], pair_scores["neg"]
                top_score = max(pos_s, neg_s)
                if (
                    top_score >= self.relevance_threshold
                    and abs(pos_s - neg_s) >= 0.15
                ):
                    sign = 1.0 if pos_s > neg_s else -1.0
                    intensity = 0.35 + 0.65 * (
                        (top_score - self.relevance_threshold)
                        / (1.0 - self.relevance_threshold)
                    )
                    clause_winners.append(
                        (top_score, col, round(sign * intensity, 2))
                    )

            # Keep only the top 2 strongest pillars per clause to prevent halo-effect inflation
            clause_winners.sort(reverse=True)
            for _, col, signed_val in clause_winners[:2]:
                pillar_hits[col].append(signed_val)

        # Aggregate across clauses: if any clear negative flag occurred on a pillar, preserve it
        final_scores: Dict[str, float] = {}
        for col in PILLAR_COLUMNS:
            hits = pillar_hits[col]
            if not hits:
                final_scores[col] = 0.0
            else:
                neg_hits = [h for h in hits if h < 0]
                final_scores[col] = min(neg_hits) if neg_hits else max(hits)

        return BehavioralRubric(**final_scores)