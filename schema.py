from pydantic import BaseModel, Field


class BehavioralRubric(BaseModel):
    public_respect: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "How the partner treats and speaks about the user around friends,"
            " family, or strangers. -1.0: Mocks, embarrasses, belittles, or"
            " makes jokes at the partner's expense in front of others. 0.0: Not"
            " mentioned or neutral public interaction. +1.0: Protects partner's"
            " dignity, speaks proudly of them, and shows clear respect"
            " publicly."
        ),
    )

    public_private_consistency: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Alignment between private words/affection and public or"
            " real-world actions. -1.0: Double standard—affectionate or"
            " attentive in private texts, but cold, distant, or hides the"
            " relationship in person/public. 0.0: Not mentioned. +1.0: Warmth,"
            " effort, and treatment remain identical in private, in person, and"
            " around others."
        ),
    )

    social_integrity: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Standards and backbone when peers make demeaning, sexist, or"
            " disrespectful remarks. -1.0: Actively joins in derogatory jokes"
            " or enables toxic peer behavior (use -0.4 to -0.6 for passive"
            " complicit silence). 0.0: Not mentioned. +1.0: Actively calls out"
            " disrespect, sets boundaries with friends, and upholds integrity."
        ),
    )

    honesty_and_transparency: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Truthfulness and congruence between excuses given and actual"
            " behavior. -1.0: Lies, hides things, or uses convenient excuses"
            " (e.g., claiming 'shyness' despite acting outgoing with everyone"
            " else). 0.0: Not mentioned. +1.0: Open, straightforward, and"
            " honest even when uncomfortable."
        ),
    )

    emotional_attentiveness: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Noticing, remembering, and caring about the partner's daily mood"
            " and small details. -1.0: Ignores emotional cues, forgets"
            " important details, or shows apathy. 0.0: Not mentioned. +1.0:"
            " Notices subtle mood shifts, remembers small preferences, and"
            " checks in thoughtfully."
        ),
    )

    empathy_and_validation: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Response when the partner expresses hurt, discomfort, or"
            " vulnerability. -1.0: Laughs off feelings, calls partner 'too"
            " sensitive,' or dismisses pain. 0.0: Not mentioned. +1.0:"
            " Listens without judgment, validates feelings, and offers genuine"
            " comfort."
        ),
    )

    accountability: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "How the partner handles mistakes, conflict, and feedback. -1.0:"
            " Gets defensive, shifts blame onto the partner, or refuses to"
            " apologize. 0.0: Not mentioned. +1.0: Owns mistakes without"
            " excuses, apologizes sincerely, and changes behavior."
        ),
    )

    boundary_respect: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Respect for personal space, consent, privacy, and stated limits."
            " -1.0: Violates privacy, pressures partner after a 'no,' or"
            " ignores stated boundaries. 0.0: Not mentioned. +1.0: Honors"
            " limits immediately without guilt-tripping or pushing."
        ),
    )

    safety_catchall: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Catch-all for any severe controlling, manipulative, abusive, or"
            " extraordinarily supportive behavior not captured above. -1.0:"
            " Coercion, threats, surveillance, isolation, or aggressive"
            " outbursts. 0.0: No additional unlisted behavior. +1.0:"
            " Exceptional protective care, maturity, or selfless green-flag"
            " support."
        ),
    )


# Helper list of the 9 numeric feature names for Pandas, XGBoost, and SHAP
PILLAR_COLUMNS = list(BehavioralRubric.model_fields.keys())