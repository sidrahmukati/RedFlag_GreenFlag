import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from extractor import BehaviorExtractor
from schema import PILLAR_COLUMNS

# Fun, casual display names for the 9 behavioral pillars
PILLAR_LABELS = {
    "public_respect": "Public Flex vs. Public Diss",
    "public_private_consistency": "Texting vs. IRL Consistency",
    "social_integrity": "Gossip & Two-Faced Radar",
    "honesty_and_transparency": "No-Cap Honesty",
    "emotional_attentiveness": "Golden Retriever Energy",
    "empathy_and_validation": "Vibe & Feelings Check",
    "accountability": "Owning Their Mess",
    "boundary_respect": "Respecting Space",
    "safety_catchall": "Safe Space Energy",
}

st.set_page_config(
    page_title="Red Flag or Green Forest? 🚩🌲",
    page_icon="✨",
    layout="centered",
)

# Bright, playful Warm Cream + Coral/Mint/Violet CSS with explicit widget contrast overrides
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #fffdf9 0%, #f5f3ff 100%);
        color: #1e1b4b;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 750px;
    }
    /* Fix label visibility above text area */
    label, .stTextArea label p {
        color: #1e1b4b !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    /* Force text area to be crisp white with dark readable text */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #1e1b4b !important;
        caret-color: #8b5cf6 !important;
        border: 2px solid #ddd6fe !important;
        border-radius: 14px !important;
        padding: 0.9rem !important;
        font-size: 0.98rem !important;
    }
    .stTextArea textarea::placeholder {
        color: #9ca3af !important;
    }
    .stTextArea textarea:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
    }
    /* Force secondary Reset button to be crisp white with dark text */
    div.stButton > button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #1e1b4b !important;
        border: 2px solid #ddd6fe !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #f5f3ff !important;
        border-color: #8b5cf6 !important;
    }
    .hero-box {
        background: linear-gradient(135deg, #ffedd5 0%, #fae8ff 50%, #dcfce7 100%);
        padding: 1.8rem 2rem;
        border-radius: 22px;
        border: 2px solid #ffffff;
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.12);
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #1e1b4b;
        margin: 0 0 0.4rem 0;
    }
    .hero-sub {
        font-size: 1rem;
        color: #4c1d95;
        font-weight: 500;
        margin: 0;
    }
    /* Bouncy Violet-Pink Primary Button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        padding: 0.6rem 1.3rem !important;
        box-shadow: 0 4px 14px rgba(236, 72, 153, 0.28) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.38) !important;
    }
    .verdict-card {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        margin: 0.9rem 0 1.3rem 0;
        background-color: #ffffff;
        border: 2px solid;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    }
    .verdict-red {
        border-color: #fb7185;
        background: linear-gradient(135deg, #fff1f2 0%, #ffffff 70%);
    }
    .verdict-green {
        border-color: #34d399;
        background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 70%);
    }
    .verdict-mixed {
        border-color: #fbbf24;
        background: linear-gradient(135deg, #fffbeb 0%, #ffffff 70%);
    }
    .verdict-heading {
        font-size: 1.45rem;
        font-weight: 800;
        color: #1e1b4b;
        margin: 0 0 0.3rem 0;
    }
    .verdict-desc {
        font-size: 0.96rem;
        color: #475569;
        margin: 0;
    }
    .factor-row {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        background: #ffffff;
        border: 1.5px solid #ede9fe;
        box-shadow: 0 2px 6px rgba(139, 92, 246, 0.05);
        margin-bottom: 0.55rem;
        font-size: 0.95rem;
        color: #1e293b;
    }
    /* Ensure metric cards and tabs stay readable regardless of OS dark mode */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #1e1b4b !important;
    }
    button[data-baseweb="tab"] {
        color: #1e1b4b !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    extractor = BehaviorExtractor(relevance_threshold=0.60)
    model = joblib.load("flag_model.joblib")
    zero_baseline = pd.DataFrame(
        np.zeros((1, len(PILLAR_COLUMNS))), columns=PILLAR_COLUMNS
    )
    explainer = shap.TreeExplainer(
        model, data=zero_baseline, model_output="probability"
    )
    return extractor, model, explainer


extractor, model, explainer = load_pipeline()

# Fun Hero Banner
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🚩 Red Flag or Green Forest? 🌲</div>
        <div class="hero-sub">
            Spill the tea below and let the AI run a 100% local vibe check.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "story_input" not in st.session_state:
    st.session_state.story_input = ""


def clear_story():
    st.session_state.story_input = ""


story = st.text_area(
    "What did they do? Drop the story here:",
    key="story_input",
    height=150,
    placeholder=(
        "e.g., They always remember my coffee order, but vanish for 3 days"
        " whenever I bring up feelings..."
    ),
)

col_btn, col_clear = st.columns([4, 1])
analyze_clicked = col_btn.button(
    "✨ Check the Vibe", type="primary", use_container_width=True
)
col_clear.button("Reset", on_click=clear_story, use_container_width=True)

if analyze_clicked:
    if len(story.strip()) < 10:
        st.warning("Give me a tiny bit more tea—at least one full sentence!")
    else:
        with st.spinner("Reading between the lines & calculating ick levels..."):
            rubric = extractor.extract(story)
            scores_dict = rubric.model_dump()
            input_df = pd.DataFrame([scores_dict])[PILLAR_COLUMNS]

            red_prob = float(model.predict_proba(input_df)[0, 1])
            green_prob = 1.0 - red_prob
            shap_vals = explainer.shap_values(input_df)[0]

        st.divider()

        # Fun Verdict Cards
        if red_prob >= 0.60:
            st.markdown(
                f"""
                <div class="verdict-card verdict-red">
                    <div class="verdict-heading">🚩 Major Red Flag Parade ({red_prob:.0%} Ick Meter)</div>
                    <div class="verdict-desc">Bestie, the warning sirens are loud on this one. Proceed with caution!</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif red_prob <= 0.40:
            st.balloons()
            st.markdown(
                f"""
                <div class="verdict-card verdict-green">
                    <div class="verdict-heading">🌲 Certified Green Forest ({green_prob:.0%} Wholesome)</div>
                    <div class="verdict-desc">Immaculate vibes detected. We love to see healthy behavior!</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="verdict-card verdict-mixed">
                    <div class="verdict-heading">🎲 Situationship Roulette ({red_prob:.0%} Red / {green_prob:.0%} Green)</div>
                    <div class="verdict-desc">Giving mixed signals—some sweet moments, some questionable moves.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Fun KPI Metrics
        active_count = sum(1 for v in scores_dict.values() if v != 0.0)
        m1, m2, m3 = st.columns(3)
        m1.metric("🚩 Red Flag Energy", f"{red_prob:.0%}")
        m2.metric("🌲 Green Flag Energy", f"{green_prob:.0%}")
        m3.metric("🔍 Clues Spotted", f"{active_count} / 9")

        # Playful Tabs
        tab_shap, tab_pillars = st.tabs(
            ["🧾 The Receipts (SHAP)", "🎛️ Full Vibe Scorecard"]
        )

        with tab_shap:
            st.caption(
                "Here is what tipped the scales up (+) toward Red Flag or down"
                " (-) toward Green Forest:"
            )
            active_rows = []
            for col, score, impact in zip(
                PILLAR_COLUMNS, input_df.iloc[0], shap_vals
            ):
                if score != 0.0 and abs(impact) >= 0.015:
                    active_rows.append(
                        {
                            "Trait": PILLAR_LABELS[col],
                            "Score": score,
                            "Vibe Shift (%)": round(impact * 100, 1),
                            "abs_impact": abs(impact),
                        }
                    )

            if active_rows:
                active_rows.sort(key=lambda x: x["abs_impact"], reverse=True)

                chart_df = pd.DataFrame(active_rows).set_index("Trait")[
                    ["Vibe Shift (%)"]
                ]
                st.bar_chart(chart_df, horizontal=True, color="#8b5cf6")

                for row in active_rows:
                    imp = row["Vibe Shift (%)"]
                    sc = row["Score"]
                    if imp > 0:
                        icon, direction, color = (
                            "🚩",
                            f"pushed Red Flag energy up by <b>+{imp:.1f}%</b>",
                            "#e11d48",
                        )
                    else:
                        icon, direction, color = (
                            "💚",
                            (
                                "boosted Green Flag vibes by"
                                f" <b>{abs(imp):.1f}%</b>"
                            ),
                            "#059669",
                        )

                    st.markdown(
                        f"""
                        <div class="factor-row">
                            {icon} <b>{row['Trait']}</b>
                            <span style="color:#6b7280;">(score: <code>{sc:+.2f}</code>)</span>
                            — <span style="color:{color};">{direction}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info(
                    "Hmm, the AI couldn't spot strong red or green clues—try"
                    " adding a bit more detail!"
                )

        with tab_pillars:
            st.caption(
                "How all 9 relationship traits scored from -1.0 (big red flag)"
                " to +1.0 (total green flag):"
            )
            pillar_df = pd.DataFrame(
                [
                    {
                        "Trait": PILLAR_LABELS[c],
                        "Score": scores_dict[c],
                        "Vibe": (
                            "🚩 Red Flag"
                            if scores_dict[c] < 0
                            else (
                                "🌲 Green Flag"
                                if scores_dict[c] > 0
                                else "🤷 Didn't Come Up"
                            )
                        ),
                    }
                    for c in PILLAR_COLUMNS
                ]
            )
            st.dataframe(pillar_df, use_container_width=True, hide_index=True)