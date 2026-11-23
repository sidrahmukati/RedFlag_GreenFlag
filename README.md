# 🚩 Red Flag or Green Forest? 🌲

A 100% local, explainable relationship vibe-check web app powered by **Zero-Shot NLI (`DeBERTa-v3-small`)**, **Pydantic**, **XGBoost**, **SHAP**, and **Streamlit**.

Paste any relationship story in plain English, and the pipeline extracts structured behavioral signals across 9 relationship pillars, predicts Red Flag vs. Green Flag risk, and shows the exact SHAP attributions behind every score.

---

## How It Works (Two-Stage Pipeline)

1. **Stage 1 — Behavioral Signal Extraction (`extractor.py` & `schema.py`)**
   - Splits input stories at sentence boundaries and contrast conjunctions (`but`, `however`, `yet`, `although`).
   - Uses `MoritzLaurer/DeBERTa-v3-small-mnli-fever-docnli-ling-2c` locally on CPU to evaluate positive and negative behavioral hypotheses across **9 core pillars** (`-1.0` to `+1.0`).
   - Applies a Top-2 pillar cap per clause to prevent single-sentence "halo effects" and validates outputs via **Pydantic** (`BehavioralRubric`).

2. **Stage 2 — Risk Prediction & Explainability (`train_model.py` & `app.py`)**
   - Uses a pre-trained **XGBoost** classifier (`flag_model.joblib`) trained via 5-fold stratified cross-validation on synthetic behavioral profiles (`generate_data.py`).
   - Computes **SHAP (`TreeExplainer`)** probability attributions relative to an all-zero neutral baseline so users see which specific traits pushed the score up or down.

---

## Project Structure

```text
├── app.py               # Streamlit web interface with SHAP breakdowns & 9-pillar scorecard
├── extractor.py         # Local Hugging Face DeBERTa-v3-small zero-shot signal extractor
├── schema.py            # Pydantic validation schema & 9 behavioral pillar definitions
├── flag_model.joblib    # Pre-trained XGBoost classifier ready for immediate execution
├── generate_data.py     # Synthetic behavioral dataset generator (optional retraining)
├── train_model.py       # 5-fold StratifiedKFold XGBoost training script (optional retraining)
└── requirements.txt     # Project dependencies
```
Setup & Local Execution
1. Clone and Install Dependencies
```Bash
git clone https://github.com/sidrahmukati/RedFlag_GreenFlag.git
cd RedFlag_GreenFlag
pip install -r requirements.txt
```

3. Launch the Streamlit App
```Bash
streamlit run app.py
```
Note: flag_model.joblib is already included in the repository so you can run the app immediately. On the very first run, Hugging Face will download DeBERTa-v3-small (~280 MB) and cache it locally for offline use.

(Optional) To regenerate the synthetic dataset and retrain the XGBoost model from scratch:
```bash
python generate_data.py
python train_model.py
```

## The 9 Behavioral Pillars
Pillar Key	UI Label	What It Measures
```bash
public_respect	Public Flex vs. Public Diss	Dignity and respect vs. public humiliation or mockery
public_private_consistency	Texting vs. IRL Consistency	Consistent affection vs. hot-and-cold ghosting
social_integrity	Gossip & Two-Faced Radar	Loyalty and honesty vs. talking behind people's backs
honesty_and_transparency	No-Cap Honesty	Open communication vs. excuses, lies, or hiding things
emotional_attentiveness	Golden Retriever Energy	Active care and attentiveness vs. emotional neglect
empathy_and_validation	Vibe & Feelings Check	Listening with empathy vs. dismissing or mocking feelings
accountability	Owning Their Mess	Sincere apologies vs. blame-shifting ("you're too sensitive")
boundary_respect	Respecting Space	Respecting consent and space vs. pressuring boundaries
safety_catchall	Safe Space Energy	Trustworthy environment vs. controlling or manipulative behavior
```

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://redflaggreenflag.streamlit.app/)

**Live App:** [redflaggreenflag.streamlit.app](https://redflaggreenflag.streamlit.app/)

## Disclaimer: This app provides automated ML and zero-shot NLI estimates for informational and educational purposes only. It is not clinical, psychological, medical, or legal advice, and should not be used to diagnose behavior or make high-stakes personal decisions.



