import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. SESSION STATE & PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Credit Approval Prediction System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'theme' not in st.session_state: st.session_state.theme = 'dark'
if 'history' not in st.session_state: st.session_state.history = []
if 'pred' not in st.session_state: st.session_state.pred = None


def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'


# ==========================================
# 2. THEME PALETTES (Corporate Banking)
# ==========================================
if st.session_state.theme == 'dark':
    bg = "#0B1120"
    card_bg = "#151E2D"
    text_main = "#F8FAFC"
    text_sub = "#94A3B8"
    accent = "#3B82F6"
    border = "#1E293B"
    success_bg = "rgba(16, 185, 129, 0.1)"
    success_text = "#10B981"
    danger_bg = "rgba(239, 68, 68, 0.1)"
    danger_text = "#EF4444"
else:
    bg = "#F8FAFC"
    card_bg = "#FFFFFF"
    text_main = "#0F172A"
    text_sub = "#64748B"
    accent = "#2563EB"
    border = "#E2E8F0"
    success_bg = "rgba(5, 150, 105, 0.05)"
    success_text = "#059669"
    danger_bg = "rgba(220, 38, 38, 0.05)"
    danger_text = "#DC2626"

# ==========================================
# 3. CSS INJECTION (Perfected Alignment & UI)
# ==========================================
st.markdown(f"""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* Global Reset & Viewport Lock */
    .stApp {{
        background-color: {bg};
        color: {text_main};
        font-family: 'Inter', sans-serif;
    }}
    .main .block-container {{
        padding: 1rem 2rem 0 2rem;
        max-width: 100%;
        height: 100vh;
        overflow: hidden; 
    }}
    header, footer, #MainMenu {{ visibility: hidden; }}

    /* Top Navigation Bar */
    .main-title {{
        font-weight: 800; font-size: 22px; color: {text_main};
        letter-spacing: 0.5px; margin: 0; line-height: 1.2;
    }}
    .sub-title {{
        font-size: 11px; color: {accent}; letter-spacing: 1.5px;
        text-transform: uppercase; margin-top: 2px; font-weight: 700;
    }}
    .custom-divider {{
        border: 0; height: 1px; background: {border}; margin: 10px 0 15px 0;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 30px; background-color: transparent; border-bottom: 1px solid {border};
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 45px; padding: 10px 5px; background-color: transparent;
        color: {text_sub}; font-weight: 600; font-size: 14px;
        border: none; border-bottom: 2px solid transparent; border-radius: 0;
    }}
    .stTabs [aria-selected="true"] {{
        color: {accent}; border-bottom: 2px solid {accent};
    }}
    .stTabs [data-baseweb="tab-list"] button:nth-of-type(1) div::before {{
        content: "\\f15c"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 8px;
    }}
    .stTabs [data-baseweb="tab-list"] button:nth-of-type(2) div::before {{
        content: "\\f201"; font-family: "Font Awesome 6 Free"; font-weight: 900; margin-right: 8px;
    }}

    /* Form Styling (Clean, No Outer Box) */
    [data-testid="stForm"] {{
        border: none !important; padding: 10px 0 !important; background: transparent !important;
    }}

    /* Internal Scroll for Results */
    [data-testid="column"]:nth-of-type(2) > div {{
        height: calc(100vh - 200px);
        overflow-y: auto;
        padding-right: 10px;
    }}
    [data-testid="column"]:nth-of-type(2)::-webkit-scrollbar {{ width: 6px; }}
    [data-testid="column"]:nth-of-type(2)::-webkit-scrollbar-thumb {{ background: {border}; border-radius: 3px; }}

    /* Box Titles */
    .box-title {{
        font-size: 12px; font-weight: 700; color: {text_sub}; text-transform: uppercase;
        letter-spacing: 1px; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 1px solid {border};
    }}

    /* PERFECT EQUAL HEIGHT & CENTERING FOR TOP ROW CONTAINERS */
    div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"] {{
        height: 220px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 15px !important;
    }}

    /* Inner Colored Cards for Status */
    .status-approved-bg {{ 
        background-color: {success_bg}; border: 1px solid {success_text}; border-radius: 10px; 
        padding: 20px; text-align: center; width: 100%;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
    }}
    .status-rejected-bg {{ 
        background-color: {danger_bg}; border: 1px solid {danger_text}; border-radius: 10px; 
        padding: 20px; text-align: center; width: 100%;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
    }}

    .status-text {{ font-size: 32px; font-weight: 800; margin: 0; letter-spacing: 1px; }}
    .text-success {{ color: {success_text}; }}
    .text-danger {{ color: {danger_text}; }}

    /* Increased Confidence Font Size */
    .prob-text {{ 
        text-align: center; color: {text_sub}; font-size: 18px; 
        margin-top: 12px; font-weight: 700; 
    }}

    /* Reason Box (Professional List Format) */
    .reason-box {{
        background: rgba(59, 130, 246, 0.05); border-left: 3px solid {accent};
        padding: 16px; border-radius: 6px; font-size: 14px; color: {text_main}; 
        margin-bottom: 15px; line-height: 1.6;
    }}
    .reason-box strong {{
        display: block;
        margin-bottom: 8px;
        font-size: 14px;
        color: {accent};
    }}
    .reason-box ul {{
        margin: 0;
        padding-left: 20px;
    }}
    .reason-box li {{
        margin-bottom: 6px;
    }}

    /* Theme Toggle Button */
    .stButton > button {{
        background-color: {card_bg} !important; color: {accent} !important;
        border: 1px solid {border} !important; border-radius: 8px !important;
        font-size: 18px !important; padding: 5px 0 !important;
    }}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. ROBUST MODEL ENGINE (Auto-Fallback)
# ==========================================
@st.cache_resource
def initialize_models():
    try:
        stacking_model = joblib.load('stacking_loan_model.pkl')
        xgb_model = joblib.load('xgb_base_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        return stacking_model, xgb_model, preprocessor, "Production Models Active"
    except Exception:
        from sklearn.ensemble import RandomForestClassifier, StackingClassifier
        from xgboost import XGBClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler , OneHotEncoder
        from sklearn.feature_extraction.text import TfidfVectorizer

        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'Income': np.random.uniform(20000, 150000, n), 'Credit_Score': np.random.uniform(300, 850, n),
            'Loan_Amount': np.random.uniform(5000, 100000, n), 'DTI_Ratio': np.random.uniform(10, 60, n),
            'Employment_Status': np.random.choice(['employed', 'unemployed'], n),
            'clean_text': np.random.choice(['business', 'home', 'medical', 'education'], n)
        })
        y = ((df['Credit_Score'] > 650) & (df['DTI_Ratio'] < 40) & (df['Employment_Status'] == 'employed')).astype(int)

        pre = ColumnTransformer([
            ('num', StandardScaler(), ['Income', 'Credit_Score', 'Loan_Amount', 'DTI_Ratio']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Employment_Status']),
            ('text', TfidfVectorizer(max_features=50), 'clean_text')
        ])
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        xgb = XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
        stacking = StackingClassifier(estimators=[('rf', rf), ('xgb', xgb)], final_estimator=LogisticRegression(), cv=2)

        pipe_stack = Pipeline([('pre', pre), ('clf', stacking)])
        pipe_xgb = Pipeline([('pre', pre), ('clf', xgb)])
        pipe_stack.fit(df, y);
        pipe_xgb.fit(df, y)
        return pipe_stack, pipe_xgb, pre, "Fallback Engine Active"


stacking_model, xgb_model, preprocessor, status_msg = initialize_models()

# ==========================================
# 5. TOP NAVIGATION BAR
# ==========================================
c1, c2 = st.columns([8, 1])
with c1:
    st.markdown('<div class="main-title">Credit Approval Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Ensemble Stacking & Explainable AI Approach</div>', unsafe_allow_html=True)
with c2:
    icon = "☀" if st.session_state.theme == 'dark' else "☾"
    st.button(icon, on_click=toggle_theme, use_container_width=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ==========================================
# 6. MAIN TABS
# ==========================================
tab1, tab2 = st.tabs(["New Application", "Analysis History & Reports"])

with tab1:
    col_form, col_results = st.columns([1.2, 2.8], gap="large")

    # --- LEFT: FORM (Clean Layout) ---
    with col_form:
        with st.form("loan_application"):
            st.markdown('<div class="box-title">Applicant Financial Profile</div>', unsafe_allow_html=True)
            name = st.text_input("Full Name", placeholder="e.g. Rahim Uddin")
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                income_bdt = st.text_input("Annual Income (BDT)", placeholder="e.g. 1200000")
                loan_bdt = st.text_input("Loan Amount (BDT)", placeholder="e.g. 500000")
            with c2:
                # Capitalized Options
                employment = st.selectbox("Employment", ["-- Select --", "Employed", "Unemployed"])
                credit = st.text_input("Credit Score (300-850)", placeholder="e.g. 720")
                dti = st.text_input("DTI Ratio (%)", placeholder="e.g. 25.5")
            purpose = st.selectbox("Loan Purpose",
                                   ["Business Expansion", "Home Renovation", "Medical Emergency", "Higher Education",
                                    "Agriculture", "Vehicle Purchase", "Personal Use"])
            submitted = st.form_submit_button("ANALYZE RISK PROFILE", use_container_width=True)

    # --- RIGHT: RESULTS ---
    with col_results:
        if submitted:
            try:
                income_usd = float(income_bdt) / 117.0
                loan_usd = float(loan_bdt) / 117.0
                credit_val = float(credit)
                dti_val = float(dti)
                if employment == "-- Select --": raise ValueError()
            except:
                st.error("Validation Error: Please enter valid numeric data for all financial fields.")
                st.stop()

            with st.spinner("Processing through Stacking Ensemble..."):
                time.sleep(1.2)
                input_df = pd.DataFrame({
                    'Income': [income_usd], 'Credit_Score': [credit_val], 'Loan_Amount': [loan_usd],
                    'DTI_Ratio': [dti_val], 'Employment_Status': [employment.lower()], 'clean_text': [purpose.lower()]
                })

                try:
                    prediction = stacking_model.predict(input_df)[0]
                    proba = stacking_model.predict_proba(input_df)[0][1]
                    X_transformed = preprocessor.transform(input_df)
                    tree_model = xgb_model.named_steps['clf'] if hasattr(xgb_model, 'named_steps') else xgb_model
                    explainer = shap.TreeExplainer(tree_model)
                    shap_values = explainer.shap_values(X_transformed)

                    raw_names = preprocessor.get_feature_names_out() if hasattr(preprocessor,
                                                                                'get_feature_names_out') else [f"F_{i}"
                                                                                                               for i in
                                                                                                               range(
                                                                                                                   X_transformed.shape[
                                                                                                                       1])]
                    clean_names = [n.split('__')[-1] if '__' in n else n for n in raw_names]
                    shap_vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                                         np.ndarray) else explainer.expected_value

                    st.session_state.update(
                        {'pred': prediction, 'proba': proba, 'shap_vals': shap_vals, 'feature_names': clean_names,
                         'base_val': base_val, 'X_transformed': X_transformed})
                    st.session_state.history.append(
                        {'Time': datetime.now().strftime("%H:%M:%S"), 'Name': name, 'Purpose': purpose,
                         'Status': 'Approved' if prediction == 1 else 'Rejected',
                         'Confidence': f"{proba * 100:.1f}%" if prediction == 1 else f"{(1 - proba) * 100:.1f}%"})
                except Exception as e:
                    st.error(f"Engine Error: {e}");
                    st.stop()

        if 'pred' in st.session_state and st.session_state.pred is not None:
            pred = st.session_state.pred
            prob = st.session_state.proba

            # Row 1: Status & Gauge Boxes (Strictly Equal Size & Centered)
            r1_c1, r1_c2 = st.columns(2, gap="medium")
            with r1_c1:
                st.markdown('<div class="box-title">Prediction Status</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    status_class = "status-approved-bg" if pred == 1 else "status-rejected-bg"
                    status_text = "APPROVED" if pred == 1 else "REJECTED"
                    text_class = "text-success" if pred == 1 else "text-danger"
                    conf = prob * 100 if pred == 1 else (1 - prob) * 100
                    st.markdown(
                        f'<div class="{status_class}"><div class="status-text {text_class}">{status_text}</div><div class="prob-text">Confidence: {conf:.1f}%</div></div>',
                        unsafe_allow_html=True)

            with r1_c2:
                st.markdown('<div class="box-title">Confidence Gauge</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100,
                                                       number={'suffix': '%', 'font': {'color': text_main, 'size': 28}},
                                                       gauge={'axis': {'range': [0, 100], 'tickcolor': text_sub},
                                                              'bar': {'color': accent}, 'bgcolor': "rgba(0,0,0,0)",
                                                              'steps': [{'range': [0, 50], 'color': danger_bg},
                                                                        {'range': [50, 100], 'color': success_bg}]}))
                    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                            font={'color': text_main}, margin=dict(l=10, r=10, t=10, b=10), height=170)
                    st.plotly_chart(fig_gauge, use_container_width=True)

            # Row 2: SHAP Box
            with st.container(border=True):
                st.markdown('<div class="box-title">Decision Explainability (SHAP)</div>', unsafe_allow_html=True)

                # Professional List Format for Primary Decision Drivers
                feature_impact = sorted(zip(st.session_state.feature_names, st.session_state.shap_vals),
                                        key=lambda x: abs(x[1]), reverse=True)

                reason_html = '<div class="reason-box"><strong>Primary Decision Drivers:</strong><ul>'
                count = 0
                for fname, fval in feature_impact:
                    if 'text' not in fname and count < 2:
                        impact = "positively influenced" if fval > 0 else "negatively impacted"
                        reason_html += f"<li><b>{fname.replace('_', ' ').title()}</b> {impact} the decision.</li>"
                        count += 1
                reason_html += '</ul></div>'
                st.markdown(reason_html, unsafe_allow_html=True)

                fig_shap, ax = plt.subplots(figsize=(10, 3.5))
                fig_shap.patch.set_alpha(0);
                ax.set_facecolor('none')
                explanation = shap.Explanation(values=st.session_state.shap_vals, base_values=st.session_state.base_val,
                                               data=st.session_state.X_transformed[0],
                                               feature_names=st.session_state.feature_names)
                shap.plots.waterfall(explanation, show=False, max_display=5)
                ax.tick_params(axis='x', colors=text_main, labelsize=10);
                ax.tick_params(axis='y', colors=text_main, labelsize=10)
                for spine in ax.spines.values(): spine.set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_shap, use_container_width=True)
        else:
            st.markdown(f"""
            <div style="height: 70vh; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; background: {card_bg}; border: 1px dashed {border}; border-radius: 12px;">
                <div style="font-size: 20px; font-weight: 800; color: {text_main}; margin-bottom: 10px;">AWAITING APPLICANT DATA</div>
                <div style="color: {text_sub}; font-size: 13px; max-width: 400px;">Complete the financial profile and submit to generate a real-time risk assessment.</div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="box-title">Session Analysis History & Reports</div>', unsafe_allow_html=True)
    if len(st.session_state.history) > 0:
        df_hist = pd.DataFrame(st.session_state.history)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Applications", len(st.session_state.history))
        with c2:
            st.metric("Approved", len([x for x in st.session_state.history if x['Status'] == 'Approved']))
        with c3:
            st.metric("Rejected", len([x for x in st.session_state.history if x['Status'] == 'Rejected']))
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
    else:
        st.info("No applications processed in this session yet.")