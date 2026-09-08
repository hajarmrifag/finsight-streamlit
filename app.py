"""FinSight AI: a public historical banking research portfolio."""
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '5')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '15')
import pandas as pd
import streamlit as st
from core import campaign, semantic_search, word_search

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='FinSight AI', page_icon='◒', layout='wide', menu_items={'About':'FinSight AI — historical banking disengagement research by Hajar Mrifag.'})


st.markdown("""
<style>
.stApp { background: #f6f2eb; color: #202b38; }
[data-testid="stHeader"] { background: #f6f2ebee; }
[data-testid="stSidebar"] { background: #202b38; border-right: 0; }
[data-testid="stSidebar"] * { color: #f6f2eb; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #b9b6af; }
[data-testid="stSidebar"] button { background: transparent; border: 1px solid #74808b; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .8rem; margin: 2rem 0; }
.block-container { max-width: 1240px; padding-top: 3.5rem; padding-bottom: 4rem; }
h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif !important; font-weight: 400 !important; letter-spacing: -.035em; }
h1 { font-size: clamp(2.6rem, 5vw, 4.6rem) !important; line-height: 1.06 !important; max-width: 850px; }
h3 { font-size: 1.8rem !important; margin-top: 1.5rem !important; }
.kicker { font-size: .72rem; letter-spacing: .2em; text-transform: uppercase; color: #a94e32; margin-bottom: 1.2rem; font-weight: 700; }
.masthead { font-family: Georgia, serif; font-size: 2.7rem; letter-spacing: -.07em; padding-top: 1rem; }
.edition { border-top: 1px solid #b8b0a4; border-bottom: 1px solid #b8b0a4; padding: .65rem 0; margin: 1.4rem 0 2rem; font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
[data-testid="stMetric"] { border-top: 2px solid #a94e32; padding: 1.1rem .2rem 1.5rem; }
[data-testid="stMetricValue"] { font-family: Georgia, serif; font-size: 2.7rem; }
[data-testid="stMetricLabel"] { color: #626972; font-size: .8rem; }
[data-testid="stForm"] { background: #ece6dc; border: 0; border-radius: 2px; padding: 1.5rem; }
.stButton button, .stDownloadButton button, .stFormSubmitButton button { border-radius: 2px; min-height: 2.8rem; }
.stFormSubmitButton button { background: #a94e32; color: white; border-color: #a94e32; }
[data-testid="stAlert"] { border-radius: 2px; }
[data-testid="stDataFrame"] { border-top: 1px solid #b8b0a4; }
[data-testid="stCaptionContainer"] { color: #626972; }
@media(max-width: 640px) { .block-container { padding-top: 4rem; } [data-testid="stMetricValue"] { font-size: 2rem; } }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    report = json.loads((ROOT/'data/research.json').read_text())
    knowledge = json.loads((ROOT/'data/knowledge.json').read_text())
    return report, knowledge


@st.cache_resource
def load_encoder():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(1)
    try:
        return SentenceTransformer('all-MiniLM-L6-v2', device='cpu', local_files_only=True)
    except OSError:
        return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


def evidence(matches, mode):
    st.caption(mode)
    if not matches:
        st.info('No sufficiently relevant evidence found. Try a question about the model, cohorts, retention, or research limitations.')
    for item in matches:
        with st.container(border=True):
            st.subheader(item['title'])
            st.write(item['text'])
            suffix = f" · Relevance {item['score']:.3f}" if 'score' in item else ''
            st.caption(item['source'] + suffix)


try:
    report, knowledge = load_data()
except (OSError, ValueError):
    st.error('Research files are unavailable. Restore the aggregate data files and reload the app.')
    st.stop()

with st.sidebar:
    st.markdown('<div class="masthead">FinSight.</div>', unsafe_allow_html=True)
    st.caption('RESEARCH BY HAJAR MRIFAG')
    page = st.radio('Explore', ['Overview', 'Evidence analyst', 'Campaign scenario', 'Methodology'], key='page')
    st.divider()
    st.caption('BERKA BANKING STUDY / 1998')
    st.download_button('Download aggregate report', json.dumps(report, indent=2), file_name='finsight-research.json', mime='application/json')

st.markdown('<div class="kicker">FinSight / Banking research</div>', unsafe_allow_html=True)
st.title(page if page != 'Overview' else 'Before an account goes quiet.')
st.markdown('<div class="edition"><span>Customer activity & retention</span><span>Historical study · July–September 1998</span></div>', unsafe_allow_html=True)
e = report['evaluation']

if page == 'Overview':
    st.write('A study of declining transaction activity: which accounts show the strongest signs of disengagement, and what would it take for a retention campaign to pay off?')
    st.caption('Historical activity rankings. These results do not measure creditworthiness or prove campaign effectiveness.')
    for column, label, value, help_text in zip(st.columns(4),
        ['Holdout ROC-AUC','Cases captured','Top-10% lift','Latest high-risk accounts'],
        [f"{e['roc_auc']:.4f}",f"{e['recall']:.2%}",f"{e['lift']:.2f}×",f"{report['high_risk_accounts']:,}"],
        ['13,177 held-out account-months',f"{e['positives_captured']} of {e['positives']} cases",f"Precision {e['precision']:.2%}",f"Of {report['latest_accounts']:,} eligible accounts"]):
        column.metric(label, value, help=help_text)
    st.subheader('Historical cohorts')
    months = pd.DataFrame(report['months']).rename(columns={'snapshot':'Snapshot','accounts':'Accounts','high_risk':'High risk','observed_disengagement_rate':'Observed disengagement rate'})
    st.dataframe(months, hide_index=True, width='stretch', column_config={'Observed disengagement rate': st.column_config.NumberColumn(format='percent')})
    chart = months.set_index('Snapshot')[['Observed disengagement rate']].mul(100)
    st.line_chart(chart, y_label='Observed disengagement (%)')
    st.caption('Accounts repeat across months. High risk is the top 10% within each monthly cohort; holdout metrics rank all test account-month observations together.')
    st.subheader('The behavior behind the ranking')
    st.caption(f"{report['latest_snapshot']} · group medians; balances in source currency")
    profile = pd.DataFrame(report['profile']).rename(columns={'feature':'Behavior','high_risk_median':'High-risk median','other_median':'Other accounts median'})
    profile['Behavior'] = profile['Behavior'].str.replace('_',' ').str.replace('90d','(90 days)').str.replace('30d','(30 days)')
    st.dataframe(profile, hide_index=True, width='stretch')
    st.caption('Group differences are associations, not causal explanations or individual feature attributions.')

elif page == 'Evidence analyst':
    st.write('Search the reproduced research and retention notes, or open a quick answer.')
    quick = {'Model performance':'performance','High-risk account count':'counts','Retention strategies':'retention','Research limitations':'limits'}
    cols = st.columns(2)
    for index, (label, id) in enumerate(quick.items()):
        if cols[index % 2].button(label, key=id, width='stretch'):
            st.session_state['answer'] = ([item for item in knowledge if item['id'] == id], 'Verified research')
    with st.form('search_form'):
        question = st.text_area('Your question', max_chars=500, placeholder='How effective is the model at finding disengagement?', key='question')
        semantic = st.checkbox('Search by meaning', value=True, help='Loads the MiniLM model once. Questions are processed on the app server, without an external LLM API.')
        submitted = st.form_submit_button('Find evidence')
    if submitted:
        question = question.strip()
        if not question:
            st.warning('Enter a question before searching.')
        else:
            mode = 'Word matching'
            if semantic:
                try:
                    with st.spinner('Searching by meaning. First use may download the model…'):
                        matches = semantic_search(question, knowledge, load_encoder())
                    mode = 'Semantic search'
                except (OSError, ValueError, RuntimeError, ImportError):
                    matches = word_search(question, knowledge)
                    mode = 'Word matching — semantic model unavailable'
            else:
                matches = word_search(question, knowledge)
            st.session_state['answer'] = (matches, mode)
    if 'answer' in st.session_state:
        evidence(*st.session_state['answer'])
    st.caption('Answers are extracted evidence, not generated financial advice. Similarity measures relevance, not certainty. Avoid entering private customer information into this public demo.')

elif page == 'Campaign scenario':
    st.write('Estimate a break-even point before assuming a campaign creates value.')
    with st.form('campaign_form'):
        a,b = st.columns(2)
        targeted = a.number_input('Accounts contacted', min_value=1, max_value=1000000, value=int(e['targeted_rows']), key='targeted')
        precision = b.number_input('Share actually disengaging (%)', min_value=0., max_value=100., value=round(e['precision']*100,2), step=.01, key='precision')
        save_rate = a.number_input('Assumed save rate (%)', min_value=0., max_value=100., value=20., key='save_rate')
        cost = b.number_input('Cost per contact (value units)', min_value=0., max_value=1000000., value=2., key='cost')
        value = a.number_input('Value per successful save (value units)', min_value=0., max_value=10000000., value=100., key='value')
        calculate = st.form_submit_button('Calculate scenario')
    if calculate or 'scenario' not in st.session_state:
        st.session_state['scenario'] = campaign(targeted, precision/100, save_rate/100, cost, value)
    scenario = st.session_state['scenario']
    cols = st.columns(4)
    for col,label,val in zip(cols,['Expected saves','Campaign cost','Expected net value','Break-even value per save'],
        [f"{scenario['expected_saves']:,.2f}",f"{scenario['cost']:,.2f}",f"{scenario['net']:,.2f}",'No expected saves' if scenario['break_even'] is None else f"{scenario['break_even']:,.2f}"]):
        col.metric(label,val)
    st.caption('Results use the last submitted assumptions. All financial values are common arbitrary value units. Starting precision is the holdout top-10% estimate, not a guaranteed future campaign rate.')
    st.warning('Save rate, cost, and retained value are assumptions. No measured intervention effect or customer lifetime value is available.')

else:
    st.subheader('Model card')
    st.write(f"{report['model']} trained on {report['training_rows']:,} development observations with {report['feature_count']} features.")
    st.write('Development: ' + ' to '.join(report['development_period']))
    st.write('Holdout: ' + ' to '.join(report['test_period']))
    st.subheader('Target definition')
    st.write(report['target_definition'])
    st.subheader('Limits of the evidence')
    for text in report['limitations']:
        st.markdown('- ' + text)
    st.subheader('Provenance')
    st.markdown('[Berka / PKDD 1999 Financial dataset](https://relational.fel.cvut.cz/dataset/Financial) · [Original research repository](https://github.com/hajarmrifag/churn-reactivation-engine)')
    st.write('The original Python workflow trained and saved the model and scored historical account-months. This public app explores the resulting aggregate artifacts; no account-level records or fitted risk model are included.')
    st.caption('Semantic search uses MiniLM sentence embeddings. No API key or paid inference service is required.')
