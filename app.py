"""Unified Tox21 project dashboard: evidence, models and saved-model inference."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
P2 = ROOT / 'phase2'
DATA = ROOT / 'tox21_member2_handoff/artifacts'
EVAL = P2 / 'member4_evaluation'
DEMO = P2 / 'member5_demo_submission'

st.set_page_config(page_title='Tox21 | Molecular Intelligence', page_icon='🧬', layout='wide')
st.markdown('''<style>
.block-container {max-width:1250px;padding-top:2rem;}
[data-testid="stMetric"] {background:#edf5f5;border-radius:10px;padding:16px;}
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricValue"] * {color:#123747 !important;}
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricLabel"] * {color:#365867 !important;}
h1,h2,h3 {color:#123747;}
[data-testid="stSidebar"] {background:#123747;color:#f1f7fa;}
[data-testid="stSidebar"] :is(h1,h2,h3,p,label,span,small) {color:#f1f7fa !important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:#c5d8e2 !important;}
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding:10px 12px;border-radius:8px;margin-bottom:5px;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#205166;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background:#206274;outline:1px solid #64d3cb;
}
[data-testid="stSidebar"] button {color:#f1f7fa;}
[data-testid="stSidebar"] hr {border-color:#527584;}
</style>''', unsafe_allow_html=True)

@st.cache_data
def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_data
def metadata():
    return pd.read_parquet(DATA / 'labels_and_splits.parquet')

def table_file(path):
    frame = read_csv(path)
    if frame.empty:
        st.info('This result is not available in the current checkout.')
    else:
        st.dataframe(frame, hide_index=True, use_container_width=True)
    return frame

def plot(path, caption):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)

with st.sidebar:
    st.title('🧬 Tox21')
    st.caption('MOLECULAR ACTIVITY PREDICTION')
    page = st.radio('Explore the project', ['Overview', 'Data & molecular features', 'Model architectures', 'Experiments & results', 'Predict a molecule', 'Report & presentation'])
    st.divider()
    st.write('**Machine learning + deep learning**')
    st.caption('12 biological assays · one frozen scaffold split · reproducible inference')

st.title(page)
metrics = read_csv(DEMO / 'models/inference_metrics.csv')
summary = metrics[metrics['split'].eq('test')].groupby('model')[['roc_auc','average_precision']].mean() if not metrics.empty else pd.DataFrame()

if page == 'Overview':
    st.markdown('## Can molecular structure predict biological activity?')
    st.write('We compare classical classifiers, a multitask neural network, and a hybrid Random Forest across 12 Tox21 assays. The project follows molecules from structure validation through controlled experiments to predictions from saved models.')
    a,b,c,d = st.columns(4)
    a.metric('Valid molecules','7,823'); b.metric('Assay endpoints','12'); c.metric('Input features','2,053'); d.metric('Test molecules','782')
    st.subheader('The complete workflow')
    st.graphviz_chart('digraph {rankdir=LR; node [shape=box, style="rounded,filled",fillcolor="#edf5f5",fontname="Arial"]; "SMILES" -> "Clean & scaffold split" -> "Fingerprints + descriptors" -> "ML / multitask DL / hybrid" -> "Evaluate & predict";}')
    left,right = st.columns([1.5,1])
    with left:
        st.subheader('Verified inference models')
        if not summary.empty:
            st.bar_chart(summary.rename(columns={'roc_auc':'ROC-AUC','average_precision':'Average precision'}), color=['#153d50','#00a69c'])
    with right:
        st.subheader('What we learned')
        st.write('**Random Forest remains the strongest single model** in the verified inference bundle, where the hybrid concatenates neural features with descriptors and does not exceed it. Member 3 has since shown that combining the two models at the *probability* level does beat the Phase 1 baseline (test ROC-AUC 0.7685 vs 0.7541); that ensemble is reported in Experiments & results and is not part of this live bundle.')
        st.write('**Reproducibility matters.** Integration checks uncovered a checkpoint/embedding mismatch. The demo uses a separately rebuilt, consistent model bundle.')
        st.caption('Scores shown here describe the live demo models. Historical and three-seed experiments are labelled separately in Experiments & results.')
    st.info('Use “Predict a molecule” to compare RF, neural-network and hybrid scores for a new SMILES string.')

elif page == 'Data & molecular features':
    frame = metadata(); endpoints = [c for c in frame if c.startswith(('NR-','SR-'))]
    st.write('The curated dataset contains 7,823 valid molecular structures. Unmeasured assay labels remain missing and are excluded from loss and evaluation.')
    a,b = st.columns(2)
    with a:
        st.subheader('Frozen scaffold partition')
        st.dataframe(pd.DataFrame({'Partition':['Train','Validation','Test'],'Molecules':[6259,782,782],'Purpose':['Fit models and scaler','Select settings and checkpoint','Final evaluation']}),hide_index=True,use_container_width=True)
        st.caption('All live models use the same artifact split. The earlier raw-data manifest is a different partition and is not mixed into these comparisons.')
    with b:
        st.subheader('Observed and missing labels')
        counts = pd.DataFrame({'Active':frame[endpoints].eq(1).sum(),'Inactive':frame[endpoints].eq(0).sum(),'Missing':frame[endpoints].isna().sum()})
        st.bar_chart(counts)
    st.subheader('A molecule becomes 2,053 numbers')
    st.dataframe(pd.DataFrame({'Representation':['Morgan fingerprint','Molecular descriptors'],'Size':[2048,5],'Meaning':['Radius-2 local substructures','MolWt, TPSA, MolLogP, H-bond donors and acceptors'],'Preprocessing':['Binary bits','Standardized using training statistics only']}),hide_index=True,use_container_width=True)
    assay = st.selectbox('Inspect assay labels',endpoints)
    st.dataframe(frame[['mol_id','smiles','split',assay]].head(100),hide_index=True,use_container_width=True)

elif page == 'Model architectures':
    ml,dl,hybrid = st.tabs(['Classical machine learning','Multitask neural network','ML + DL hybrid'])
    with ml:
        st.subheader('Three interpretable reference methods')
        st.dataframe(pd.DataFrame({'Model':['Logistic Regression','Decision Tree','Random Forest'],'Configuration':['Balanced classes; max_iter=1000','Balanced classes; max_depth=10','100 trees; max_depth=12; balanced classes'],'Training':['One model per endpoint']*3}),hide_index=True,use_container_width=True)
        st.write('Each classifier learns from observed labels for its endpoint. Random Forest captures nonlinear feature interactions directly from the original molecular fingerprint and descriptors.')
    with dl:
        config=json.loads((P2/'member2_deep_learning_model/configs/model_config.json').read_text())
        st.subheader('One shared network, 12 independent outputs')
        st.graphviz_chart('digraph {rankdir=LR; node [shape=box,style="rounded,filled",fillcolor="#edf5f5",fontname="Arial"]; "2053 inputs" -> "512 + BN + ReLU" -> "256 + BN + ReLU" -> "128 bottleneck" -> "12 logits" -> "Sigmoid scores";}')
        st.write('The shared layers learn representations across assays. Each output is an independent binary activity prediction, so multiple assays can be active for one molecule.')
        a,b,c=st.columns(3); a.metric('Trunk dropout','0.35'); b.metric('Bottleneck dropout','0.175'); c.metric('Batch size','64')
        st.markdown('**Masked weighted BCE:** missing labels contribute no loss; positive-class weights come from observed training counts. Adam, weight decay and validation-based early stopping control optimization.')
        history=read_csv(P2/'member2_deep_learning_model/reports/training_history.csv')
        if not history.empty:
            a,b=st.columns(2)
            with a: st.subheader('Training and validation loss'); st.line_chart(history.set_index('epoch')[['train_loss','val_loss']])
            with b: st.subheader('Validation ranking metrics'); st.line_chart(history.set_index('epoch')[['val_macro_roc_auc','val_macro_pr_auc']])
            st.caption('Original training history. The inference bundle loads the frozen on-disk checkpoint; its measured results are reported separately.')
        with st.expander('Exact training configuration'): st.json(config['training'])
    with hybrid:
        st.subheader('Neural representation + endpoint-specific forests')
        st.graphviz_chart('digraph {rankdir=LR; node [shape=box,style="rounded,filled",fillcolor="#edf5f5",fontname="Arial"]; "Frozen neural encoder" -> "128 learned features" -> "133 features"; "5 scaled descriptors" -> "133 features"; "133 features" -> "12 Random Forests" -> "Assay scores";}')
        st.write('The hypothesis is that shared neural features can help a forest generalize across scaffolds. Forests use 100 trees, depth 12 and balanced class weights. Only observed training labels are used.')
        st.info('This feature-concatenation hybrid does not outperform the standalone RF or DL model: Random Forests pick splits by impurity gain, so dense neural features crowd out the sparse Morgan bits. Blending the two models\' probabilities instead (alpha=0.80) does beat the Phase 1 baseline — see Experiments & results.')

elif page == 'Experiments & results':
    current,historical,ablations = st.tabs(['Verified demo bundle','Historical benchmark','Ablation studies'])
    with current:
        st.write('These results describe the exact model bundle used for live predictions, evaluated on the frozen test partition.')
        if not summary.empty:
            st.dataframe(summary.round(4),use_container_width=True)
            measure=st.selectbox('Per-endpoint metric',['roc_auc','average_precision'])
            st.bar_chart(metrics[metrics['split'].eq('test')].pivot(index='endpoint',columns='model',values=measure))
        with st.expander('Why these scores differ from the historical benchmark'):
            st.write('The original embedding files did not match the saved neural checkpoint. The trainer now restores the on-disk weights, and the live hybrid was rebuilt from matching embeddings using training labels only. Historical artifacts are preserved. Software-version differences may also affect refitted RF results.')
    with historical:
        st.caption('Original team experiments; do not treat these as live-bundle metrics.')
        table_file(EVAL/'results/final_model_comparison.csv')
        plot(ROOT/'results/plots/scaffold_vs_random_split.png','Historical experiment: scaffold versus random split')
    with ablations:
        choice=st.selectbox('Controlled comparison',['Multitask versus single-task','Dropout','Class weighting'])
        files={'Multitask versus single-task':'multitask_vs_single_task_summary.csv','Dropout':'dropout_ablation_summary.csv','Class weighting':'class_weight_ablation_summary.csv'}
        table_file(EVAL/'results'/files[choice])
        st.caption('Separate three-seed studies. Standard deviations describe variation across training seeds, not confidence intervals. Recall and F1 depend on the decision threshold.')
        st.write('The three-seed multitask study has higher mean ROC-AUC than single-task learning. Removing class weights increases missed actives at the evaluated operating point. Removing dropout produces mixed metric changes, so no universal benefit is claimed.')

elif page == 'Predict a molecule':
    st.write('Enter a molecular structure and compare three saved models across all 12 assays.')
    presets={'Custom input':'','Aspirin':'CC(=O)Oc1ccccc1C(=O)O','Ethanol':'CCO','Bisphenol A':'CC(C)(c1ccc(O)cc1)c2ccc(O)cc2'}
    selected=st.selectbox('Example molecule',list(presets))
    smiles=st.text_input('SMILES',value=presets[selected],key='input_'+selected)
    if st.button('Predict all 12 endpoints',type='primary'):
        try:
            from phase2.member5_demo_submission.predict import predict
            with st.spinner('Running saved models…'): result=predict(smiles)
            st.session_state['molecule_result']=result
        except (ValueError,FileNotFoundError,ImportError,RuntimeError) as exc:
            st.session_state.pop('molecule_result',None); st.error(str(exc))
    result=st.session_state.get('molecule_result')
    if result is not None:
        canonical=result.canonical_smiles.iloc[0]
        st.caption('Results for: '+canonical)
        from rdkit import Chem
        from rdkit.Chem import Draw
        a,b=st.columns([1,3])
        with a: st.image(Draw.MolToImage(Chem.MolFromSmiles(canonical),size=(350,250)))
        with b:
            scores=result.pivot(index='endpoint',columns='model',values='activity_score')
            st.dataframe(scores.style.format('{:.3f}'),use_container_width=True)
        st.bar_chart(scores)
        st.download_button('Download prediction CSV',result.to_csv(index=False),'tox21_predictions.csv','text/csv')
    st.caption('Default activity threshold: 0.5. Scores are uncalibrated assay predictions, not probabilities of human toxicity. An inactive prediction does not establish safety.')

else:
    st.write('Explore the full methodology, evidence, limitations and reproduction instructions.')
    for label,path,mime in [('Formal report — LaTeX source',DEMO/'report/main.tex','text/plain'),('Project presentation',DEMO/'presentation/phase2_presentation_professional.pptx','application/vnd.openxmlformats-officedocument.presentationml.presentation'),('Setup and reproduction guide',DEMO/'README.md','text/markdown')]:
        if path.exists(): st.download_button(label,path.read_bytes(),path.name,mime)
    st.markdown('**Reproduce locally**')
    st.code('python -m pip install -r requirements.txt\npython -m phase2.member5_demo_submission.prepare_models\nstreamlit run app.py',language='bash')
    st.caption('Model hashes and library versions are recorded with the inference bundle. No model training occurs during a prediction.')
