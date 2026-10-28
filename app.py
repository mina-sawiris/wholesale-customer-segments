from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ----------------------------------------------------------------------------
# 0. PAGE CONFIG + STYLE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wholesale Customer Segmentation",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
FEATURES = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicatessen"]
OUTLIER_INDICES = [65, 66, 75, 128, 154] 

SEGMENTS = {
    0: {
        "name": "Supermarket / Retailer",
        "emoji": "🛒",
        "color": "#1D9E75",
        "delivery": "3-day delivery schedule",
        "reason": (
            "This customer buys mostly non-perishable and bulk items (Grocery, "
            "Milk, Detergents), so a reduced schedule will not disrupt them."
        ),
    },
    1: {
        "name": "Restaurant / Café",
        "emoji": "🍽️",
        "color": "#D85A30",
        "delivery": "Keep the 5-day delivery schedule",
        "reason": (
            "This customer depends on Fresh food that spoils quickly, so they "
            "need frequent deliveries and would react badly to only 3 days."
        ),
    },
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .hero {
        background: linear-gradient(120deg, #0F6E56 0%, #185FA5 100%);
        padding: 1.6rem 2rem; border-radius: 14px; color: white; margin-bottom: 1.2rem;
      }
      .hero h1 {color: white; margin: 0; font-size: 2rem;}
      .hero p  {color: #E1F5EE; margin: .3rem 0 0 0; font-size: 1.02rem;}
      .seg-card {
        border-radius: 12px; padding: 1rem 1.2rem; border-left: 6px solid;
        background: rgba(128,128,128,.08); height: 100%;
      }
      .result-box {
        border-radius: 14px; padding: 1.4rem 1.6rem; color: white; margin-top: .6rem;
      }
      .result-box h2 {color: white; margin: 0 0 .3rem 0;}
      .result-box p  {color: white; margin: .2rem 0; font-size: 1.02rem;}
      div[data-testid="stMetric"] {
        background: rgba(128,128,128,.08); padding: .8rem 1rem; border-radius: 10px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_raw_data() -> pd.DataFrame:
    """Original dataset with Channel/Region dropped (same as the notebook)."""
    df = pd.read_csv(BASE_DIR / "original_analysis" / "customers.csv")
    return df.drop(["Region", "Channel"], axis=1)


# ----------------------------------------------------------------------------
# 2. PREPROCESSING           
# ----------------------------------------------------------------------------
def preprocess_input(values: dict) -> pd.DataFrame:
    """Raw spending dict for ONE customer -> log-scaled 1-row DataFrame."""
    row = pd.DataFrame([values], columns=FEATURES).astype(float)
    return np.log(row.clip(lower=1))  # clip avoids log(0)


def get_training_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Full dataset -> log-scaled data with outliers removed (= `good_data`)."""
    log_data = np.log(raw)
    return log_data.drop(log_data.index[OUTLIER_INDICES]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. MODEL LOADING            
# ----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    raw = load_raw_data()
    good_data = get_training_data(raw)

    pca_path = BASE_DIR / "Model" / "pca_model.pkl"
    if pca_path.exists():
        pca = joblib.load(pca_path)
    else:
        pca = PCA(n_components=2).fit(good_data)

    gmm_path = BASE_DIR / "Model" / "gmm_model.pkl"
    if not gmm_path.exists():
        gmm_path = BASE_DIR / "Model" / "GMM_model.pkl"  
    gmm = joblib.load(gmm_path)
    return pca, gmm


@st.cache_data
def compute_results():
    """Everything the tabs need, computed once."""
    raw = load_raw_data()
    good_data = get_training_data(raw)
    pca, gmm = load_models()

    reduced = pd.DataFrame(pca.transform(good_data), columns=["Dimension 1", "Dimension 2"])
    preds = gmm.predict(reduced)
    score = silhouette_score(reduced, preds)

    full_pca = PCA(n_components=6).fit(good_data)  # for the variance chart only
    true_centers = pd.DataFrame(
        np.round(np.exp(pca.inverse_transform(gmm.means_))),
        columns=FEATURES,
        index=[f"Segment {i}" for i in range(len(gmm.means_))],
    )
    return {
        "raw": raw,
        "good_data": good_data,
        "reduced": reduced,
        "preds": preds,
        "score": score,
        "evr_full": full_pca.explained_variance_ratio_,
        "centers_2d": gmm.means_,
        "true_centers": true_centers,
        "components": pca.components_,
    }


# ----------------------------------------------------------------------------
# 4. SIDEBAR  (global inputs, visible in every tab)
# ----------------------------------------------------------------------------
raw_df = load_raw_data()

with st.sidebar:
    st.markdown("## 📦 New Customer")
    st.caption(
        "Enter the customer's **estimated annual spending** (monetary units) in each "
        "category, then go to the **Business Tool** tab and press *Predict*."
    )
    user_inputs = {}
    for feat in FEATURES:
        user_inputs[feat] = st.number_input(
            feat.replace("_", " "),
            min_value=1,
            max_value=int(raw_df[feat].max() * 2),
            value=int(raw_df[feat].median()),
            step=100,
            help=f"Dataset range: {int(raw_df[feat].min()):,} – {int(raw_df[feat].max()):,}  "
            f"(median {int(raw_df[feat].median()):,})",
            key=f"in_{feat}",
        )
    st.divider()
    st.caption("Wholesale Customers · PCA + Gaussian Mixture Model")


# ----------------------------------------------------------------------------
# 5. HEADER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>📦 Wholesale Customer Segmentation</h1>
      <p>Which customers can move from <b>5-day</b> to <b>3-day</b> delivery without hurting
      the business? PCA + Gaussian Mixture Models find the answer.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

res = compute_results()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Data Overview",
        "🧭 Dimensionality Reduction (PCA)",
        "🧩 Customer Segmentation",
        "🎯 Business Tool (Prediction)",
    ]
)

# ----------------------------------------------------------------------------
# TAB 1 - DATA OVERVIEW
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("The raw data")
    st.write(
        "The dataset holds the **annual spending** of wholesale customers across six "
        "product categories. `Channel` and `Region` are excluded from the analysis."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(raw_df):,}")
    c2.metric("Product categories", len(FEATURES))
    c3.metric("Total spending", f"{int(raw_df.sum().sum()):,}")

    st.markdown("**Sample of the dataset**")
    st.dataframe(raw_df.head(10), width="stretch")
    with st.expander("Statistical summary"):
        st.dataframe(raw_df.describe().round(1), width="stretch")

    left, right = st.columns(2)
    with left:
        totals = raw_df.sum().reset_index()
        totals.columns = ["Category", "Total spending"]
        fig = px.bar(
            totals, x="Category", y="Total spending", color="Category",
            title="Total spending per category",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with right:
        feat_choice = st.selectbox("Distribution of…", FEATURES, key="hist_feat")
        fig = px.histogram(raw_df, x=feat_choice, nbins=40, title=f"{feat_choice} spending distribution")
        st.plotly_chart(fig, width="stretch")
    st.info(
        "💡 Every category is **heavily right-skewed** (most customers spend little, a few "
        "spend a lot). That is why we apply a **log transform** before PCA."
    )

# ----------------------------------------------------------------------------
# TAB 2 - PCA
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Principal Component Analysis")
    st.write(
        "PCA compresses the 6 spending features into new axes (**dimensions**) that capture "
        "as much variation as possible. We keep the first two for clustering."
    )
    evr = res["evr_full"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Dimension 1", f"{evr[0]:.2%}")
    m2.metric("Dimension 2", f"{evr[1]:.2%}")
    m3.metric("Dim 1 + Dim 2", f"{evr[:2].sum():.2%}")

    left, right = st.columns(2)
    with left:
        dims = [f"Dim {i+1}" for i in range(6)]
        fig = go.Figure()
        fig.add_bar(x=dims, y=evr, name="Explained variance", marker_color="#185FA5")
        fig.add_scatter(x=dims, y=np.cumsum(evr), name="Cumulative", mode="lines+markers",
                        line=dict(color="#D85A30"))
        fig.update_layout(title="Explained variance ratio", yaxis_tickformat=".0%",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
    with right:
        # Biplot: customers in PC space + arrows for the original features
        red = res["reduced"]
        comps = res["components"]
        scale = float(np.abs(red.values).max()) * 0.9
        fig = go.Figure()
        fig.add_scatter(x=red["Dimension 1"], y=red["Dimension 2"], mode="markers",
                        marker=dict(size=5, opacity=0.4, color="#888"), name="Customers")
        for i, feat in enumerate(FEATURES):
            x, y = comps[0, i] * scale, comps[1, i] * scale
            fig.add_annotation(x=x, y=y, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowcolor="#D85A30")
            fig.add_annotation(x=x * 1.08, y=y * 1.08, text=feat, showarrow=False,
                               font=dict(color="#D85A30", size=12))
        fig.update_layout(title="Biplot: customers + feature directions",
                          xaxis_title="Dimension 1", yaxis_title="Dimension 2", showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.info(
        "💡 **Grocery, Milk and Detergents_Paper** point in the same direction, meaning they "
        "are bought together (retail-style shopping). **Fresh, Frozen and Delicatessen** "
        "point in another direction (food-service shopping)."
    )
    with st.expander("Feature weights for each dimension"):
        st.dataframe(
            pd.DataFrame(comps, columns=FEATURES, index=["Dimension 1", "Dimension 2"]).round(3),
            width="stretch",
        )

# ----------------------------------------------------------------------------
# TAB 3 - CLUSTERING
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Customer segments (Gaussian Mixture Model)")
    st.write(
        "A Gaussian Mixture Model with **2 clusters** groups the customers in the PCA space. "
        "It uses *soft clustering*, so each customer gets a probability for each segment."
    )
    k1, k2, k3 = st.columns(3)
    k1.metric("Silhouette score", f"{res['score']:.4f}", help="Closer to 1 = better separated clusters")
    counts = np.bincount(res["preds"], minlength=2)
    k2.metric("Segment 0 customers", int(counts[0]))
    k3.metric("Segment 1 customers", int(counts[1]))

    red = res["reduced"].copy()
    red["Segment"] = [f"Segment 0 · {SEGMENTS[0]['name']}" if p == 0 else f"Segment 1 · {SEGMENTS[1]['name']}"
                      for p in res["preds"]]
    fig = px.scatter(
        red, x="Dimension 1", y="Dimension 2", color="Segment", opacity=0.65,
        color_discrete_sequence=[SEGMENTS[0]["color"], SEGMENTS[1]["color"]],
        category_orders={"Segment": sorted(red["Segment"].unique())},
        title="GMM clusters in PCA space",
    )
    ctr = res["centers_2d"]
    fig.add_scatter(x=ctr[:, 0], y=ctr[:, 1], mode="markers", name="Cluster centers",
                    marker=dict(symbol="star", size=18, color="gold", line=dict(width=1.5, color="black")))
    st.plotly_chart(fig, width="stretch")

    st.markdown("**What does the average customer of each segment buy?**")
    st.dataframe(res["true_centers"], width="stretch")

    a, b = st.columns(2)
    for col, i in zip((a, b), (0, 1)):
        s = SEGMENTS[i]
        col.markdown(
            f"""<div class="seg-card" style="border-color:{s['color']}">
            <h4>{s['emoji']} Segment {i} – {s['name']}</h4>
            <p>{s['reason']}</p><p><b>Delivery:</b> {s['delivery']}</p></div>""",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# TAB 4 - BUSINESS TOOL
# ----------------------------------------------------------------------------
def predict_segment(values: dict):
    """
    Raw spending dict -> (segment index, probabilities, point in PCA space).
    
    """
    pca, gmm = load_models()
    log_row = preprocess_input(values)
    point = pd.DataFrame(pca.transform(log_row), columns=["Dimension 1", "Dimension 2"])
    probs = gmm.predict_proba(point)[0]
    return int(np.argmax(probs)), probs, point.iloc[0]


with tab4:
    st.subheader("Should this customer get 3-day delivery?")
    st.write("Review the values entered in the **sidebar**, then press **Predict**.")

    st.dataframe(pd.DataFrame([user_inputs]), width="stretch", hide_index=True)

    if st.button("🔮 Predict", type="primary"):
        seg, probs, point = predict_segment(user_inputs)
        s = SEGMENTS[seg]

        st.markdown(
            f"""<div class="result-box" style="background:{s['color']}">
            <h2>{s['emoji']} Prediction: {s['name']}</h2>
            <p><b>Recommendation:</b> {s['delivery']}</p>
            <p>{s['reason']}</p></div>""",
            unsafe_allow_html=True,
        )

        p0, p1 = st.columns(2)
        p0.metric("Probability: Supermarket / Retailer", f"{probs[0]:.1%}")
        p1.metric("Probability: Restaurant / Café", f"{probs[1]:.1%}")
        if max(probs) < 0.75:
            st.warning("⚠️ This customer sits near the border between the two segments. "
                       "Consider testing the 3-day schedule before rolling it out.")

        left, right = st.columns(2)
        with left:
            red = res["reduced"].copy()
            red["Segment"] = [SEGMENTS[p]["name"] for p in res["preds"]]
            fig = px.scatter(
                red, x="Dimension 1", y="Dimension 2", color="Segment", opacity=0.35,
                color_discrete_map={SEGMENTS[0]["name"]: SEGMENTS[0]["color"],
                                    SEGMENTS[1]["name"]: SEGMENTS[1]["color"]},
                title="Where the new customer lands",
            )
            fig.add_scatter(x=[point["Dimension 1"]], y=[point["Dimension 2"]], mode="markers",
                            name="New customer",
                            marker=dict(symbol="star", size=20, color="gold",
                                        line=dict(width=2, color="black")))
            st.plotly_chart(fig, width="stretch")
        with right:
            comp = res["true_centers"].T.copy()
            comp.columns = [SEGMENTS[0]["name"], SEGMENTS[1]["name"]]
            comp["New customer"] = pd.Series(user_inputs)
            comp = comp.reset_index().rename(columns={"index": "Category"})
            fig = px.bar(comp.melt(id_vars="Category", var_name="Who", value_name="Spending"),
                         x="Category", y="Spending", color="Who", barmode="group",
                         title="New customer vs. average segment",
                         color_discrete_map={SEGMENTS[0]["name"]: SEGMENTS[0]["color"],
                                             SEGMENTS[1]["name"]: SEGMENTS[1]["color"],
                                             "New customer": "gold"})
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("👈 Adjust the spending values in the sidebar, then click **Predict**.")