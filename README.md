# Wholesale Customer Segmentation

An interactive Streamlit dashboard that helps a wholesale distributor decide **which customers can move from 5-day to 3-day delivery** without hurting their business.

**Live demo:** _add your Streamlit link here_

## The business problem

A wholesale distributor in Lisbon delivers to its customers five days a week. To cut costs, it wants to deliver only three days a week to some of them. The question is: **who will accept that, and who will not?**

Using unsupervised machine learning (PCA and a Gaussian Mixture Model), the customers fall into two natural segments based on their annual spending:

| Segment | Who they are | What they buy | Recommendation |
|---------|--------------|---------------|----------------|
| 0 | Supermarkets / Retailers | Groceries, milk, detergents (non-perishable, bulk) | Move to a **3-day** delivery schedule |
| 1 | Restaurants / Cafés | Mostly fresh food | **Keep 5-day** delivery |

## What the app does

The dashboard has a sidebar for entering a new customer's annual spending in six product categories (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicatessen) and four tabs:

1. **Data Overview**: the raw dataset and the spending distribution per category.
2. **Dimensionality Reduction (PCA)**: explained variance and a biplot showing how products group together.
3. **Customer Segmentation**: the Gaussian Mixture Model clusters and the silhouette score.
4. **Business Tool (Prediction)**: enter a customer's spending, click **Predict**, and get the segment with a delivery recommendation.

## Results

- Silhouette score: **0.4223**
- The two segments contain **155** and **280** customers.
- The segments agree with the real `Channel` label (Hotel/Restaurant/Cafe vs. Retail) for **more than 85%** of customers, even though the model never used that label.

## How it works

1. **Preprocessing** (`preprocess.py`): spending values are log-transformed, the same transformation used in the original analysis. Five outlier customers are removed from the training data.
2. **PCA**: reduces the six spending features to two dimensions.
3. **Gaussian Mixture Model**: clusters customers into two segments and returns the probability of belonging to each.
4. **Prediction**: a new customer goes through the same log transform, PCA, and GMM, and the app shows the segment probabilities with a delivery recommendation.

## Project structure

```
wholesale-customer-segments/
├── app.py                # Streamlit app (sidebar + 4 tabs)
├── preprocess.py         # Preprocessing for new customers
├── export_pca.py         # Re-creates Model/pca_model.pkl
├── test_part.py          # Automated checks for the app
├── requirements.txt
├── Model/                # Trained PCA and GMM models
└── original_analysis/    # Original analysis and customers.csv dataset
```

## Run locally

```bash
git clone https://github.com/mina-sawiris/wholesale-customer-segments.git
cd wholesale-customer-segments

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

### Run the tests

```bash
python test_part.py
```

The script checks that the app starts, that the silhouette score and segment sizes match the notebook, and that predictions for sample customers match the trained models.

## Deployment

The app is deployed on **Streamlit Community Cloud**:

- Repository: this repository
- Branch: `main`
- Main file path: `app.py`

## Tech stack

Python, Streamlit, scikit-learn, pandas, NumPy, Plotly, joblib

## Data

[Wholesale customers dataset](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers) from the UCI Machine Learning Repository: 440 customers of a wholesale distributor in Portugal. Values are annual spending in monetary units. The project is based on the "Creating Customer Segments" project from the Udacity Machine Learning curriculum.

## Team

| Role | Member |
|------|--------|
| Data Pipeline Engineer | Mina |
| Machine Learning Engineer | Heba |
| Frontend & UI (Streamlit) | Shahd |
| Visualization & Integration | Nemat |
| DevOps & Documentation | Youssef |
