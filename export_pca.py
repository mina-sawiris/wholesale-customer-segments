from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

BASE = Path(__file__).parent
OUTLIERS = [65, 66, 75, 128, 154]

data = pd.read_csv(BASE / "original_analysis" / "customers.csv").drop(["Region", "Channel"], axis=1)
log_data = np.log(data)
good_data = log_data.drop(log_data.index[OUTLIERS]).reset_index(drop=True)

pca = PCA(n_components=2).fit(good_data)
joblib.dump(pca, BASE / "Model" / "pca_model.pkl")
print("Saved Model/pca_model.pkl | explained variance:", pca.explained_variance_ratio_.round(4))
