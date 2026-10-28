import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
FEATS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicatessen"]
raw = pd.read_csv(BASE / "original_analysis" / "customers.csv")
data = raw.drop(["Region", "Channel"], axis=1)

results = []


def check(ok, name):
    results.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name)


def new_app():
    return AppTest.from_file(str(BASE / "app.py"), default_timeout=120).run()


def predict(at):
    next(b for b in at.button if "Predict" in b.label).click().run()
    return [m.value for m in at.metric][-2:]


at = new_app()
check(not at.exception, "the app starts without errors")
check(len(at.tabs) == 4, "4 tabs are shown")
check(len(at.number_input) == 6, "6 sidebar inputs, one per product category")
metrics = [m.value for m in at.metric]
check("0.4223" in metrics, "silhouette score is 0.4223 (same as the notebook)")
check(metrics[7:9] == ["155", "280"], "segments hold 155 and 280 customers (same as the notebook)")
check(float(metrics[9].strip("%")) > 85, f"segments agree with the real Channel ({metrics[9]})")

pca = joblib.load(BASE / "Model" / "pca_model.pkl") if (BASE / "Model" / "pca_model.pkl").exists() else None
gmm_file = next(p for p in (BASE / "Model").iterdir() if p.name.lower() == "gmm_model.pkl")
gmm = joblib.load(gmm_file)
if pca is not None:
    for i, expected_channel in [(39, "Restaurant"), (85, "Retailer"), (97, "Restaurant"), (330, "Restaurant")]:
        for f in FEATS:
            at.session_state[f"in_{f}"] = int(data.loc[i, f])
        at.run()
        shown = predict(at)
        point = pd.DataFrame(pca.transform(np.log(data.loc[[i]].astype(float))), columns=["Dimension 1", "Dimension 2"])
        probs = gmm.predict_proba(point)[0]
        check(shown == [f"{probs[0]:.1%}", f"{probs[1]:.1%}"] and not at.exception,
              f"customer {i}: app shows the same probabilities as the notebook ({shown[0]} retailer)")
        want = 0 if expected_channel == "Retailer" else 1
        check(int(np.argmax(probs)) == want, f"customer {i} lands in the expected segment ({expected_channel})")

at = new_app()
next(b for b in at.button if "Restaurant" in b.label).click().run()
check(at.session_state["in_Fresh"] == 56159 and "100.0%" in predict(at)[1], "'Restaurant' example button works")
at = new_app()
next(b for b in at.button if "Retailer" in b.label).click().run()
check(predict(at)[0] == "99.8%", "'Retailer' example button works")

at = new_app()
for f in FEATS:
    at.session_state[f"in_{f}"] = 1
at.run()
predict(at)
check(not at.exception, "the smallest possible input (all 1) does not crash")

print(f"\n{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
