import numpy as np
from src.preprocess import DataPipeline
from src.expandingwindow import ExpandingWindow
from src.linear_models import OLS
from src.neuralnet import NeuralNetModel

N = 25000

pipe = DataPipeline("data/datashare.csv")
pipe.load().impute_missing().normalize()
df = pipe.df.iloc[:N].reset_index(drop=True)
ind = pipe.indicator_matrix.iloc[:N].reset_index(drop=True)

train, val, test = next(ExpandingWindow(3, 1, 1).split(df))
X_tr, y_tr = ExpandingWindow.get_features_and_target(train, ind)
X_te, _ = ExpandingWindow.get_features_and_target(test, ind)

ols = OLS()
ols.fit(X_tr, y_tr.values)
ols_out = test[["DATE", "permno", "ret"]].copy()
ols_out["pred"] = ols.predict(X_te)

nn = NeuralNetModel()
nn.fit(X_tr, y_tr)
nn_out = test[["DATE", "permno", "ret"]].copy()
nn_out["pred"] = nn.predict(X_te)


def sharpe(frame):
    spreads = []
    for _, g in frame.groupby("DATE"):
        g = g.sort_values("pred")
        k = max(len(g) // 10, 1)
        spreads.append(g["ret"].iloc[-k:].mean() - g["ret"].iloc[:k].mean())
    r = np.array(spreads)
    return r.mean() / r.std() * np.sqrt(12)


print(f"OLS Sharpe: {sharpe(ols_out):.3f}")
print(f"NN  Sharpe: {sharpe(nn_out):.3f}")
