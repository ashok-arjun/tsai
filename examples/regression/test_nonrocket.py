from tsai.all import *
X, y, splits = get_regression_data('AppliancesEnergy', split_data=False)
batch_tfms = TSStandardize(by_sample=True)
reg = TSRegressor(X, y, splits=splits, path='models', arch=TSTPlus, batch_tfms=batch_tfms, metrics=rmse, cbs=ShowGraph(), verbose=True)
reg.fit_one_cycle(100, 3e-4)
reg.export("reg.pkl")

from tsai.inference import load_learner
reg = load_learner("models/reg.pkl")
raw_preds, target, preds = reg.get_X_preds(X[splits[0]], y[splits[0]])