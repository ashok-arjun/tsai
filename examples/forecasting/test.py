from tsai.all import *
ts = get_forecasting_time_series("Weather").values

print(ts.shape)

X, y = SlidingWindow(60, horizon=3)(ts)

print(X.shape, y.shape)

splits = TimeSplitter(235)(y) 
batch_tfms = TSStandardize()
fcst = TSForecaster(X, y, splits=splits, path='models', batch_tfms=batch_tfms, bs=512, arch=TSTPlus, metrics=mae, cbs=ShowGraph())
fcst.fit_one_cycle(50, 1e-3)
fcst.export("fcst.pkl")

from tsai.inference import load_learner
fcst = load_learner("models/fcst.pkl", cpu=False)
raw_preds, target, preds = fcst.get_X_preds(X[splits[0]], y[splits[0]])
print(raw_preds.shape)