from tsai.all import *
import pandas as pd

# import pdb; pdb.set_trace()
# ts = get_forecasting_time_series("Weather").values

df = pd.read_csv("datasets/PM2.5/PM2.5_train_start_0/data.csv")

train_slice_start = 0.
train_slice_end = 0.6
valid_slice_end = 0.8

# Should we shuffle data before doing this?
train_slice = slice(int(train_slice_start * len(df)), int(train_slice_end * len(df)))
valid_slice = slice(int(train_slice_end * len(df)), int(valid_slice_end * len(df)))
test_slice = slice(int(valid_slice_end * len(df)), None)

train_slice_pd = df.iloc[train_slice]
valid_slice_pd = df.iloc[valid_slice]
test_slice_pd = df.iloc[test_slice]

print("Start and end dates of splits:")
print("Train: Start: {} End: {}".format(train_slice_pd.index[0], train_slice_pd.index[-1]))
print("Valid: Start: {} End: {}".format(valid_slice_pd.index[0], valid_slice_pd.index[-1]))
print("Test: Start: {} End: {}".format(test_slice_pd.index[0], test_slice_pd.index[-1]))

ts = train_slice_pd.values
print("Shape of time series:", )

X, y = SlidingWindow(2, get_x=list(range(15)), get_y=[15], horizon=24)(ts)

print(df.shape, ts.shape, X.shape, y.shape, type(X), type(y))

splits = TimeSplitter(235)(y) 
batch_tfms = TSStandardize()
fcst = TSForecaster(X, y, splits=splits, path='models', batch_tfms=batch_tfms, bs=512, arch=XCMPlus, metrics=mae, cbs=ShowGraph())
fcst.fit_one_cycle(50, 1e-3)
fcst.export("fcst.pkl")

# from tsai.inference import load_learner
# fcst = load_learner("models/fcst.pkl", cpu=False)
# raw_preds, target, preds = fcst.get_X_preds(X[splits[0]], y[splits[0]])
# print(raw_preds.shape)