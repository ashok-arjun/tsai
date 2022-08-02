# from sklearn.metrics import mean_squared_error, make_scorer
# from tsai.all import *
# from tsai.models.MINIROCKET import *

# X_train, y_train, X_test, y_test = get_regression_data('BeijingPM25Quality')
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# X, y, splits = get_regression_data('BeijingPM25Quality', split_data=False)
# print(X.shape, y.shape)

# print(X[0])



# print(len(splits))

# weather = get_Monash_forecasting_data('cif_2016_dataset')
# print(weather.shape)

# # rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
# # mr_reg = MiniRocketRegressor(scoring=rmse_scorer)
# # mr_reg.fit(X_train, y_train)
# # mr_reg.save("minirocket_regressor")

# # mr_reg = load_rocket("minirocket_regressor")
# # y_pred = mr_reg.predict(X_test)
# # print("Mse:", mean_squared_error(y_test, y_pred, squared=False))


"""

# Source: https://github.com/timeseriesAI/tsai/issues/483

import seaborn as sns
from tsai.all import *

flight_data = sns.load_dataset("flights")
data = flight_data['passengers'].values
plt.figure(figsize=(10, 4))
plt.plot(data)
plt.show()

x, y = SlidingWindow(window_len=15)(data)
splits = TimeSplitter()(y)
tfms  = [ToFloat(), ToFloat()] # convert int to float
batch_tfms = TSStandardize()
dls = get_ts_dls(x, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=64)
learn = ts_learner(dls, InceptionTime, loss_func=MSELossFlat(), metrics=[mae, rmse], cbs=ShowGraph())
# learn.lr_find()
learn.fit_one_cycle(100, 1e-1)


preds, target, *_ = learn.get_X_preds(x, y)
valid_preds = preds.clone()
valid_preds[splits[0]] = np.nan
plt.figure(figsize=(10, 4))
plt.plot(target.flatten().numpy(), color='gray', label='target')
plt.plot(preds.flatten().numpy(), color='orange', label='train preds')
plt.plot(valid_preds.flatten().numpy(), color='purple', label='valid preds')
plt.legend()
plt.show()
"""

import seaborn as sns
from tsai.all import *

flight_data = sns.load_dataset("flights")
data = flight_data['passengers'].values
plt.figure(figsize=(10, 4))
plt.plot(data)
plt.show()

x, y = SlidingWindow(window_len=15)(data)

print(data.shape, x.shape, y.shape)
# splits = TimeSplitter()(y)
# tfms  = [ToFloat(), ToFloat()] # convert int to float
# batch_tfms = TSStandardize()
# dls = get_ts_dls(x, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=64)
# learn = ts_learner(dls, InceptionTime, loss_func=MSELossFlat(), metrics=[mae, rmse], cbs=ShowGraph())
# # learn.lr_find()
# learn.fit_one_cycle(100, 1e-1)


# preds, target, *_ = learn.get_X_preds(x, y)
# valid_preds = preds.clone()
# valid_preds[splits[0]] = np.nan
# plt.figure(figsize=(10, 4))
# plt.plot(target.flatten().numpy(), color='gray', label='target')
# plt.plot(preds.flatten().numpy(), color='orange', label='train preds')
# plt.plot(valid_preds.flatten().numpy(), color='purple', label='valid preds')
# plt.legend()
# plt.show()