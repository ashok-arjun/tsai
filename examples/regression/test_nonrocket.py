from tsai.all import *
from fastai.callback.wandb import *
import wandb
import sklearn.metrics as skm

my_setup(wandb)

X, y, splits = get_regression_data('AppliancesEnergy', split_data=False)

print(X.shape, y.shape)
print(splits)

print(X.shape, y.shape, [len(split) for split in splits])

wandb_cb = WandbCallback()
batch_tfms = TSStandardize(by_var=True)
reg = TSRegressor(X, y, splits=splits, path='models', arch=TSTPlus, batch_tfms=batch_tfms, metrics=rmse, \
                    verbose=True, device='cuda:0')
reg.fit_one_cycle(100, 3e-4)
reg.export("reg.pkl")

from tsai.inference import load_learner
reg = load_learner("models/reg.pkl")

raw_preds, target, preds = reg.get_X_preds(X[splits[0]], y[splits[0]])
train_rmse = skm.mean_squared_error(target, preds, squared=False)
train_mae = skm.mean_absolute_error(target, preds)

raw_preds, target, preds = reg.get_X_preds(X[splits[1]], y[splits[1]])
valid_rmse = skm.mean_squared_error(target, preds, squared=False)
valid_mae = skm.mean_absolute_error(target, preds)

print("Train RMSE:{} | MAE: {}".format(train_rmse, train_mae))
print("Valid RMSE:{} | MAE: {}".format(valid_rmse, valid_mae))

# YOU CAN MODIFY YOUR CONFIG AND/OR TRAINING SCRIPT IN THIS CELL AND RE-RUN MANUAL EXPERIMENTS THAT WILL BE TRACKED BY W&B

"""
config = AttrDict (
    batch_tfms = TSStandardize(by_sample=True),
    arch = TSiTPlus,
    arch_config = {},
    lr = 1e-3,
    n_epoch = 10,   
)

config = AttrDict (
    batch_tfms = TSStandardize(by_sample=True),
    arch = TSiTPlus,
    arch_config = {},
    lr = 1e-3,
    n_epoch = 10,   
)

# with wandb.init(project="LSST_v01", config=config, name='baseline_log_all'):
    # X, y, splits = get_UCR_data('LSST', split_data=False)
    # tfms = [None, TSClassification()]
    # cbs = [ShowGraph(), WandbCallback(log_preds=True, log_model=True, dataset_name='LSST')] 
learn = TSClassifier(X, y, splits=splits, tfms=tfms, batch_tfms=config.batch_tfms, arch=config.arch, 
                        arch_config=config.arch_config, metrics=accuracy, cbs=cbs, verbose=False, device='cuda:0')
    # learn.fit_one_cycle(config.n_epoch, config.lr)

    # learn.export("reg.pkl")

X, y, splits = get_UCR_data('LSST', split_data=False)

from tsai.inference import load_learner
reg = load_learner("models/reg.pkl")

print(X[splits[0]].shape, y[splits[0]].shape)
raw_preds, target, preds = reg.get_X_preds(X[splits[0]], y[splits[0]])
print(raw_preds.shape, targets.shape, preds.shape)

# train_rmse = skm.mean_squared_error(target, preds, squared=False)
# train_mae = skm.mean_absolute_error(target, preds)

print(X[splits[1]].shape, y[splits[1]].shape)
raw_preds, target, preds = reg.get_X_preds(X[splits[1]], y[splits[1]])
print(raw_preds.shape, targets.shape, preds.shape)

# valid_rmse = skm.mean_squared_error(target, preds, squared=False)
# valid_mae = skm.mean_absolute_error(target, preds)

# print("Train RMSE:{} | MAE: {}".format(train_rmse, train_mae))
# print("Valid RMSE:{} | MAE: {}".format(valid_rmse, valid_mae))

    # wandb.log({"final/train_rmse":train_rmse, "final/train_mae": train_mae})
    # wandb.log({"final/valid_rmse":valid_rmse, "final/valid_mae": valid_mae})

"""