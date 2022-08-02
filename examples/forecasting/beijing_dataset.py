from tsai.all import *
from tsai.inference import load_learner
from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *

from fastai.callback.wandb import *
from fastai.callback.tracker import SaveModelCallback, EarlyStoppingCallback

import wandb
import sklearn.metrics as skm
import pandas as pd
from utils import total_params, set_seed

config = {}

DATA="data/beijing/PM2.5.csv"
PATH="models"
BATCH_SIZE=32 # Try other batch sizes
METRICS=[mse, mae]
NUM_EPOCHS=100
WANDB_RUN_NAME='trials_newcode'
TAGS=['tests']
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 4
METHOD='rocket'

# TODO: Add seq_len to those relevant models
# TODO: Add TST with hyperparams
# TODO: Add dropout as a hyperparam
METHOD_HYPERPARAM_MAP = {
    'rocket': {'model': MiniRocket, 
            'arch_args': {
                            'num_features': [10000, 20000, 50000, 100000],
                            'max_dilations_per_kernel': [16, 32, 64, 128]      
                        }
            },
    'transformer': {'model': TransformerModel, 
            'arch_args': {
                            'd_model': [64, 128, 256, 512],
                            'n_head': [1, 2, 4, 8, 16],
                            'n_layers': [1,2,3,4,5]      
                        }
            },
    'rnn': {'model': RNN, 
            'arch_args': {
                            'hidden_size': [64, 128, 256, 512],
                            'n_layers': [1,2,3,4,5],
                            'bidirectional': [True, False],
                            'rnn_dropout': [0, 0.3, 0.5],
                            'fc_dropout': [0, 0.3, 0.5]
                        }
            },
    'lstm': {'model': LSTM, 
            'arch_args': {
                            'hidden_size': [64, 128, 256, 512],
                            'n_layers': [1,2,3,4,5],
                            'bidirectional': [True, False],
                            'rnn_dropout': [0, 0.3, 0.5],
                            'fc_dropout': [0, 0.3, 0.5]
                        }
            },
    'rnn_fcn': {'model': RNN_FCN, 
            'arch_args': {
                            'hidden_size': [64, 128, 256, 512],
                            'rnn_layers': [1,2,3,4,5],
                            'bidirectional': [True, False],
                            'rnn_dropout': [0, 0.3, 0.5],
                            'fc_dropout': [0, 0.3, 0.5],
                            'conv_layers': [[128, 256, 128], [128, 256, 512]],
                            'kss': [[7, 5, 3]]
                        }
            },
    # 'rnnplus': {'model': RNNPlus, 
    #         'arch_args': {
    #                         'hidden_size': [64, 128, 256, 512],
    #                         'n_layers': [1,2,3,4,5],
    #                         'bidirectional': [True, False],
    #                         'rnn_dropout': [0, 0.3, 0.5],
    #                         'fc_dropout': [0, 0.3, 0.5]
    #                     }
    #         },
    # 'lstmplus': {'model': LSTMPlus, 
    #         'arch_args': {
    #                         'hidden_size': [64, 128, 256, 512],
    #                         'n_layers': [1,2,3,4,5],
    #                         'bidirectional': [True, False],
    #                         'rnn_dropout': [0, 0.3, 0.5],
    #                         'fc_dropout': [0, 0.3, 0.5]
    #                     }
    #         }
}

set_seed(24)

wandb.init(project="tsai", config=config, name=WANDB_RUN_NAME, tags=TAGS)

df = pd.read_csv(DATA)

train_slice_start = 0.
train_slice_end = 0.6
valid_slice_end = 0.8

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

ts = df.values
print("Shape of DF:", ts.shape)

X, y = SlidingWindow(24, get_x=list(range(12)), get_y=[12], horizon=0)(ts)
train_split = list(range(int(train_slice_start*len(df)), int(train_slice_end*len(df))))
valid_split = list(range(int(train_slice_end*len(df)), int(valid_slice_end*len(df))))
test_split = list(range(int(valid_slice_end*len(df)), int(len(df))))

train_valid_splits = (train_split, valid_split)
all_splits = (train_split, valid_split, test_split)

splits = train_valid_splits

print("X:{}, y:{}, Split Sizes:{}".format(X.shape, y.shape, [len(split) for split in splits]))

wandb_cb = WandbCallback()
cbs = [wandb_cb] 

batch_tfms = TSStandardize(by_var=True)

saveModelCallback = SaveModelCallback(monitor='valid_loss')
earlyStoppingCallback = EarlyStoppingCallback(monitor='valid_loss', patience=EARLY_STOPPING_PATIENCE)
loss_func = MSELossFlat()

tfms  = [None, [TSRegression()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[BATCH_SIZE, BATCH_SIZE], \
                                num_workers=NUM_WORKERS, device='cuda:0', batch_tfms=batch_tfms)
arch=METHOD_HYPERPARAM_MAP[METHOD]['model']
k=METHOD_HYPERPARAM_MAP[METHOD]['arch_args']
model = create_model(arch, dls=dls, **k, verbose=True, device='cuda:0')
print(model.__class__.__name__)
reg = Learner(dls, model,  metrics=METRICS, cbs=cbs, loss_func=loss_func)
lr_max = reg.lr_find().valley
print("Found learning rate:", lr_max)  
reg.fit_one_cycle(NUM_EPOCHS, lr_max, cbs=[saveModelCallback, earlyStoppingCallback])
reg.export("reg.pkl")

raw_preds, target, preds = reg.get_X_preds(X[splits[0]], y[splits[0]])
train_mse = skm.mean_squared_error(target, preds, squared=True)
train_mae = skm.mean_absolute_error(target, preds)

raw_preds, target, preds = reg.get_X_preds(X[splits[1]], y[splits[1]])
valid_mse = skm.mean_squared_error(target, preds, squared=True)
valid_mae = skm.mean_absolute_error(target, preds)

print("Train MSE:{} | MAE: {}".format(train_mse, train_mae))
print("Valid MSE:{} | MAE: {}".format(valid_mse, valid_mae))

wandb.log({"final/train_mse": train_mse, "final/train_mae": train_mae})
wandb.log({"final/valid_mse": valid_mse, "final/valid_mae": valid_mae})

wandb.finish()

"""
reg = TSRegressor(X, y, splits=splits, path=PATH, arch=TSTPlus, batch_tfms=batch_tfms, \
                    bs=BATCH_SIZE, metrics=METRICS, \
                    verbose=True, device='cuda:0', cbs=cbs, num_workers=NUM_WORKERS, loss_func=loss_func)
lr_max = reg.lr_find().valley
print("Found learning rate:", lr_max)  
reg.fit_one_cycle(NUM_EPOCHS, lr_max, cbs=[saveModelCallback, earlyStoppingCallback])
reg.export("reg.pkl")
"""

"""
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[bs, bs*2])
arch=InceptionTime
k={}
model = create_model(arch, dls=dls, **k)
print(model.__class__.__name__)
learn = Learner(dls, model,  metrics=accuracy, cbs=[])
start = time.time()
learn.fit_one_cycle(100, 1e-3)
"""

"""
tfms  = [None, [TSRegression()]]
batch_tfms = TSStandardize(by_sample=True, by_var=True)
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=128)
dls.one_batch()

learn = ts_learner(dls, InceptionTime, metrics=[mae, rmse], cbs=ShowGraph())
learn.lr_find()

PATH = Path('./models/Regression.pkl')
learn = load_learner(PATH, cpu=False)

probas, _, preds = learn.get_X_preds(X[splits[1]])
skm.mean_squared_error(y[splits[1]], preds, squared=False)
"""

"""
X, y, splits = get_UCR_data('LSST', split_data=False)
tfms = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, path='/data/')
learn = ts_learner(dls, InceptionTimePlus, metrics=accuracy, cbs=[ShowGraph()])
learn.fit_one_cycle(10, 1e-2)
"""



"""
# ROCKET - SKLEARN type

# Multivariate regression ensemble with sklearn-type API
from sklearn.metrics import mean_squared_error, make_scorer
dsid = 'AppliancesEnergy'
X_train, y_train, X_valid, y_valid = get_Monash_regression_data(dsid)
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
model = MiniRocketVotingRegressor(n_estimators=5, scoring=rmse_scorer)
timer.start(False)
model.fit(X_train, y_train)
t = timer.stop()
y_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f'valid rmse        : {rmse:.5f} time: {t}')


from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *
# Create the MiniRocket features and store them in memory.
dsid = 'LSST'
X, y, splits = get_UCR_data(dsid, split_data=False)

tfms = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
model = build_ts_model(MiniRocket, dls=dls)
learn = Learner(dls, model, metrics=accuracy, cbs=ShowGraph())
learn.lr_find()

tfms = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
model = build_ts_model(MiniRocket, dls=dls)
learn = Learner(dls, model, metrics=accuracy, cbs=ShowGraph())
timer.start()
learn.fit_one_cycle(10, 3e-4)
timer.stop()

"""