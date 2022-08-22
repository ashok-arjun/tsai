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
import itertools as it
import argparse
import os

METHOD_HYPERPARAM_MAP = {
        'rocket': {'model': MiniRocket, 
                'arch_args': {
                                'num_features': [10000],
                                'max_dilations_per_kernel': [16, 32, 64]      
                        }
                },
        'transformer': {'model': TransformerModel, 
                'arch_args': {
                                'd_model': [512],
                                'n_head': [1, 2, 4, 8, 16],
                                'n_layers': [1,2,3,4,5]      
                        }
                },
        'tstplus': {'model': TSTPlus, 
                'arch_args': {
                                'd_model': [512],
                                'n_head': [1, 2, 4, 8, 16],
                                'n_layers': [1,2,3,4,5],
                                'max_seq_len': [201]      
                        }
                },
        'rnn': {'model': RNN, 
                'arch_args': {
                                'hidden_size': [512],
                                'n_layers': [1,2,3,4,5],
                                'bidirectional': [True, False],
                                # 'rnn_dropout': [0, 0.3, 0.5],
                                # 'fc_dropout': [0, 0.3, 0.5]
                        }
                },
        'lstm': {'model': LSTM, 
                'arch_args': {
                                'hidden_size': [512],
                                'n_layers': [1,2,3,4,5],
                                'bidirectional': [True, False],
                                # 'rnn_dropout': [0, 0.3, 0.5],
                                # 'fc_dropout': [0, 0.3, 0.5]
                        }
                },
        'rnnplus': {'model': RNNPlus, 
                'arch_args': {
                                'hidden_size': [512],
                                'n_layers': [1,2,3,4,5],
                                'bidirectional': [True, False],
                                # 'rnn_dropout': [0, 0.3, 0.5],
                                # 'fc_dropout': [0, 0.3, 0.5]
                        }
                },
        'lstmplus': {'model': LSTMPlus, 
                'arch_args': {
                                'hidden_size': [512],
                                'n_layers': [1,2,3,4,5],
                                'bidirectional': [True, False],
                                # 'rnn_dropout': [0, 0.3, 0.5],
                                # 'fc_dropout': [0, 0.3, 0.5]
                        }
                },
        'rnn_fcn': {'model': RNN_FCN, 
                'arch_args': {
                                'hidden_size': [512],
                                'rnn_layers': [1,2,3,4,5],
                                'bidirectional': [True, False],
                                # 'rnn_dropout': [0, 0.3, 0.5],
                                # 'fc_dropout': [0, 0.3, 0.5],
                                'conv_layers': [[128, 256, 512]],
                                'kss': [[7, 5, 3]]
                        }
                },
        'rnn_fcnplus': {'model': RNN_FCNPlus, 
                'arch_args': {
                                'hidden_size': [512],
                                'rnn_layers': [1,2,3,4,5],
                                'bidirectional': [True, False],
                                # 'rnn_dropout': [0, 0.3, 0.5],
                                # 'fc_dropout': [0, 0.3, 0.5],
                                'conv_layers': [[128, 256, 512]],
                                'kss': [[7, 5, 3]]
                        }
                },
        'resnet': {'model': ResNet, 'arch_args': {}},
        'resnetplus': {'model': ResNetPlus, 'arch_args': {}},
        'inceptiontime': {'model': InceptionTime, 'arch_args': {}},
        'inceptiontimeplus': {'model': InceptionTimePlus, 'arch_args': {}},
        'xresnet1d18': {'model': xresnet1d18, 'arch_args': {}},
        'xresnet1d18_deep': {'model': xresnet1d18_deep, 'arch_args': {}},
        'xresnet1d34': {'model': xresnet1d34, 'arch_args': {}},
        'xresnet1d34_deep': {'model': xresnet1d34_deep, 'arch_args': {}},
        'xresnet1d18plus': {'model': xresnet1d18plus, 'arch_args': {}},
        'xresnet1d18_deepplus': {'model': xresnet1d18_deepplus, 'arch_args': {}},
        'xresnet1d34plus': {'model': xresnet1d34plus, 'arch_args': {}},
        'xresnet1d34_deepplus': {'model': xresnet1d34_deepplus, 'arch_args': {}}
}

if __name__ == "__main__":        

        parser = argparse.ArgumentParser()
        parser.add_argument('--method', choices=list(METHOD_HYPERPARAM_MAP.keys()))
        parser.add_argument('--data', type=str, default="data/beijing/PM2.5.csv")
        parser.add_argument('--path', type=str, default="models")
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_epochs', type=int, default=100)
        parser.add_argument('--run_name', type=str, default="")
        parser.add_argument('--tags', nargs='+', default=[])
        parser.add_argument('--early_stopping_patience', type=int, default=10)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--task', type=str, default='regression')
        parser.add_argument('--train_slice_start', type=float, default=0., help='device ids of multile gpus')
        parser.add_argument('--train_slice_end', type=float, default=0.6, help='device ids of multile gpus')
        parser.add_argument('--valid_slice_end', type=float, default=0.8, help='device ids of multile gpus')
        parser.add_argument('--lr', type=float, default=0.001, help='lr')

        args = parser.parse_args()
        METHOD = args.method
        NUM_EPOCHS=args.num_epochs
        WANDB_RUN_NAME=args.run_name
        TAGS=args.tags
        EARLY_STOPPING_PATIENCE = args.early_stopping_patience
        NUM_WORKERS = args.num_workers
        DATA=args.data
        PATH=args.path
        BATCH_SIZE=args.batch_size

        if args.task in ['regression', 'forecasting']:
                METRICS=[mse, mae]
        elif args.task == 'classification':
                METRICS=[accuracy, RocAuc]

        arch=METHOD_HYPERPARAM_MAP[METHOD]['model']
        k=METHOD_HYPERPARAM_MAP[METHOD]['arch_args']

        keys = list(k.keys())

        allNames = sorted(k)
        combinations = it.product(*(k[Name] for Name in allNames))
        allConfigDicts = [dict(zip(allNames, x)) for x in list(combinations)]

        print(allConfigDicts)

        if len(allConfigDicts) == 0:
                allConfigDicts = [{}]

        for configDict in allConfigDicts:
                print(configDict)

                run_name = WANDB_RUN_NAME + "_" + METHOD
                for k,v in configDict.items():
                        run_name = run_name + "_{}={}".format(k, v)

                set_seed(24)

                wandb.init(project="tsai", config=args, name=run_name, tags=TAGS)
                PATH = os.path.join(PATH, wandb.run.id)

                df = pd.read_csv(DATA)

                train_slice_start = args.train_slice_start
                train_slice_end = args.train_slice_end
                valid_slice_end = args.valid_slice_end

                # train_slice = slice(int(train_slice_start * len(df)), int(train_slice_end * len(df)))
                # valid_slice = slice(int(train_slice_end * len(df)), int(valid_slice_end * len(df)))
                # test_slice = slice(int(valid_slice_end * len(df)), None)

                # train_slice_pd = df.iloc[train_slice]
                # valid_slice_pd = df.iloc[valid_slice]
                # test_slice_pd = df.iloc[test_slice]

                # print("Start and end dates of splits:")
                # print("Train: Start: {} End: {}".format(train_slice_pd.index[0], train_slice_pd.index[-1]))
                # print("Valid: Start: {} End: {}".format(valid_slice_pd.index[0], valid_slice_pd.index[-1]))
                # print("Test: Start: {} End: {}".format(test_slice_pd.index[0], test_slice_pd.index[-1]))

                ts = df.values
                print("Shape of DF:", ts.shape)

                if args.task == 'regression':
                        horizon = 0
                elif args.task == 'forecasting':
                        horizon = 24
                X, y = SlidingWindow(1, get_x=list(range(15)), get_y=[15], horizon=horizon)(ts)
                train_split = list(range(int(train_slice_start*len(X)), int(train_slice_end*len(X))))
                valid_split = list(range(int(train_slice_end*len(X)), int(valid_slice_end*len(X))))
                test_split = list(range(int(valid_slice_end*len(X)), int(len(X))))

                train_valid_splits = (train_split, valid_split)
                all_splits = (train_split, valid_split, test_split)

                splits = train_valid_splits

                print("X:{}, y:{}, Split Sizes:{}".format(X.shape, y.shape, [len(split) for split in all_splits]))

                wandb_cb = WandbCallback()
                cbs = [wandb_cb] 

                batch_tfms = TSStandardize(by_var=True)

                saveModelCallback = SaveModelCallback(monitor='valid_loss')
                earlyStoppingCallback = EarlyStoppingCallback(monitor='valid_loss', patience=EARLY_STOPPING_PATIENCE)
                loss_func = MSELossFlat()

                if args.task == 'regression':
                        tfms  = [None, [TSRegression()]]
                elif args.task == 'forecasting':
                        tfms  = [None, [TSForecasting()]]

                dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
                dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[BATCH_SIZE, BATCH_SIZE], \
                                                num_workers=NUM_WORKERS, device='cuda:0', batch_tfms=batch_tfms)
                if arch == TSTPlus:
                        reg = TSRegressor(X, y, splits=splits, path=PATH, arch=TSTPlus, arch_config=configDict, batch_tfms=batch_tfms, metrics=METRICS,  cbs=cbs, verbose=True)
                else:
                        model = create_model(arch, dls=dls, **configDict, verbose=True, device='cuda:0')
                        reg = Learner(dls, model,  metrics=METRICS, cbs=cbs, loss_func=loss_func, path=PATH)
                num_params = total_params(reg.model)
                print("Number of parameters:", num_params)
                wandb.log({"num_params": num_params})
                # lr_max = reg.lr_find().valley
                # print("Found learning rate:", lr_max)  
                reg.fit_one_cycle(NUM_EPOCHS, args.lr, cbs=[saveModelCallback, earlyStoppingCallback])
                reg.export("reg.pkl")

                raw_preds, target, preds = reg.get_X_preds(X[splits[0]], y[splits[0]])
                train_mse = skm.mean_squared_error(target, preds, squared=True)
                train_mae = skm.mean_absolute_error(target, preds)

                raw_preds, target, preds = reg.get_X_preds(X[splits[1]], y[splits[1]])
                valid_mse = skm.mean_squared_error(target, preds, squared=True)
                valid_mae = skm.mean_absolute_error(target, preds)

                raw_preds, target, preds = reg.get_X_preds(X[all_splits[-1]], y[all_splits[-1]])
                test_mse = skm.mean_squared_error(target, preds, squared=True)
                test_mae = skm.mean_absolute_error(target, preds)

                print("Train MSE:{} | MAE: {}".format(train_mse, train_mae))
                print("Valid MSE:{} | MAE: {}".format(valid_mse, valid_mae))
                print("Test MSE:{} | MAE: {}".format(test_mse, test_mae))

                wandb.log({"final/train_mse": train_mse, "final/train_mae": train_mae})
                wandb.log({"final/valid_mse": valid_mse, "final/valid_mae": valid_mae})
                wandb.log({"final/test_mse": test_mse, "final/test_mae": test_mae})

                wandb.finish()