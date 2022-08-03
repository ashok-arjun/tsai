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