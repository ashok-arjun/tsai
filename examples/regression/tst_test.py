from tsai.all import *
X, y, splits = get_regression_data('AppliancesEnergy', split_data=False)
batch_tfms = TSStandardize(by_sample=True)
reg = TSRegressor(X, y, splits=splits, path='models', arch=TSTPlus, batch_tfms=batch_tfms, metrics=rmse, cbs=ShowGraph(), verbose=True)
# reg = TSRegressor(X, y, splits=splits, path=PATH, arch=TSTPlus, arch_config=configDict, batch_tfms=batch_tfms, metrics=METRICS,  cbs=cbs, verbose=True)

reg.fit_one_cycle(100, 3e-4)
reg.export("reg.pkl")