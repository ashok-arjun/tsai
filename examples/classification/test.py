from tsai.all import *
# X, y, splits = get_classification_data('ArticularyWordRecognition', split_data=False)
# batch_tfms = TSStandardize(by_sample=True)
# mv_clf = TSClassifier(X, y, splits=splits, path='models', arch=InceptionTimePlus, batch_tfms=batch_tfms, metrics=accuracy)
# mv_clf.fit_one_cycle(10, 1e-2)
# mv_clf.export("mv_clf.pkl")

# from tsai.inference import load_learner
# mv_clf = load_learner("models/mv_clf.pkl")
# probas, target, preds = mv_clf.get_X_preds(X[splits[0]], y[splits[0]])

# """
# X, y, splits = get_UCR_data('FaceDetection', return_split=False)
# tfms  = [None, [Categorize()]]
# dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
# dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=64, batch_tfms=TSStandardize(by_var=True))
# model = TST(dls.vars, dls.c, dls.len, dropout=0.3, fc_dropout=0.9)
# learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropyFlat(), 
#                 metrics=[RocAucBinary(), accuracy],  cbs=ShowGraphCallback2())
# learn.fit_one_cycle(100, 1e-4)
# """

X, y, splits = get_UCR_data('FaceDetection', return_split=False)
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=64, batch_tfms=TSStandardize(by_var=True))
model = build_ts_model(InceptionTimePlus, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
learn.fit_one_cycle(100, 1e-4)