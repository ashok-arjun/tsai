from sklearn.metrics import mean_squared_error, make_scorer
from tsai.all import *
from tsai.models.MINIROCKET import *

X_train, y_train, X_test, y_test = get_regression_data('AppliancesEnergy')

# rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
# mr_reg = MiniRocketRegressor(scoring=rmse_scorer)
# mr_reg.fit(X_train, y_train)
# mr_reg.save("minirocket_regressor")

mr_reg = load_rocket("minirocket_regressor")
y_pred = mr_reg.predict(X_test)
print("Mse:", mean_squared_error(y_test, y_pred, squared=False))