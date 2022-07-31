from tsai.imports import *
from tsai.utils import *
from tsai.data.validation import combine_split_data

X_train, y_train, X_valid, y_valid = np.random.rand(1000,12,1), np.random.randint(1000,1,1), \
                                    np.random.rand(1000,12,1), np.random.randint(1000,1,1)
X, y, splits = combine_split_data([X_train, X_valid], [y_train, y_valid])
# test_eq(X_train, X[splits[0]])
# test_eq(X_valid, X[splits[1]])
# test_type(X_train, X)
# test_type(y_train, y)
