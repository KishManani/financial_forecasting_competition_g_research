import pandas as pd
import numpy as np
from collections import OrderedDict


def load_data():
    df_train = pd.read_csv('../data/train.csv.zip', index_col=0)
    df_test = pd.read_csv('../data/test.csv.zip', index_col=0)
    return (df_train, df_test)


def wMSE(preds, y, weights):
    """ Weighted mean square error. 
    
    Competition organisers have added a 1/N normalising factor into the weights already. 
    Still divide by number of samples so the score is invariant to the size of the sample 
    being scored.
    """    
    return (np.sum(np.square((preds - y)) * weights))/len(preds)


def train_model(model, X, y, w=None):
    """
    Train model with error handling of sample weights.
    """
    if isinstance(w, (pd.DataFrame, pd.Series)):
        w = w.values

    print('Fitting: {}'.format(model))
    try:
        model.fit(X, y, sample_weight=w)
    except TypeError:
        print('This model does not accept sample weights')
        model.fit(X, y)
    return None


def test_model(model, X, y, w=None):
    """
    Evaluate model using weighted mean square error. Returns predictions and error.
    """
    preds = model.predict(X)
    error = wMSE(preds=preds, y=y, weights=w)
    return preds, error


def train_and_test_models(models, X_train, y_train, X_test, y_test, weights_train=None, weights_test=None):
    """
    Trains a collection of models and evaluates each model on a test set. Returns predictions and errors.
    """
    df_preds_train = pd.DataFrame()
    df_preds_test = pd.DataFrame()
    train_errors = OrderedDict()
    test_errors = OrderedDict()

    for model_name, model in models.items():
        # Training
        train_model(model, X_train, y_train, w=weights_train)

        # Test
        preds_train, train_error = test_model(model, X_train, y_train, weights_train)
        preds_test, test_error = test_model(model, X_test, y_test, weights_test)
        scale = 1
        print('Train error: {} Test error: {} \n'.format(train_error * scale, test_error * scale))

        # Append test and train predictions to a dataframe
        data = {model_name + '_preds_train': preds_train}
        df = pd.DataFrame(data=data, index=X_train.index)
        df_preds_train = pd.concat([df_preds_train, df], axis=1)

        data = {model_name + '_preds_test': preds_test}
        df = pd.DataFrame(data=data, index=X_test.index)
        df_preds_test = pd.concat([df_preds_test, df], axis=1)

        # Attach weights
        if weights_train is not None:
            df_preds_train = df_preds_train.merge(right=weights_train.to_frame(), left_index=True, right_index=True)

        if weights_test is not None:
            df_preds_test = df_preds_test.merge(right=weights_test.to_frame(), left_index=True, right_index=True)

        # Save errors
        train_errors[model_name] = train_error
        test_errors[model_name] = test_error

    return df_preds_train, df_preds_test, train_errors, test_errors
