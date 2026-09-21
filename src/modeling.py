import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression


def prepare_features(df, feature_columns, target_column):
    """
    Prepare explanatory variables and target variable
    for accident prediction.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    feature_columns : list
        Variables used to explain accident probability.

    target_column : str
        Binary target variable:
        1 = accident
        0 = no accident.

    Returns
    -------
    X : pandas.DataFrame
        Explanatory variables.

    y : pandas.Series
        Binary target variable.
    """

    data = df[
        feature_columns + [target_column]
    ].dropna()

    X = data[feature_columns]
    y = data[target_column]

    return X, y


def fit_glm_logistic(X, y):
    """
    Fit a Generalized Linear Model with a binomial family.
    """

    X_with_constant = sm.add_constant(X)

    model = sm.GLM(
        y,
        X_with_constant,
        family=sm.families.Binomial()
    )

    results = model.fit()

    return results


def predict_glm(model, X):
    """
    Estimate accident probabilities using a fitted GLM.
    """

    X_with_constant = sm.add_constant(
        X,
        has_constant="add"
    )

    return model.predict(X_with_constant)


def fit_logistic_regression(X, y):
    """
    Fit a logistic regression model using scikit-learn.
    """

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X, y)

    return model


def predict_probabilities(model, X):
    """
    Return the estimated probability of an accident.
    """

    return model.predict_proba(X)[:, 1]


if __name__ == "__main__":

    print(
        "Statistical modeling module for "
        "avalanche accident prediction."
    )
