import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


class MeanEncoder():
    """
    Applies mean encoding (a.k.a. target encoding).
    """

    def __init__(self):
        self.replacement_values = dict()
        self.most_common_value = dict()

    def fit(self, df, features, target):
        df = df.copy()
        df = pd.concat([df, target], axis=1)
        for f in features:
            encoding = df.groupby(f)[target.name].mean().to_dict()
            self.replacement_values[f] = encoding
            self.most_common_value[f] = encoding[df.loc[:, f].value_counts().index[0]]

        return None

    def transform(self, df):
        df = df.copy()
        for f in self.replacement_values.keys():
            encoding = self.replacement_values[f]

            most_common_value = df.loc[:, f].value_counts().index[0]
            unique_values = set(df.loc[:, f].unique())
            unique_values_training = set(encoding.keys())
            diff = unique_values.symmetric_difference(unique_values_training)
            if diff:
                print('Detected unseen values for encoding for feature {}: {}'.format(f, diff))
                for new_value in diff:
                    encoding[new_value] = self.most_common_value[f]

            df.loc[:, f + '_mean_encoded'] = df.loc[:, f].map(encoding)
        return df

    
class TreeBinner():
    """
    Uses decision trees to bin continuous numerical variables into discrete bins, also known as, discretisation.
    """

    def __init__(self):
        self.trees = dict()
        self._tree_depths = dict()

    def fit(self, df, target, weights, features=None, max_depth='auto'):
        if features is None:
            features = df.columns

        for feat in features:
            if max_depth == 'auto':
                max_tree_depth = self._select_optimum_depth(df.loc[:, feat].values.reshape(-1, 1), target)
                self._tree_depths[feat] = max_tree_depth
            else:
                max_tree_depth = max_depth
                self._tree_depths[feat] = max_tree_depth

            tree = DecisionTreeRegressor(max_depth=max_tree_depth)
            tree.fit(df.loc[:, feat].values.reshape(-1, 1), target, sample_weight=weights.values)
            self.trees[feat] = tree
        return None

    def transform(self, df):
        df = df.copy()
        for feat in self.trees.keys():
            tree = self.trees[feat]
            df[feat + '_binned'] = tree.predict(df.loc[:, feat].values.reshape(-1, 1))
        return df

    @staticmethod
    def _select_optimum_depth(X, y):
        """
        Selects optimum tree depth using 5-fold cross validation.
        """
        score_ls = []
        score_std_ls = []

        for tree_depth in range(1, 5):
            tree_model = DecisionTreeRegressor(max_depth=tree_depth)
            scores = cross_val_score(tree_model, X, y, cv=5, scoring='neg_mean_squared_error')
            score_ls.append(np.mean(scores))
            score_std_ls.append(np.std(scores))

        temp = pd.concat([pd.Series([1, 2, 3, 4]), pd.Series(score_ls), pd.Series(score_std_ls)], axis=1)
        temp.columns = ['depth', 'mse_mean', 'mse_std']
        return temp.depth[temp.mse_mean.idxmax()]

    
class LogTransformer():
    """
    Compute the log transform of a feature.

    Handles log(0) as a special value, replaces -inf to a constant which is far from the distribution but still within
    the distribution. Creates a new feature, *_log10_is_inf, to flag when log(0) occurred.
    """

    def __init__(self):
        self.replacement_values = dict()

    def fit(self, df, features):
        df = df.copy()
        for f in features:
            df[f + '_log10'] = np.log10(df[f])
            mask_not_inf = df[f + '_log10'] != -np.inf
            # Compute mean and std to replace inf values far from
            # distribution but still within distribution
            mean = df.loc[mask_not_inf, f + '_log10'].mean()
            std = df.loc[mask_not_inf, f + '_log10'].std()
            self.replacement_values[f] = mean + 5 * std
        return None

    def transform(self, df):
        df = df.copy()
        for f in self.replacement_values.keys():
            df[f + '_log10'] = np.log10(df[f])
            df[f + '_log10_is_inf'] = (df[f + '_log10'] == -np.inf).astype(np.int16)
            df[f + '_log10'].replace([-np.inf], self.replacement_values[f], inplace=True)
        return df


class Imputer():
    """
    Imputes missing value in feature the by mean. Creates a new feature, *_is_null, to flag when feature was null.
    """

    def __init__(self):
        self.features = []
        self.replacement_values = dict()

    def fit(self, df, features=None):
        if features is None:
            self.features = list(df.columns)
        else:
            self.features = features

        for f in self.features:
            if df.loc[:, f].isnull().any():
                replacement_value = df.loc[:, f].mean()
                self.replacement_values[f] = replacement_value
        return None

    def transform(self, df, features=None):
        df = df.copy()

        if features is None:
            features = self.replacement_values.keys()

        for f in features:
            df_NA = df.loc[:, f].isnull().astype(np.int16)
            df_NA.name = f + '_is_null'
            df.loc[:, f].fillna(self.replacement_values[f], inplace=True)
            df = pd.concat([df, df_NA], axis=1)

        return df
    

def compute_combined_variable(df, var1, var2):
    """Compute new categorical variable by concatenating two categorical variables
    """
    combined_var_name = var1 + '_' + var2
    df[combined_var_name] = df.loc[:, var1].astype('str') + '_' + df.loc[:, var2].astype('str')
    return df


def transform_to_embedding_vec(X, embedding_matrix):
    """ Convert a categorical feature vector to a dataframe of embedded vectors
    """
    embedding_dim = embedding_matrix.shape[1]
    X_transformed = np.zeros([ X.shape[0], embedding_dim])
    
    for ix, category in enumerate(X):
        X_transformed[ix, :] = embedding_matrix[category, :]
        
    col_names = [X.name + '_embedding_{0}'.format(i) for i in range(embedding_dim)]
        
    return pd.DataFrame(data=X_transformed, index=X.index, columns=col_names)