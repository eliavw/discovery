import numpy as np

from itertools import product
from scipy.special import expit
from sklearn.neighbors import KernelDensity
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class Palmboom():

    def preliminaries_score_samples(self, X_test, y_test, joint=False):
        # Some checks
        assert X_test.shape[0] == y_test.shape[0]
        if y_test.ndim == 1:
            y_test = np.atleast_2d(y_test).T

        n, m = y_test.shape
        assert m == self.n_outputs_, "m : {}, n_outputs_: {}".format(m, self.n_outputs_)

        if joint:
            out = np.zeros(n, dtype=np.float)
        else:
            out = np.zeros((n, m), dtype=np.float)
        return y_test, n, m, out

    def build_contents_leaf(self, X, y):
        """
        Useful datastructure for learning the kdes.
        """
        # Checks
        assert X.shape[0] == y.shape[0]
        if y.ndim == 1:
            y = np.atleast_2d(y).T

        # Actions
        n, m = y.shape

        contents_leaf = np.zeros((n, m + 1))
        contents_leaf[:, :m] = y[:, :]          # First m columns are ground truth contents
        contents_leaf[:, -1] = self.apply(X)    # Last column is leaf index
        return contents_leaf


class PalmboomClassifier(DecisionTreeClassifier, Palmboom):
    def __init__(self, **kwargs):

        DecisionTreeClassifier.__init__(self, **kwargs)

        return

    def score_samples(self, X_test, y_test):

        def create_idx_map(classlabels):
            return {idx: label for idx, label in enumerate(classlabels)}

        # Preliminaries
        y_test, n, m, out = self.preliminaries_score_samples(X_test, y_test)

        # Datastructures
        l_pred = [np.unique(y_test[:, c]) for c in range(m)]

        y_prob = self.predict_proba(X_test)

        if m ==1:
            y_prob = [y_prob]
            idx_map = [create_idx_map(self.classes_)]
        else:
            idx_map = [create_idx_map(classlabels)
                       for classlabels in self.classes_]

        assert m == len(y_prob) == len(idx_map) == len(l_pred)

        # Figure out what the proba was of a given pred.
        for c in range(m):
            for label in l_pred[c]:

                mask = y_test[:, c] == label

                l_column_idx = idx_map[c].get(label, np.nan)

                if np.isnan(l_column_idx):
                    out[mask, c] = np.nan
                else:
                    out[mask, c] = y_prob[c][mask, l_column_idx]

        if out.shape[1]==1:
            return out.ravel()
        else:
            return out


class PalmboomRegressor(DecisionTreeRegressor, Palmboom):

    def __init__(self, **kwargs):

        DecisionTreeRegressor.__init__(self, **kwargs)

        self.marginal_kdes_ = dict()

        return

    def fit(self, X, y, marginal_kdes=False, joint_kdes=False, **kwargs):
        DecisionTreeRegressor.fit(self, X, y, **kwargs)

        if marginal_kdes:
            self.fit_marginal_kdes(X, y, **kwargs)

        if joint_kdes:
            self.fit_joint_kdes(X, y, **kwargs)

        return

    def fit_joint_kdes(self, X, y, **kwargs):

        # Checks
        m = self.n_outputs_
        assert y.ndim == 1 or y.shape[1] == m

        # Actions
        contents_leaf = self.build_contents_leaf(X, y)

        def scotts_factor(a):

            n, m = a.shape

            h = n ** (-1.0 / (m + 4))
            return h

        leaves = np.unique(contents_leaf[:, -1])
        joint_kdes = {(leaf_idx, 'joint'): None
                      for leaf_idx in leaves}

        for leaf_idx in leaves:
            leaf_mask = contents_leaf[:, -1] == leaf_idx    # Last column is the leaf index
            leaf_data = contents_leaf[leaf_mask, :-1]       # All the other contents is y_true

            if m == 1:
                leaf_data = np.atleast_2d(leaf_data)

            # Dynamical bandwidth selection (= per leaf bandwidth)
            bandwidth = scotts_factor(leaf_data)

            print("Training Data Dimension: {}".format(leaf_data.shape))
            joint_kdes[(leaf_idx, 'joint')] = KernelDensity(bandwidth=bandwidth, **kwargs).fit(leaf_data)

        self.joint_kdes_ = joint_kdes
        return

    def fit_marginal_kdes(self, X, y, **kwargs):

        # Checks
        m = self.n_outputs_
        assert y.ndim == 1 or y.shape[1] == m

        # Actions
        contents_leaf = self.build_contents_leaf(X, y)

        def scotts_factor(a):
            n, m = a.shape

            assert m == 1 # In the marginal case, we want single attributes.

            h = n ** (-1.0 / (m + 4))
            return h

        leaves = np.unique(contents_leaf[:, -1])
        marginal_kdes = {(leaf_idx, targ_idx): None
                         for (leaf_idx, targ_idx) in product(leaves, range(m))}

        for leaf_idx in leaves:
            leaf_mask = contents_leaf[:, -1] == leaf_idx    # Last column is the leaf index
            leaf_data = contents_leaf[leaf_mask, :-1]       # All the other contents is y_true

            for targ_idx in range(m):
                # Dynamical bandwidth selection
                leaf_targ_data = leaf_data[:, [targ_idx]]
                bandwidth = scotts_factor(leaf_targ_data)
                marginal_kdes[(leaf_idx, targ_idx)] = KernelDensity(bandwidth=bandwidth, **kwargs).fit(leaf_targ_data)

        self.marginal_kdes_ = marginal_kdes
        return

    def score_samples(self, X_test, y_test, kind='joint'):
        if kind in {'joint'}:
            assert len(self.joint_kdes_) > 0, "self.joint_kdes is empty"
            return self.joint_score_samples(X_test, y_test)
        elif kind in {'marginal'}:
            assert len(self.marginal_kdes_) > 0, "self.marginal_kdes is empty"
            return self.marginal_score_samples(X_test, y_test)
        else:
            msg = """
            Did not recognize kind:     {}
            Marginal or joint scores can be computed.
            """.format(kind)
            raise ValueError(msg)

    def joint_score_samples(self, X_test, y_test):

        # Preliminaries
        y_test, n, m, out = self.preliminaries_score_samples(X_test, y_test, joint=True)

        leaf_idxs = self.apply(X_test)
        u_leaf_idxs = np.unique(leaf_idxs).astype(int)

        for leaf_idx in u_leaf_idxs:
            leaf_mask = leaf_idxs == leaf_idx
            leaf_data = y_test[leaf_mask, :]
            if m == 1:
                leaf_data = np.atleast_2d(leaf_data)

            print("Test Data Dimension: {}".format(leaf_data.shape))
            out[leaf_mask] = self.joint_kdes_[(leaf_idx, 'joint')].score_samples(leaf_data)

        out = np.exp(out)
        return out

    def marginal_score_samples(self, X_test, y_test):

        # Preliminaries
        y_test, n, m, out = self.preliminaries_score_samples(X_test, y_test)

        leaf_idxs = self.apply(X_test)
        u_leaf_idxs = np.unique(leaf_idxs).astype(int)

        for leaf_idx in u_leaf_idxs:
            leaf_mask = leaf_idxs == leaf_idx

            for targ_idx in range(m):
                leaf_data = np.atleast_2d(y_test[leaf_mask, targ_idx]).T
                out[leaf_mask, targ_idx] = self.marginal_kdes_[(leaf_idx, targ_idx)].score_samples(leaf_data)

        out = np.exp(out)
        if out.shape[1] == 1:
            return out.ravel()
        else:
            return out
