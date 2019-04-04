import json

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (MaxAbsScaler,
                                   MinMaxScaler,
                                   StandardScaler,
                                   RobustScaler)
from timeit import default_timer

from ..algo.induction import base_ind_algo
from ..algo.prediction import (
    mi_pred_algo,
    ma_pred_algo,
    mafi_pred_algo,
    it_pred_algo,
    rw_pred_algo,
    full_prune_strat,
)
from ..algo.selection import *
from ..io.io import save_output_data
from ..models.PolyModel import *
from ..settings import *
from ..utils.encoding import codes_to_query, code_to_query
from ..utils.keywords import *
from ..utils.metadata import (
    get_metadata_df,
    extract_nominal_numeric_attributes,
    only_nominal_targ,
    only_numeric_targ,
)

from ..utils.debug import debug_print

VERBOSITY = 0


class MERCS(object):
    """
    A MERCS model

        Multi-directional
        Ensembles of
        Regression and
        Classification
    treeS
    """

    # Main Methods
    def __init__(self, settings_fname=None, **kwargs):
        """
        Return an initialized MERCS object.

        Parameters
        ----------
        settings_fname: string
            Filename of .json file containing the settings
        """

        if settings_fname is None:
            self.s = create_settings()
        else:
            # TODO(elia): Merge what I load with default dict
            self.load_settings(settings_fname)

        self.m_codes = None
        self.m_list = None
        self.q_models = None
        self.imputer = None
        self.scores_scaler_ = None

        self.update_settings(mode="init", **kwargs)

        return

    def fit(self, X, **kwargs):
        """
        Fit the MERCS model to a dataset X.

        There are several steps to this, i.e.;
            1. Preliminaries
                1. Collecting metadata on the dataset
                2. Updating MERCS' settings dict
                3. Fitting the imputator
            2. Selection
                In this step, we decide the {descriptive,target} attributes
                of the component models. This results in a np.ndarray of
                shape (nb_models, nb_attributes) in which each row encodes
                the role of each attribute for a specific model.
            3. Induction
                Once it is determined which models compose the MERCS model,
                we train the individual components.
            4.  Post-processing
                1. Update metadata
                    Feature importances and classlabels are extracted.
                2. Update model_data
                    The induction time is saved in the model_data


        Parameters
        ----------
        X: pd.DataFrame, shape (nb_samples, nb_attributes)
            DataFrame of our dataset
        kwargs: dict
            Keyword arguments that can modify specific settings

        Returns
        -------

        """
        if isinstance(X,np.ndarray):
            X = pd.DataFrame(X)

        # 1. Prelims
        tick = default_timer()
        self.s["metadata"] = get_metadata_df(X)

        msg = """
        metadata of our model is: {}
        """.format(
            self.s["metadata"]
        )
        debug_print(msg, V=VERBOSITY)

        self.update_settings(mode="fit", **kwargs)
        self.fit_imputer(X)

        # 2. Selection = Prepare Induction
        self.m_codes = self.perform_selection(self.s["metadata"])

        # 3. Induction
        self.m_codes, self.m_list = self.perform_induction(
            X, self.m_codes, self.s["induction"], self.s["metadata"]
        )
        # 4. Post processing
        tock = default_timer()
        self.update_settings(mode="metadata")  # Save info on learned models
        self.update_settings(mode="model_data", mod_ind_time=tock - tick)

        return

    def fit_scores_scaler(self, X, scaler_kind=None, **kwargs):

        if scaler_kind in {None, 'StandardScaler'}:
            scaler = StandardScaler()
        elif scaler_kind in {'MaxAbsScaler'}:
            scaler = MaxAbsScaler()
        elif scaler_kind in {'MinMaxScaler'}:
            scaler = MinMaxScaler()
        elif scaler_kind in {'RobustScaler', 'Robust'}:
            scaler = RobustScaler()
        else:
            msg = """
            Unknown `scaler_kind`:      {}
            Current options are:        'StandardScaler'
                                        'MaxAbsScaler',
                                        'MinMaxScaler',
                                        'RobustScaler'}
            """.format(scaler_kind)
            raise ValueError(msg)

        scores = self.score_samples_(X, **kwargs)
        scaler.fit(scores)

        self.scores_scaler_ = scaler
        return

    def predict(self, X, q_idx=0, **kwargs):
        """
        Predict Y from X.

        Which attributes to predict are specified by queries.

        :param q_idx:       Optional keyword identifying which of the already
                            loaded queries you want to predict.
        :param X:           Test dataset. (Pandas DataFrame)
        :param kwargs:      Optional keyword arguments.
                            Used in updating the settings of the MERCSClassifier.
        :return:
        """

        # 1. Preliminaries
        tick = default_timer()
        self.update_settings(mode="predict", **kwargs)

        # 2. Prediction = Prepare Inference
        self.q_models = self.query_to_model(
            self.m_list,
            self.m_codes,
            self.s["prediction"],
            self.s["metadata"],
            self.s["queries"]["codes"][[q_idx]],
        )

        msg = """
        Predicting query id: \t{}\n
        Predicting query code: \t{}\n
        """.format(
            q_idx, self.s["queries"]["codes"][[q_idx]]
        )
        debug_print(msg, V=VERBOSITY)

        # 3. Inference
        X_query = perform_imputation(
            X, self.s["queries"]["codes"][q_idx], self.imputer
        )  # Generate X data

        Y = self.q_models[q_idx].predict(X_query)

        # 4. Post processing
        tock = default_timer()
        self.update_settings(mode="model_data", mod_inf_time=tock - tick)

        del X_query
        # del self.q_models

        return Y

    def predict_proba(self, X, q_idx=0, **kwargs):
        """
        Predict Y from X_test.

        Which attributes to predict are specified by queries.

        :param q_idx:       Optional keyword identifying which of the already
                            loaded queries you want to predict.
        :param X:           Test dataset. (Pandas DataFrame)
        :param kwargs:      Optional keyword arguments.
                            Used in updating the settings of the MERCSClassifier.
        :return:
        """

        tick = default_timer()
        # 0. Settings
        self.update_settings(mode="predict", **kwargs)

        # 1. Prediction = Prepare Inference
        self.q_models = self.query_to_model(
            self.m_list,
            self.m_codes,
            self.s["prediction"],
            self.s["metadata"],
            self.s["queries"]["codes"][[q_idx]],
        )
        # 2. Inference
        X_query = perform_imputation(
            X, self.s["queries"]["codes"][q_idx], self.imputer
        )  # Generate X data

        Y_proba = self.q_models[q_idx].predict_proba(X_query)
        del X_query

        tock = default_timer()
        self.update_settings(mode="model_data", mod_inf_time=tock - tick)

        return Y_proba

    def batch_predict(self, X, fnames, **kwargs):

        # 1. Preliminaries
        tick = default_timer()
        self.update_settings(mode="batch_predict", **kwargs)

        nb_queries = self.s["queries"]["codes"].shape[0]
        assert nb_queries == len(fnames)

        # 1. Prediction = Prepare Inference
        self.q_models = self.query_to_model(
            self.m_list,
            self.m_codes,
            self.s["prediction"],
            self.s["metadata"],
            self.s["queries"]["codes"],
        )

        # 2. Inference
        for q_idx in range(nb_queries):
            # Generate X data for queries with index q_idx
            X_query = perform_imputation(
                X, self.s["queries"]["codes"][q_idx], self.imputer
            )

            Y = self.q_models[q_idx].predict(X_query)

            del X_query

            save_output_data(Y, self.s["queries"]["q_targ"][q_idx], fnames[q_idx])
            del Y

        # 3. Post processing
        tock = default_timer()
        self.update_settings(mode="model_data", mod_inf_time=tock - tick)

        return

    def score_samples(self, X, aggregation=None, k=None, **kwargs):

        scores = self.score_samples_(X, **kwargs)

        if self.scores_scaler_ is not None:
            scores = self.scores_scaler_.transform(scores)

        out = self.aggregate_scores(scores, aggregation=aggregation, k=k)
        return out

    def score_samples_(self, X, **kwargs):
        n, _ = X.shape
        nb_models = len(self.m_list)

        scores = np.zeros((n, nb_models), dtype=float)

        for m_idx in range(nb_models):
            scores[:, m_idx] = self.score_samples_model(X, m_idx, **kwargs)
        return scores

    def score_samples_model(self, X, m_idx, **kwargs):
        m = self.m_list[m_idx]
        m_code = self.m_codes[m_idx, :]
        m_desc, m_targ, _ = code_to_query(m_code)

        i, o = X[:, m_desc], X[:, m_targ]

        scores = m.score_samples(i, o, **kwargs)

        return scores

    @staticmethod
    def aggregate_scores(scores, aggregation=None, k=None):

        if k is None:
            k = 3

        n, m = scores.shape

        if aggregation in {None, 'max'}:
            res = np.max(scores, axis=1)

        elif aggregation in {'mean'}:
            res = np.mean(scores, axis=1)

        elif aggregation in {'topk'}:
            if k > m:
                msg = """
                Given k value:                                      {}
                is higher than the amount of scores per instance:    {}
                Hence, we adjust k to be equal to m, and this method
                becomes equivalent to taking the mean again.
                """.format(k, m)
                warnings.warn(msg)
                k = m

            topk_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
            res = [np.mean(scores[row_idx, topk_idx[row_idx, :]])
                   for row_idx in range(n)]

        elif aggregation in {'topksum'}:
            if k > m:
                msg = """
                Given k value:                                      {}
                is higher than the amount of scores per instance:    {}
                Hence, we adjust k to be equal to m, and this method
                becomes equivalent to taking the mean again.
                """.format(k, m)
                warnings.warn(msg)
                k = m

            topk_idx = np.argpartition(scores, -k ,axis=1)[:, -k:]
            res = [np.sum(scores[row_idx, topk_idx[row_idx, :]])
                   for row_idx in range(n)]

        elif aggregation in {'sum'}:
            res = np.sum(scores, axis=1)

        else:
            msg = """
            Did not recognize kind: {}
            """.format(aggregation)
            raise ValueError(msg)

        return res

    # 0. Preliminaries
    def load_settings(self, filename, mode=None):
        """
        Load a JSON settingsfile.

        The settings are saved and loaded in JSON (dict) format.
        """

        with open(filename) as f:
            new_settings = json.load(f)

        self.import_settings(new_settings, mode=mode)

        return

    def import_settings(self, new_settings, mode=None):
        """
        Import an external settings dictionary into the classifier.



        :param new_settings:
        :param mode:
        :return:
        """

        if mode in {"induction", "ind"}:
            self.s["induction"] = new_settings
        elif mode in {"selection", "sel"}:
            self.s["selection"] = new_settings
        elif mode in {"prediction", "pred"}:
            self.s["prediction"] = new_settings
        elif mode in {"queries", "queries", "q", "qry"}:
            self.s["queries"] = new_settings
        elif mode in {"metadata", "md"}:
            self.s["metadata"] = new_settings
        elif mode in {"model_data", "mod"}:
            self.s["model_data"] = new_settings
        elif mode in {"algo", "do", "main_old"}:
            self.import_settings(new_settings["induction"], mode="induction")
            self.import_settings(new_settings["selection"], mode="selection")
            self.import_settings(new_settings["prediction"], mode="prediction")
        elif mode == "fit":
            # Assuming that new_settings has keys induction and selection
            self.import_settings(new_settings["induction"], mode="induction")
            self.import_settings(new_settings["selection"], mode="selection")
        elif mode in {"predict", "batch_predict"}:
            # Assuming that new_settings has key prediction
            self.import_settings(new_settings["prediction"], mode="prediction")
        else:
            # If no mode provided, assume global settings
            warnings.warn(
                "Did not recognize mode: {}."
                "Assuming algorithm settings.".format(mode)
            )
            self.s = new_settings

        return

    def update_settings(self, mode=None, delimiter="_", **kwargs):
        """
        Update the settings dictionary.

        :param mode:            Settings category that has to be updated
        :param delimiter:       Delimiter of the settings keywords
        :param kwargs:          Keyword arguments
        :return:
        """

        if mode in {"induction", "ind"}:
            self.s["induction"] = filter_kwargs_update_settings(
                self.s["induction"], prefix="ind", delimiter=delimiter, **kwargs
            )
        elif mode in {"selection", "sel"}:
            self.s["selection"] = filter_kwargs_update_settings(
                self.s["selection"], prefix="sel", delimiter=delimiter, **kwargs
            )
        elif mode in {"prediction", "pred"}:
            self.s["prediction"] = filter_kwargs_update_settings(
                self.s["prediction"], prefix="predict", delimiter=delimiter, **kwargs
            )
        elif mode in {"queries", "query", "q", "qry"}:
            nb_atts = self.s["metadata"].get("nb_atts", 0)
            if nb_atts > 1:
                self.s["queries"] = update_query_settings(
                    self.s["queries"], nb_atts, delimiter=delimiter, **kwargs
                )
        elif mode in {"metadata", "md"}:
            self.s["metadata"] = update_meta_data(
                self.s["metadata"], self.m_list, self.m_codes
            )
        elif mode in {"model_data"}:
            self.s["model_data"] = filter_kwargs_update_settings(
                self.s["model_data"], prefix="mod", delimiter=delimiter, **kwargs
            )
        elif mode in {"fit"}:
            self.update_settings(mode="induction", delimiter=delimiter, **kwargs)
            self.update_settings(mode="selection", delimiter=delimiter, **kwargs)
        elif mode in {"predict", "batch_predict"}:
            self.update_settings(mode="prediction", delimiter=delimiter, **kwargs)
            self.update_settings(mode="qry", delimiter=delimiter, **kwargs)
        elif mode in {None, 'init'}:
            self.update_settings(mode="induction", delimiter=delimiter, **kwargs)
            self.update_settings(mode="selection", delimiter=delimiter, **kwargs)
            self.update_settings(mode="prediction", delimiter=delimiter, **kwargs)
            self.update_settings(mode="queries", delimiter=delimiter, **kwargs)
            self.update_settings(mode="model_data", delimiter=delimiter, **kwargs)
        else:
            msg = """
            Did not recognize settings keyword:     {}
            Running a initialization procedure instead.
            """.format(mode)
            warnings.warn(msg)
            self.update_settings(mode="init", delimiter=delimiter, **kwargs)

        return

    def fit_imputer(self, X):
        """
        Construct and fit an imputer based on input data_csv.

        This to fill in missing values later on.
        """
        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        imputer.fit(X)

        self.imputer = imputer

        return

    # 1. Selection = Prepare Induction
    def perform_selection(self, metadata):

        if only_nominal_targ(metadata["is_nominal"]):
            m_codes = self.perform_selection_algorithm(metadata)
        elif only_numeric_targ(metadata["is_nominal"]):
            m_codes = self.perform_selection_algorithm(metadata)
        else:
            # Mixed case requires special treatment
            nominal_atts, numeric_atts = extract_nominal_numeric_attributes(
                self.s["metadata"]
            )

            nominal_m_codes = self.perform_selection_algorithm(
                self.s["metadata"], target_atts_list=nominal_atts
            )

            numeric_m_codes = self.perform_selection_algorithm(
                self.s["metadata"], target_atts_list=numeric_atts
            )
            m_codes = np.concatenate((nominal_m_codes, numeric_m_codes))

        return m_codes

    def perform_selection_algorithm(self, metadata, target_atts_list=None):
        """
        Generate model codes by a selection algorithm

        Model codes are arrays which encode the role each attribute of the
        dataset takes with respect to a certain model. An attribute can be
        descriptive, target or missing.

        Parameters
        ----------
        metadata: dict
            Metadata of the dataset
        target_atts_list: list, shape (nb_targ_attributes,)
            List of indices of target attributes.

        Returns
        -------

        """

        sel_type = self.s["selection"]["type"]
        keywords = kw_sel_type()

        if sel_type in keywords["base"]:
            m_codes = base_selection_algo(
                metadata, self.s["selection"], target_atts_list=target_atts_list
            )
        elif sel_type in keywords["random"]:
            m_codes = random_selection_algo(metadata, self.s["selection"])
        else:
            msg = """
            Did not recognize user-provided selection algorithm {}
            Available algorithms are {}
            Assuming `base` selection instead
            """.format(sel_type, keywords.keys())
            warnings.warn(msg)

            self.s["selection"]["type"] = next(iter(keywords["base"]))
            m_codes = self.perform_selection_algorithm(
                metadata, target_atts_list=target_atts_list
            )

        return m_codes

    # 2. Perform Induction
    def perform_induction(self, df, m_codes, settings, metadata):
        """
        Actual induction of the mde model.

        Returns a list of sklearn classifiers.
        """

        m_desc, m_targ, _ = codes_to_query(m_codes)

        # Build m_list (unfitted)
        m_list = base_ind_algo(metadata, settings, m_targ)
        nb_models = len(m_list)

        # Fit all the component models
        for m_idx in range(nb_models):
            assert isinstance(m_desc[m_idx], list)
            assert isinstance(m_targ[m_idx], list)

            m_atts = m_desc[m_idx] + m_targ[m_idx]
            X_Y = df.iloc[:, m_atts].dropna().values # Ignore rows with missing values while training

            X = X_Y[:, : len(m_desc[m_idx])]
            Y = X_Y[:, len(m_desc[m_idx]) :]

            msg = """
            X.shape: {}\n
            Y.shape: {}\n
            """.format(
                X.shape, Y.shape
            )
            debug_print(msg, V=VERBOSITY, warn=True)

            assert X.shape[1] == len(m_desc[m_idx])
            assert Y.shape[1] == len(m_targ[m_idx])

            # Convert np.array with shape (m,1) to shape (m,)
            if 1 in list(Y.shape):
                Y = Y.ravel()

            m_list[m_idx].fit(X, Y)
            del X, Y, X_Y

        flatten = self.s["induction"].get("flatten", False)
        if flatten:
            m_list, m_codes = self.flatten_model(m_list, m_codes)

        return m_codes, m_list

    # 3. Prediction = Prepare Inference
    def query_to_model(self, m_list, m_codes, settings, metadata, q_codes):
        """
        Convert query codes to query models.

        A query model is a composite model that exactly answers its
        corresponding query.

        This happens in two steps:
            1. From each query, derive a Model Activation Strategy (MAS) and a
            Attribute Activation Strategy (AAS). The first encodes when to use
            a particular model, whereas the latter encodes when to predict a
            particular attribute

            2. Convert MAS and AAS to a query-specific composite model.

        Parameters
        ----------
        m_list: list, shape (nb_models,)
            List of component models of this MERCS model
        m_codes: np.ndarray, shape (nb_models, nb_attributes)
            Two-dimensional numpy array that encodes all the component models.
            Encoding means that each component model, i, is associated with a
            code, m_codes[i, :] = code_i. Each entry, j, of that code is
            associated with an attribute, an tells us which role that attribute
            plays in model i.
        settings: dict
            Dictionary of all the settings of the MERCS model
        metadata dict
            Dictionary of all the metadata of the MERCS model
        q_codes np.ndarray, shape (nb_queries, nb_attributes)
            Two-dimensional numpy array that encodes all the queries.
            Each entry encodes the role of an attribute in the query.

        Returns
        -------

        """
        # TODO: Optimize this! Many things are unnecessarily re-derived

        # Prelims
        new_settings = {
            **settings,
            "clf_labels": metadata["clf_labels"],
            "FI": metadata["FI"],
        }  # TODO(elia): This is crap!

        if "algo" in new_settings.keys():
            algo = new_settings.get("algo", "MI")
        else:
            # Legacy compatibility.
            algo = new_settings.get("type", "MI")

        # Actual work
        if algo in {"MI", "default"}:
            mas, aas = mi_pred_algo(m_codes, q_codes)
            query_models = self.strategy_to_model(
                m_list, m_codes, q_codes, mas, aas, metadata
            )
        elif algo in {"MA", "most-appropriate"}:
            mas, aas = ma_pred_algo(m_codes, q_codes, new_settings)
            query_models = self.strategy_to_model(
                m_list, m_codes, q_codes, mas, aas, metadata
            )
        elif algo in {"MAFI", "most-appropriate-feature-importance"}:
            mas, aas = mafi_pred_algo(m_codes, q_codes, new_settings)
            query_models = self.strategy_to_model(
                m_list, m_codes, q_codes, mas, aas, metadata
            )
        elif algo in {"IT", "iterative"}:
            mas, aas = it_pred_algo(m_codes, q_codes, new_settings)
            for i, query_code in enumerate(q_codes):
                # TODO(elia) This should not happen here
                mas[i], aas[i] = full_prune_strat(m_codes, q_codes[i], mas[i], aas[i])
            query_models = self.strategy_to_model(
                m_list, m_codes, q_codes, mas, aas, metadata
            )
        elif algo in {"RW", "random-walks"}:
            assert isinstance(new_settings["param"], int)
            nb_walks = new_settings["param"]
            all_q_mods = [None] * nb_walks

            for rw_idx in range(nb_walks):
                mas, aas = rw_pred_algo(m_codes, q_codes, new_settings)

                all_q_mods[rw_idx] = self.strategy_to_model(m_list,
                                                            m_codes,
                                                            q_codes,
                                                            mas,
                                                            aas,
                                                            metadata)

            all_q_mods = np.array(all_q_mods)
            _, q_targ, _ = codes_to_query(q_codes)
            query_models = [
                build_ensemble_model(all_q_mods[:, i], targ, metadata)
                for i, targ in enumerate(q_targ)
            ]
        else:
            msg = """
            \nDid not recognize prediction method: '{}'\n
            Assuming MI algorithm instead.
            """.format(
                new_settings["type"]
            )
            warnings.warn(msg)

            settings["type"] = "MI"
            query_models = self.query_to_model(
                m_list, m_codes, settings, metadata, q_codes
            )

        return query_models

    @staticmethod
    def strategy_to_model(m_list, m_codes, q_codes, mas, aas, metadata):
        """
        Convert the MAS and AAS to a single, grouped model.

        The use of this is that the model and attribute activation strategy gets 'locked in'
        a special purpose model.

        :param m_list:          List of models
        :param m_codes:         Characteristic code of the models
        :param mas:             Model Activation Strategy
        :param aas:             Attribute Activation Strategy
        :param q_codes:         Characteristic code of the queries this model was built to address
        :param metadata:        Metadata of the composing models
        :return:
        """

        # Prelims
        assert len(mas) == len(aas) == len(q_codes)
        m_desc, m_targ, _ = codes_to_query(m_codes)
        _, q_targ, _ = codes_to_query(q_codes)

        # Convert every MAS-AAS combination to a dedicated model
        query_models = [
            build_chained_model(
                m_list, m_desc, m_targ, q_targ[i], mas[i], aas[i], metadata
            )
            for i in range(len(q_codes))
        ]

        return np.array(query_models)

    # 4. Extra stuff
    def merge(self, other):
        """
        Merge this MERCS model with another.

        Parameters
        ----------
        other: MERCS
            Other MERCS model

        Returns
        -------

        """

        own_codes = self.m_codes
        new_codes = other.m_codes
        self.m_codes = np.concatenate((own_codes, new_codes))
        self.m_list.extend(other.m_list)

        self.q_models = None  # Just a reset
        self.update_settings(mode="metadata")  # Save info on learned models

        return

    @staticmethod
    def flatten_model(m_list, m_codes):
        """
        Unravel composite models

        Unpack a composite model (e.g.; a RandomForest) to its
        fundamental components (e.g.; a DecisionTree).

        Parameters
        ----------
        m_list: list, shape (nb_models)
            List where each entry is a model
        m_codes: np.ndarray, shape (nb_models, nb_atts)
            Each row corresponds to a code which encodes the function of each
            attribute in the model the row corresponds to.

        Returns
        -------

        """

        if isinstance(m_list[0], (RandomForestClassifier, RandomForestRegressor)):

            # Actual actions
            new_m_list = []
            new_m_codes = m_codes[0:1]

            for i, m in enumerate(m_list):
                new_m_list.extend(m)
                codes = np.tile(m_codes[i], (len(m), 1))
                new_m_codes = np.concatenate((new_m_codes, codes))

            new_m_codes = new_m_codes[
                1:
            ]  # First line was filled in as an initialization and has to be gone.

            return new_m_list, new_m_codes

        else:
            return m_list, m_codes
