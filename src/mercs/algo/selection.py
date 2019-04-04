# -*- coding: UTF-8 -*-
"""
mercs.algo.prediction
--- - --- - --- - ---

This module takes care of selection in MERCS.

The various selection algorithms are implemented in this module.

author:
    Evgeniya Korneva, Elia Van Wolputte
copyright:
    Copyright 2017-2018 KU Leuven, DTAI Research Group.
license:
    Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np
import warnings

from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.ensemble import *


def base_selection_algo(metadata, settings, target_atts_list=None):
    """
    Base selection strategy.

    This method implements the base selection strategy, when attributes are grouped together
    randomly into disjoint target sets. More specifically,
        - if sel_param < 1, each model predicts (100*sel_param)% of the dataset attributes
        - if sel_param >= 1, each model predicts exactly (sel_param) attributes
        - default: 1 model per attribute

    For each selection iteration (sel_its parameter), each attribute appears exactly once in the target set.
    For each model, all the attributes that are not in the target set constitute the descriptive set.

    Parameters
    ----------
    metadata: dict
        Dictionary that contains metadata of the training set
    settings: dict
        Dictionary of the settings of MERCS. Relevant settings are:
            1. settings['param']
            2. settings['its']
    target_atts_list: list, shape (nb_targ_atts, )
        List of indices of target attributes.

    Returns
    -------
    codes: np.ndarray, shape (nb_models, nb_atts)
        Two-dimensional np.ndarray where each row encodes a single model.
    """

    nb_atts = metadata["nb_atts"]
    param = settings["param"]
    nb_partitions = settings["its"]

    # If not specified, all attributes can appear as targets
    if target_atts_list is None:
        target_atts_list = list(range(nb_atts))
    # Otherwise, use only indicated attributes

    nb_target_atts = len(target_atts_list)

    nb_out_atts = _set_nb_out_params_(param, nb_atts)

    # Number of models per partition
    nb_models_part = int(np.ceil(nb_target_atts / nb_out_atts))
    # Total number of models
    nb_models = int(nb_partitions * nb_models_part)

    # One code per model
    codes = [[]] * nb_models

    # We start with everything descriptive
    for tree in range(nb_models):
        codes[tree] = [0] * nb_atts

    for partition in range(nb_partitions):
        for attribute in target_atts_list:
            # Randomly pick up a model to assign the attribute to
            random_model = np.random.randint(nb_models_part)
            iter = 0
            # Move to the first model that can still have additional target attribute
            while (
                np.sum(codes[partition * nb_models_part + random_model]) >= nb_out_atts
            ):
                random_model = np.mod(random_model + 1, nb_models_part)
                iter = iter + 1
                # Avoiding infinite loop
                if iter > nb_models_part:
                    break
            codes[partition * nb_models_part + random_model][attribute] = 1

    codes = np.array(codes)

    return codes


def random_selection_algo(metadata, settings):
    nb_atts = metadata["nb_atts"]
    nb_tgt = settings.get("param", 1)
    nb_iterations = settings.get("its", 1)
    fraction_missing = settings.get("fraction", 0.2)

    nb_tgt = _set_nb_out_params_(nb_tgt, nb_atts)

    att_idx = np.array(range(nb_atts))
    result = np.zeros((1, nb_atts))

    for it_idx in range(nb_iterations):
        np.random.shuffle(att_idx)

        codes = _create_init(nb_atts, nb_tgt)
        codes = _add_missing(codes, fraction=fraction_missing)
        codes = _ensure_desc_atts(codes)
        codes = codes[:, att_idx]
        result = np.concatenate((result, codes))

    return result[1:, :]


# Helpers
def _create_init(nb_atts, nb_tgt):
    res = np.zeros((nb_atts, nb_atts))
    for k in range(nb_tgt):
        res += np.eye(nb_atts, k=k)

    return res[0::nb_tgt, :]


def _add_missing(init, fraction=0.2):
    random = np.random.rand(*init.shape)

    noise = np.where(init == 0, random, init)
    missing = np.where(noise < fraction, -1, noise)

    res = np.floor(missing)

    res = _ensure_desc_atts(res)
    return res


def _ensure_desc_atts(m_codes):
    for row in m_codes:
        if 0 not in np.unique(row):
            idx_of_minus_ones = np.where(row == -1)[0]
            idx_to_change_to_zero = np.random.choice(idx_of_minus_ones)
            row[idx_to_change_to_zero] = 0

    return m_codes


def _set_nb_out_params_(param, nb_atts):
    if (param > 0) & (param < 1):
        nb_out_atts = int(np.ceil(param * nb_atts))
    elif (param >= 1) & (param < nb_atts):
        nb_out_atts = int(param)
    else:
        msg = """
        Impossible number of output attributes per model: {}\n
        This means the value of settings['selection']['param'] was set
        incorrectly.\n
        Re-adjusted to default; one model per attribute.
        """.format(param)
        warnings.warn(msg)
        nb_out_atts = 1

    return nb_out_atts
