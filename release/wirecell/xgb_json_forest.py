#!/usr/bin/env python3
"""A reader of an xgboost JSON model, written the way a C++ class in the toolkit would be.

The point is not to score in Python -- `xgboost` itself does that -- but to pin down, field by
field, exactly what a `XgbJsonForest` next to `TmvaGradForest` has to do, and to prove on real
models that nothing else is needed. Every value this reader keeps is a plain array of numbers,
so the C++ form is the same arrays.

What a node is (xgboost's JSON, `learner.gradient_booster.model.trees[t]`):
  left_children[i] == -1            -> leaf, its value is split_conditions[i]
  otherwise                          -> split on feature split_indices[i] at split_conditions[i]
  x < cut   -> left_children[i];  x >= cut -> right_children[i]
  x missing -> default_left[i] ? left : right          <- this is what TMVA XML cannot express
The model's value is the sum of one leaf per tree plus the base margin, which for
`base_score = 0.5` with `binary:logistic` is 0. That sum is the raw margin (`output_margin`).
"""
import json
import math

import numpy as np


class XgbJsonForest:
    """One xgboost JSON model: the arrays, the feature names, and the raw-margin evaluation."""

    def __init__(self, path):
        with open(path) as f:
            doc = json.load(f)
        learner = doc['learner']
        model = learner['gradient_booster']['model']
        self.path = path
        self.names = list(learner['feature_names'])
        self.objective = learner['objective']['name']
        # the margin the model starts from; `binary:logistic` with base_score 0.5 starts at 0
        # xgboost 3.x writes this as a one-element vector, "[5E-1]"; 2.x writes "5E-1"
        base = float(learner['learner_model_param']['base_score'].strip('[]'))
        self.base_margin = math.log(base / (1.0 - base)) if self.objective == 'binary:logistic' else base
        if any(t['categories'] for t in model['trees']):
            raise ValueError('categorical splits are not handled: %s' % path)
        if any(s != 0 for t in model['trees'] for s in t['split_type']):
            raise ValueError('a non-numeric split type is not handled: %s' % path)
        self.trees = []
        for t in model['trees']:
            self.trees.append(dict(
                left=np.asarray(t['left_children'], dtype=np.int32),
                right=np.asarray(t['right_children'], dtype=np.int32),
                ivar=np.asarray(t['split_indices'], dtype=np.int32),
                cut=np.asarray(t['split_conditions'], dtype=np.float32),
                default_left=np.asarray(t['default_left'], dtype=np.int32),
            ))

    # --- numbers a C++ class would report at configure time -------------------
    @property
    def ntrees(self):
        return len(self.trees)

    @property
    def nnodes(self):
        return sum(len(t['left']) for t in self.trees)

    @property
    def nvars(self):
        return len(self.names)

    def default_left_fraction(self):
        """Share of the splitting nodes whose missing value goes left."""
        n = d = 0
        for t in self.trees:
            inner = t['left'] >= 0
            n += int(inner.sum())
            d += int(t['default_left'][inner].sum())
        return d, n

    def margin_bounds(self):
        """The lowest and highest raw margin the forest can give (one leaf per tree)."""
        lo = hi = self.base_margin
        for t in self.trees:
            leaf = t['left'] < 0
            lo += float(t['cut'][leaf].min())
            hi += float(t['cut'][leaf].max())
        return lo, hi

    # --- the evaluation itself ------------------------------------------------
    def evaluate(self, x):
        """Raw margin for one row `x` (a float32 array of nvars() values; NaN = missing).

        The sum is kept in float32, as xgboost's own predictor does: in float64 the result
        differs from xgboost by up to 1e-5, in float32 it agrees to the last bit.
        """
        total = np.float32(self.base_margin)
        for t in self.trees:
            left, right, ivar, cut, dleft = t['left'], t['right'], t['ivar'], t['cut'], t['default_left']
            i = 0
            while left[i] >= 0:
                v = x[ivar[i]]
                if np.isnan(v):
                    i = left[i] if dleft[i] else right[i]
                else:
                    i = left[i] if v < cut[i] else right[i]
            total = np.float32(total + cut[i])
        return float(total)

    def evaluate_many(self, X):
        return np.array([self.evaluate(np.asarray(r, dtype=np.float32)) for r in X], dtype=np.float32)
