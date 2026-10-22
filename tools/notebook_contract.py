"""Load selected definitions from the assessed notebook without running its workflow.

This adapter executes only explicitly named functions/constants, with controlled
scientific imports. No cells that fit models, write files or fetch data execute.
Tests therefore exercise the notebook implementation rather than a copied model.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

FUNCTIONS = {
    'find_exports', 'load_exports', 'audit_pairs', 'run_offline_checks',
    'split_by_season', 'make_pipeline', 'candidate_models',
    'prepare_model_inputs', 'evaluate_split', 'prediction_frame',
}
CONSTANTS = {
    'REQUIRED_COLUMNS', 'NUMERIC_FIELDS', 'OPTIONAL_COLUMNS', 'KEYS',
    'BASE_NUM', 'BASE_CAT', 'TELEM_NUM', 'FEATURE_SETS',
}


def definition_tree(path):
    """Return allowlisted notebook definitions as an AST; fail if any disappear."""
    notebook = json.loads(Path(path).read_text(encoding='utf-8'))
    nodes, found = [], set()
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
                nodes.append(node)
                found.add(node.name)
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in CONSTANTS):
                nodes.append(node)
                found.add(node.targets[0].id)
    missing = (FUNCTIONS | CONSTANTS) - found
    if missing:
        raise ValueError(f'Notebook contract definitions missing: {sorted(missing)}')
    return ast.Module(body=nodes, type_ignores=[])


def load_contract(path):
    """Return actual notebook helpers, configured to the frozen season split."""
    namespace = dict(globals(), SEED=42, MAX_RF_TREES=150,
                     TRAIN_YEARS=(2022, 2023), VAL_YEARS=(2024,), TEST_YEARS=(2025,))
    exec(compile(definition_tree(path), str(path), 'exec'), namespace)
    return namespace
