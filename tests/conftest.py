"""Make the repo root importable so tests can import the top-level modules
(local_vuln_scanner, to_sarif, evaluate_model, ...) when pytest is run from
anywhere. Kept torch-free: importing these modules must not pull in torch."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
