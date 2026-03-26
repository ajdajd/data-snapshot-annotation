import json

from dsa.constants import ROOT
from dsa.utils import load_json
from dsa.adapters.labelstudio import (
    convert_labelstudio_export_to_eval_v13,
    LS_EXPORT_JSON_PATH,
)


def test_labelstudio():
    ref_path = ROOT / "tests/data/ground_truth.json"
    test_path = ROOT / "tests/data/ground_truth_test.json"

    convert_labelstudio_export_to_eval_v13(LS_EXPORT_JSON_PATH, test_path)

    ref = load_json(ref_path)
    test = load_json(test_path)

    assert json.dumps(ref) == json.dumps(test)

    # Delete test file
    test_path.unlink()
