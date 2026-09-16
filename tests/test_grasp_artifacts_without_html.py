"""Large catalog exports can omit meshes in HTML while preserving grasp bundles."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from grasp_planning.pipeline import fabrica_pipeline as pipeline


@pytest.mark.parametrize("stage", [1, 2])
def test_bundle_export_is_retained_without_html(tmp_path, stage):
    bundle = object()
    result = SimpleNamespace(bundle=bundle, accepted_bundle=bundle)
    output = tmp_path / "grasps.json"
    kwargs = dict(planning=None, output_json=output, output_html=None)
    if stage == 1:
        kwargs["geometry"] = None
    with patch.object(pipeline, "save_grasp_bundle") as save, patch.object(pipeline, "write_debug_html") as render:
        getattr(pipeline, f"write_stage{stage}_artifacts")(result, **kwargs)
    save.assert_called_once_with(bundle, output)
    render.assert_not_called()
