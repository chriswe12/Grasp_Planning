from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from grasp_planning.grasping.mesh_antipodal_grasp_generator import TriangleMesh
from grasp_planning.pipeline import fabrica_pipeline as pipeline
from scripts.run_grasp_ablation_suite import paired_interval, rank_candidates, save
from scripts.run_grasp_pipeline import _planning_config


def test_all_upright_switch_reaches_generator_and_separates_cache(monkeypatch, tmp_path):
    mesh = TriangleMesh(
        vertices_obj=np.array([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]]),
        faces=np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]),
    )
    configs = []

    class FakeGenerator:
        collision_backend_name = "test"

        def __init__(self, config):
            configs.append(config)

        def generate(self, mesh):
            return []

    monkeypatch.setattr(pipeline, "load_asset_mesh", lambda *a, **k: mesh)
    monkeypatch.setattr(pipeline, "AntipodalMeshGraspGenerator", FakeGenerator)
    geometry = pipeline.GeometryConfig(target_mesh_path="obj/fabrica/plumbers_block/2.obj", mesh_scale=0.01)
    p = _planning_config(
        {
            "planning": {
                "stage1_upright_axes_enabled": False,
                "stage1_cache_enabled": True,
                "stage1_cache_dir": str(tmp_path),
                "skip_stage1_collision_checks": True,
            }
        }
    )
    disabled = pipeline.generate_stage1_result(
        geometry=geometry, planning=p, upright_approach_axes_obj=((1.0, 0.0, 0.0),)
    )
    enabled = pipeline.generate_stage1_result(geometry=geometry, planning=replace(p, stage1_upright_axes_enabled=True))
    assert configs[0].upright_approach_axes_obj == ()
    assert configs[1].upright_approach_axes_obj
    assert disabled.bundle.metadata["upright_approach_axes_obj"] == []
    assert disabled.bundle.metadata["stage1_cache_key"] != enabled.bundle.metadata["stage1_cache_key"]
    assert not enabled.bundle.metadata["stage1_cache_hit"]


def test_rank_ablation_removes_support_but_preserves_candidate_pool():
    def candidate(name, align, support):
        return SimpleNamespace(
            grasp_id=name,
            score_components={
                "antipodal_alignment": align,
                "centering": 0.5,
                "contact_support": support,
                "com_offset": 0.5,
                "top_down_approach": 0.5,
                "top_grasp_score_weight": 0.35,
            },
        )

    values = [candidate("support", 0.5, 1), candidate("alignment", 0.8, 0)]
    assert rank_candidates(values, "full") == values
    reordered = rank_candidates(values, "without_contact_support")
    assert reordered[0].grasp_id == "alignment"
    assert {id(c) for c in values} == {id(c) for c in reordered}
    assert values[0].grasp_id == "support"


def test_paired_interval_refuses_missing_cases():
    rows = [dict(assembly="a", part_id=str(i), orientation_id="o", status="direct_success") for i in range(3)]
    assert paired_interval(rows[:-1], rows) is None
    assert paired_interval(rows, rows) == [0.0, 0.0, 0.0]


def test_atomic_result_write(tmp_path):
    p = tmp_path / "job/result.json"
    save(p, {"state": "complete"})
    assert p.read_text().find("complete") >= 0
    assert not Path(str(p) + ".tmp").exists()
