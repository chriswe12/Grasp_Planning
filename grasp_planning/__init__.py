"""Grasp planning package."""

from importlib import import_module

from .grasping import (
    AntipodalGraspGeneratorConfig,
    AntipodalMeshGraspGenerator,
    CubeFaceGraspGenerator,
    FingerBoxGripperCollisionModel,
    GraspCandidate,
    GraspCollisionEvaluator,
    ObjectFrameGraspCandidate,
    PickupPlacementSpec,
    SavedGraspBundle,
    SavedGraspCandidate,
    TriangleMesh,
    WorldFrameGraspCandidate,
    accepted_grasps,
    build_pickup_pose_world,
    evaluate_saved_grasps_against_pickup_pose,
    export_grasp_candidates_json,
    finger_box_corners,
    finger_boxes_from_grasp,
    load_grasp_bundle,
    sample_pickup_placement_spec,
    saved_grasp_to_world_grasp,
    score_grasps,
    select_first_feasible_grasp,
)

_LAZY_EXPORTS = {
    "FR3PickController": ("grasp_planning.controllers", "FR3PickController"),
    "AdmittanceControllerCfg": ("grasp_planning.planning", "AdmittanceControllerCfg"),
    "FR3AdmittanceController": ("grasp_planning.planning", "FR3AdmittanceController"),
    "FR3MoveToPoseController": ("grasp_planning.planning", "FR3MoveToPoseController"),
    "JointTrajectory": ("grasp_planning.planning", "JointTrajectory"),
    "PlanResult": ("grasp_planning.planning", "PlanResult"),
    "PoseCommand": ("grasp_planning.planning", "PoseCommand"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'grasp_planning' has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "AdmittanceControllerCfg",
    "AntipodalGraspGeneratorConfig",
    "AntipodalMeshGraspGenerator",
    "CubeFaceGraspGenerator",
    "FingerBoxGripperCollisionModel",
    "FR3AdmittanceController",
    "FR3MoveToPoseController",
    "FR3PickController",
    "GraspCandidate",
    "GraspCollisionEvaluator",
    "JointTrajectory",
    "ObjectFrameGraspCandidate",
    "PlanResult",
    "PickupPlacementSpec",
    "PoseCommand",
    "SavedGraspBundle",
    "SavedGraspCandidate",
    "TriangleMesh",
    "WorldFrameGraspCandidate",
    "accepted_grasps",
    "build_pickup_pose_world",
    "evaluate_saved_grasps_against_pickup_pose",
    "export_grasp_candidates_json",
    "finger_box_corners",
    "finger_boxes_from_grasp",
    "load_grasp_bundle",
    "sample_pickup_placement_spec",
    "saved_grasp_to_world_grasp",
    "score_grasps",
    "select_first_feasible_grasp",
]
