"""Adapt existing stage-2 bundles to Panda task inputs without redefining grasps."""
from pathlib import Path
import hashlib
import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import load_grasp_bundle, quat_to_rotmat_xyzw, rotmat_to_quat_xyzw
from grasp_planning.grasping.grasp_transforms import saved_grasp_to_world_grasp
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.mujoco import build_bundle_local_mesh

ROOT = Path(__file__).resolve().parents[2]
# Planner finger joint offset + fingertip contact height (collision.py).
PLANNER_HAND_TO_CONTACT_M = .0584 + .04525


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def task_pose(grasp, object_pose, tcp_offset=(0., 0., .1034)):
    """Use the shared pad-offset-corrected transform, then match task TCP."""
    world = saved_grasp_to_world_grasp(grasp, object_pose, pregrasp_offset=.065, gripper_width_clearance=.008)
    rotation = quat_to_rotmat_xyzw(world.orientation_xyzw)
    position = np.asarray(world.position_w) + rotation @ (np.asarray(tcp_offset)-[0, 0, PLANNER_HAND_TO_CONTACT_M])
    x, y, z, w = world.orientation_xyzw
    return np.r_[position, [w, x, y, z]], rotation[:, 2]


def collect_targets(benchmark_root, *, per_orientation=24, tcp_offset=(0., 0., .1034), part_key=None):
    """One object's stage-2 sources, diversified and grouped across splits."""
    pattern = 'parts/*/*/orientations/*/stage2.json'
    if part_key is not None:
        assembly, part = part_key.split('__part_')
        pattern = f'parts/{assembly}/{part}/orientations/*/stage2.json'
    paths = sorted(Path(benchmark_root).glob(pattern))
    if not paths:
        raise ValueError('No benchmark stage-2 bundles found')
    targets, sources, identity, mesh, first_bundle = [], [], None, None, None
    for path in paths:
        bundle = load_grasp_bundle(path)
        if (bundle.metadata or {}).get('gripper_collision_model') != 'franka_hand':
            raise ValueError(f'Expected a franka_hand bundle: {path}')
        local = build_bundle_local_mesh(bundle)
        key = hashlib.sha256(np.asarray(local.vertices_obj).tobytes()+np.asarray(local.faces).tobytes()).hexdigest()
        if identity is not None and key != identity:
            raise ValueError('One training catalog must reference one identical bundle-local object mesh')
        identity, mesh = key, local
        first_bundle = first_bundle or path
        source_id = portable_path(path)
        sources.append({'path': source_id, 'sha256': sha256_file(path), 'candidates': len(bundle.candidates)})
        pose = bundle.metadata['execution_world_pose']
        base_rotation = quat_to_rotmat_xyzw(pose['orientation_xyzw_world'])
        # Prefer geometric diversity before filling with nearby patch variants.
        candidates = sorted(bundle.candidates, key=lambda c: (-float(c.score or 0), c.grasp_id))
        candidates = [c for c in candidates if .002 <= c.jaw_width <= .072]
        chosen = []
        for c in candidates:
            rc = quat_to_rotmat_xyzw(c.grasp_orientation_xyzw_obj)
            if all(np.linalg.norm(np.asarray(c.grasp_position_obj)-p.grasp_position_obj) > .004 or
                   np.linalg.norm(rc-quat_to_rotmat_xyzw(p.grasp_orientation_xyzw_obj)) > .35 for p in chosen):
                chosen.append(c)
            if len(chosen) >= per_orientation:
                break
        for c in candidates:
            if len(chosen) >= per_orientation:
                break
            if c.grasp_id not in {p.grasp_id for p in chosen}:
                chosen.append(c)
        for c in chosen:
            group = f'{bundle.target_mesh_path}:{c.grasp_id}'
            bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
            split = 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'
            for placement, (px, py, yaw) in enumerate(((.43, -.035, -.18), (.48, .035, .18))):
                co, si = np.cos(yaw), np.sin(yaw)
                rz = np.array([[co,-si,0],[si,co,0],[0,0,1]])
                rotation = rz @ base_rotation
                # Ground from the actual bundle mesh, not an asset-frame bounding box.
                pz = -float((np.asarray(mesh.vertices_obj) @ rotation.T)[:, 2].min())
                quat = rotmat_to_quat_xyzw(rotation)
                obj = ObjectWorldPose(position_world=(px,py,pz), orientation_xyzw_world=quat)
                goal, approach = task_pose(c, obj, tcp_offset)
                targets.append({'target_id': f'{path.parent.name}__{c.grasp_id}__p{placement}',
                    'source_bundle': source_id, 'source_grasp_id': c.grasp_id,
                    'orientation_id': path.parent.name, 'split': split,
                    'jaw_width_m': c.jaw_width, 'open_width_m': c.jaw_width+.008,
                    'object_pose': [px,py,pz,quat[3],*quat[:3]],
                    'goal_pose': goal.tolist(), 'approach_axis': approach.tolist(), 'score': c.score})
    if part_key is not None:
        for target in targets:
            target['target_id'] = f"{part_key}__{target['target_id']}"
            target['part_key'] = part_key
            # Reserve gamepad entirely for unseen-assembly evaluation and a
            # deterministic fifth of other parts for unseen-part validation.
            bucket = int(hashlib.sha256(part_key.encode()).hexdigest()[:8], 16) % 5
            if part_key.startswith('gamepad__'):
                target['split'] = 'test'
            elif bucket == 0:
                target['split'] = 'validation'
    return {'schema_version': 1, 'sources': sources, 'mesh_identity': identity,
            'first_bundle': portable_path(first_bundle), 'targets': targets,
            'planner_hand_to_contact_m': PLANNER_HAND_TO_CONTACT_M,
            'task_tcp_offset_m': list(tcp_offset)}, mesh
