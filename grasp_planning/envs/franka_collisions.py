"""Runtime collision helpers for Franka assets."""

from __future__ import annotations


def expose_franka_mesh_collisions(robot_prim_path: str = "/World/envs/env_0/Robot") -> tuple[int, tuple[str, ...]]:
    """Expose existing Franka meshes as PhysX collision geometry in the current stage."""

    import omni.usd
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(robot_prim_path)
    if not root_prim.IsValid():
        return 0, ()

    to_visit = [root_prim]
    while to_visit:
        prim = to_visit.pop(0)
        if prim.IsInstance():
            prim.SetInstanceable(False)
        to_visit.extend(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))

    enabled_paths = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        prim_path = prim.GetPath().pathString
        if not prim_path.startswith(robot_prim_path):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue

        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            PhysxSchema.PhysxCollisionAPI.Apply(prim)
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr("convexHull")
        enabled_paths.append(prim_path)
    return len(enabled_paths), tuple(enabled_paths)
