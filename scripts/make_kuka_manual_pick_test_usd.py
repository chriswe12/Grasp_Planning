"""Create a standalone USD scene for manual KUKA/Y-gripper collision testing."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_USD = REPO_ROOT / "assets/usd/kuka_iiwa7_y_gripper/kuka_iiwa7_y_gripper.usda"
OBJ_PATH = REPO_ROOT / "assets/obj/fabrica/plumbers_block/0.obj"
OUTPUT_USD = REPO_ROOT / "assets/usd/kuka_iiwa7_y_gripper/kuka_iiwa7_y_gripper_manual_pick_test.usda"

MESH_SCALE = 0.01
OBJECT_XY = (0.4252643585205078, 0.05988234281539917)
GROUND_CENTER = (0.0, 0.0, -0.005)
GROUND_SCALE = (2.0, 2.0, 0.01)
ISAAC_MIN_CONTACT_OFFSET_M = 1.0e-5


def _fmt(value: float) -> str:
    text = f"{float(value):.9g}"
    return "0" if text == "-0" else text


def _parse_obj(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            _, x, y, z, *_ = line.split()
            vertices.append((float(x) * MESH_SCALE, float(y) * MESH_SCALE, float(z) * MESH_SCALE))
        elif line.startswith("f "):
            raw_indices = [part.split("/")[0] for part in line.split()[1:]]
            indices = [int(index) - 1 if int(index) > 0 else len(vertices) + int(index) for index in raw_indices]
            if len(indices) < 3:
                continue
            for i in range(1, len(indices) - 1):
                faces.append((indices[0], indices[i], indices[i + 1]))
    return vertices, faces


def _center_vertices(vertices: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    mins = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
    maxs = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    center = [(mins[axis] + maxs[axis]) * 0.5 for axis in range(3)]
    return [tuple(vertex[axis] - center[axis] for axis in range(3)) for vertex in vertices]


def _bounds(vertices: list[tuple[float, float, float]]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mins = tuple(min(vertex[axis] for vertex in vertices) for axis in range(3))
    maxs = tuple(max(vertex[axis] for vertex in vertices) for axis in range(3))
    return mins, maxs


def _write_vec3_array(handle, name: str, values: list[tuple[float, float, float]], indent: str) -> None:
    handle.write(f"{indent}{name} = [\n")
    for x, y, z in values:
        handle.write(f"{indent}    ({_fmt(x)}, {_fmt(y)}, {_fmt(z)}),\n")
    handle.write(f"{indent}]\n")


def _write_int_array(handle, name: str, values: list[int], indent: str) -> None:
    handle.write(f"{indent}{name} = [\n")
    chunk_size = 24
    for start in range(0, len(values), chunk_size):
        chunk = ", ".join(str(value) for value in values[start : start + chunk_size])
        handle.write(f"{indent}    {chunk},\n")
    handle.write(f"{indent}]\n")


def main() -> None:
    if not ROBOT_USD.is_file():
        raise FileNotFoundError(ROBOT_USD)
    if not OBJ_PATH.is_file():
        raise FileNotFoundError(OBJ_PATH)

    vertices, faces = _parse_obj(OBJ_PATH)
    vertices = _center_vertices(vertices)
    mins, maxs = _bounds(vertices)
    object_center = (float(OBJECT_XY[0]), float(OBJECT_XY[1]), -float(mins[2]))
    face_indices = [index for face in faces for index in face]
    face_counts = [3] * len(faces)

    robot_layer = ROBOT_USD.relative_to(OUTPUT_USD.parent).as_posix()
    OUTPUT_USD.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_USD.open("w", encoding="utf-8") as handle:
        handle.write("#usda 1.0\n")
        handle.write("(\n")
        handle.write('    defaultPrim = "World"\n')
        handle.write("    metersPerUnit = 1\n")
        handle.write(f"    subLayers = [ @{robot_layer}@ ]\n")
        handle.write("    upAxis = \"Z\"\n")
        handle.write(")\n\n")
        handle.write('def Xform "World"\n{\n')
        handle.write('    def PhysicsScene "PhysicsScene"\n    {\n')
        handle.write("        vector3f physics:gravityDirection = (0, 0, -1)\n")
        handle.write("        float physics:gravityMagnitude = 9.81\n")
        handle.write("    }\n\n")
        handle.write('    def Scope "Looks"\n    {\n')
        handle.write('        def Material "object_green"\n        {\n')
        handle.write("            token outputs:surface.connect = </World/Looks/object_green/Shader.outputs:surface>\n")
        handle.write('            def Shader "Shader"\n            {\n')
        handle.write('                uniform token info:id = "UsdPreviewSurface"\n')
        handle.write("                color3f inputs:diffuseColor = (0.05, 0.45, 0.34)\n")
        handle.write("                float inputs:roughness = 0.6\n")
        handle.write("                token outputs:surface\n")
        handle.write("            }\n")
        handle.write("        }\n")
        handle.write('        def Material "support_blue"\n        {\n')
        handle.write("            token outputs:surface.connect = </World/Looks/support_blue/Shader.outputs:surface>\n")
        handle.write('            def Shader "Shader"\n            {\n')
        handle.write('                uniform token info:id = "UsdPreviewSurface"\n')
        handle.write("                color3f inputs:diffuseColor = (0.15, 0.23, 0.72)\n")
        handle.write("                float inputs:roughness = 0.55\n")
        handle.write("                token outputs:surface\n")
        handle.write("            }\n")
        handle.write("        }\n")
        handle.write('        def Material "high_friction_physics" (\n')
        handle.write('            prepend apiSchemas = ["PhysicsMaterialAPI"]\n')
        handle.write("        )\n        {\n")
        handle.write("            float physics:staticFriction = 3\n")
        handle.write("            float physics:dynamicFriction = 2.5\n")
        handle.write("            float physics:restitution = 0\n")
        handle.write("        }\n")
        handle.write("    }\n\n")
        handle.write('    def Cube "GroundPlane" (\n')
        handle.write('        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]\n')
        handle.write("    )\n    {\n")
        handle.write("        double size = 1\n")
        handle.write(
            f"        double3 xformOp:translate = ({_fmt(GROUND_CENTER[0])}, {_fmt(GROUND_CENTER[1])}, {_fmt(GROUND_CENTER[2])})\n"
        )
        handle.write(
            f"        double3 xformOp:scale = ({_fmt(GROUND_SCALE[0])}, {_fmt(GROUND_SCALE[1])}, {_fmt(GROUND_SCALE[2])})\n"
        )
        handle.write('        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n')
        handle.write("        bool physics:collisionEnabled = true\n")
        handle.write("        rel material:binding = </World/Looks/support_blue>\n")
        handle.write("        rel physics:material:binding = </World/Looks/high_friction_physics>\n")
        handle.write("    }\n\n")
        handle.write('    def Xform "TestObject" (\n')
        handle.write('        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsMassAPI"]\n')
        handle.write("    )\n    {\n")
        handle.write(
            f"        double3 xformOp:translate = ({_fmt(object_center[0])}, {_fmt(object_center[1])}, {_fmt(object_center[2])})\n"
        )
        handle.write('        uniform token[] xformOpOrder = ["xformOp:translate"]\n')
        handle.write("        bool physics:rigidBodyEnabled = true\n")
        handle.write("        bool physxRigidBody:disableGravity = false\n")
        handle.write("        float physics:mass = 0.04\n")
        handle.write("        def Mesh \"plumbers_block0\" (\n")
        handle.write('            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]\n')
        handle.write("        )\n        {\n")
        handle.write('            uniform token subdivisionScheme = "none"\n')
        handle.write("            bool doubleSided = true\n")
        handle.write("            bool physics:collisionEnabled = true\n")
        handle.write('            uniform token physics:approximation = "convexHull"\n')
        handle.write(f"            float physxCollision:contactOffset = {_fmt(ISAAC_MIN_CONTACT_OFFSET_M)}\n")
        handle.write("            float physxCollision:restOffset = 0\n")
        handle.write("            rel material:binding = </World/Looks/object_green>\n")
        handle.write("            rel physics:material:binding = </World/Looks/high_friction_physics>\n")
        _write_vec3_array(handle, "            point3f[] points", vertices, "")
        _write_int_array(handle, "            int[] faceVertexCounts", face_counts, "")
        _write_int_array(handle, "            int[] faceVertexIndices", face_indices, "")
        handle.write("        }\n")
        handle.write("    }\n\n")
        handle.write('    def DistantLight "KeyLight"\n    {\n')
        handle.write("        float inputs:intensity = 500\n")
        handle.write("        angle inputs:angle = 0.5\n")
        handle.write("    }\n")
        handle.write("}\n")

    print(f"{OUTPUT_USD} object_center={object_center} object_bounds_local=({mins}, {maxs})")


if __name__ == "__main__":
    main()
