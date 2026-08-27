from __future__ import annotations

import sys
from types import ModuleType

from grasp_planning.isaac_visual_scene import (
    VISUAL_SERVO_KEY_ROTATION_WXYZ,
    spawn_visual_servo_lights,
)


def test_global_key_light_orientation_is_authored_by_the_spawner(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeLightCfg:
        def __init__(self, **_kwargs: object) -> None:
            self.func = self.spawn

        def spawn(
            self,
            prim_path: str,
            _cfg: object,
            **kwargs: object,
        ) -> object:
            calls.append((prim_path, kwargs))
            return object()

    isaaclab_module = ModuleType("isaaclab")
    sim_module = ModuleType("isaaclab.sim")
    sim_module.DomeLightCfg = FakeLightCfg
    sim_module.DistantLightCfg = FakeLightCfg
    isaaclab_module.sim = sim_module
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_module)

    paths = spawn_visual_servo_lights()

    assert paths == {
        "dome": "/World/DomeLight",
        "key": "/World/VisualServoKeyLight",
    }
    assert calls == [
        ("/World/DomeLight", {}),
        (
            "/World/VisualServoKeyLight",
            {"orientation": VISUAL_SERVO_KEY_ROTATION_WXYZ},
        ),
    ]
