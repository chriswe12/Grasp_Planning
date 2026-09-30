"""Versioned Franka rendering optimization; robot, surface colors and poses are unchanged."""

from copy import deepcopy

FAST_PROFILE = {
    "version": "franka_fast_rgbd_v2",
    "background": "remove_background_details_v1",
    "wear_visibility": "seeded_once_per_environment",
    "render_after_physics_and_resets": True,
    "reset_render_count": 5,
    "renderer": "preserve_isaaclab_2_3_balanced",
    "protected_surfaces": ["table", "floor", "parts"],
    "color_policy": "preserve_source_materials_and_randomization_palette",
}


def validate_performance_profile(profile):
    expected = deepcopy(FAST_PROFILE)
    expected["renderer"] = profile.get("renderer")
    if profile != expected or profile["renderer"] not in ("preserve_isaaclab_2_3_balanced", "balanced_fxaa"):
        raise ValueError("Unsupported Franka performance profile")
    return profile


def optimized_contract(source, renderer="preserve_isaaclab_2_3_balanced"):
    """Only this opt-in visual upgrade may reuse the validated geometry/pose bank."""
    if not source.get("lab_scene") or not source.get("appearance_randomization"):
        raise ValueError("Fast profile requires the existing randomized lab contract")
    result = deepcopy(source)
    result["performance_profile"] = deepcopy(FAST_PROFILE)
    result["performance_profile"]["renderer"] = renderer
    validate_performance_profile(result["performance_profile"])
    return result


def configure_fast_render(cfg, profile):
    validate_performance_profile(profile)
    # Preserve lighting and five-frame reset settling. The FXAA variant replaces
    # neural reconstruction but retains the standard lighting denoiser. One-frame
    # alternatives were rejected after rendered tests showed noise/ghosting.
    if profile["renderer"] == "balanced_fxaa":
        cfg.sim.render.antialiasing_mode = "FXAA"
        cfg.sim.render.enable_dl_denoiser = False
    cfg.num_rerenders_on_reset = profile["reset_render_count"]


def simplify_background(stage, roots):
    """Hide decorative background objects once. Preserve all physics and protected surfaces.

    The original asset and its shaders/textures are never edited. Table, floor,
    robot, task parts and optional tabletop distractors are not selected here.
    """
    from pxr import UsdGeom

    hidden = []
    for root in roots:
        lab = stage.GetPrimAtPath(root + "/Lab")
        candidates = [p for p in lab.GetChildren() if p.GetName().startswith("Loose_floor_cable")]
        room = stage.GetPrimAtPath(root + "/Lab/Room")
        if room:
            candidates += [p for p in room.GetChildren() if p.GetName().startswith(("Outlet_", "Wall_power_lead"))]
        desk = stage.GetPrimAtPath(root + "/Lab/Background_Workbench")
        if desk:
            candidates += [p for p in desk.GetChildren() if p.GetName().startswith(("Keycap", "Keyboard", "Monitor_"))]
        for prim in candidates:
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            hidden.append(str(prim.GetPath()))
    return hidden


def resume_epoch_for_frames(frames, total_envs, horizon=64):
    rollout = total_envs * horizon
    if frames < 0 or total_envs <= 0 or frames % rollout:
        raise ValueError("Checkpoint experience must align with the new global rollout batch")
    return frames // rollout
