"""Download, expand, and convert multiple Franka robot description variants.

This script is intended to make asset-comparison experiments reproducible:

1. Download one or more official ``franka_description`` release archives.
2. Expand selected robot xacro files into plain URDF with absolute mesh paths.
3. Optionally convert each generated URDF into a USD articulation using Isaac Lab's
   ``UrdfConverter`` when Isaac Sim is available.
4. Write a manifest JSON describing all prepared variants.

The script can also include the repository's vendored ``assets/urdf/franka_description``
    tree as a local baseline source.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_TAGS = ("2.3.2", "2.4.0", "2.6.0")
DEFAULT_ROBOT_TYPES = ("fr3", "fr3v2", "fr3v2_1")
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_VENDOR_SOURCE = REPO_ROOT / "assets" / "urdf" / "franka_description"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        default=[],
        help="Official franka_description release tag to download. May be passed multiple times.",
    )
    parser.add_argument(
        "--robot-type",
        dest="robot_types",
        action="append",
        default=[],
        help="Robot type to expand and convert. May be passed multiple times.",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=REPO_ROOT / "third_party" / "franka_description_tags",
        help="Directory used for upstream archives and extracted source trees.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "franka_asset_variants",
        help="Directory used for generated URDF/USD assets and manifest files.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional explicit manifest output path. Defaults under --output-root.",
    )
    parser.add_argument(
        "--include-local-vendor",
        action="store_true",
        help="Also prepare variants from this repository's vendored franka_description tree.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download missing release archives. Expect them to already exist under --download-root.",
    )
    parser.add_argument(
        "--skip-usd",
        action="store_true",
        help="Generate URDF only and skip URDF-to-USD conversion.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Re-download archives even if the tarballs already exist.",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Re-extract upstream source trees even if they already exist.",
    )
    parser.add_argument(
        "--force-regenerate-urdf",
        action="store_true",
        help="Rebuild URDF files even if they already exist.",
    )
    parser.add_argument(
        "--force-usd-conversion",
        action="store_true",
        help="Force URDF-to-USD conversion even when output USDs already exist.",
    )
    parser.add_argument(
        "--fix-base",
        action="store_true",
        default=True,
        help="Import URDF variants with a fixed base. Default: true.",
    )
    parser.add_argument(
        "--no-fix-base",
        action="store_false",
        dest="fix_base",
        help="Import URDF variants without fixing the base.",
    )
    parser.add_argument(
        "--merge-fixed-joints",
        action="store_true",
        default=False,
        help="Merge fixed joints during URDF import. Default: false.",
    )
    parser.add_argument(
        "--no-merge-fixed-joints",
        action="store_false",
        dest="merge_fixed_joints",
        help="Disable fixed-joint merging during URDF import.",
    )
    parser.add_argument(
        "--self-collision",
        action="store_true",
        default=False,
        help="Enable self-collision during URDF import.",
    )
    parser.add_argument(
        "--joint-stiffness",
        type=float,
        default=400.0,
        help="Joint position-drive stiffness used when importing URDFs to USD.",
    )
    parser.add_argument(
        "--joint-damping",
        type=float,
        default=40.0,
        help="Joint position-drive damping used when importing URDFs to USD.",
    )
    return parser.parse_args()


def _download_url_for_tag(tag: str) -> str:
    return f"https://github.com/frankarobotics/franka_description/archive/refs/tags/{tag}.tar.gz"


def _canonical_robot_type(robot_type: str) -> str:
    aliases = {
        "fr3v2.1": "fr3v2_1",
        "fr3v2_1": "fr3v2_1",
    }
    return aliases.get(robot_type, robot_type)


@dataclass
class PreparedVariant:
    source_label: str
    source_root: str
    source_kind: str
    source_tag: str | None
    robot_type: str
    urdf_path: str
    usd_path: str | None
    metadata: dict[str, Any]


def _download_archive(url: str, dst_path: Path, *, force: bool) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and not force:
        print(f"[INFO] Reusing archive {dst_path}", flush=True)
        return
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"[INFO] Downloading {url} -> {dst_path}", flush=True)
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as file_obj:
        shutil.copyfileobj(response, file_obj)
    tmp_path.replace(dst_path)


def _looks_like_franka_description_root(source_root: Path) -> bool:
    return (
        (source_root / "robots").is_dir()
        and (source_root / "robots" / "common").is_dir()
        and (source_root / "end_effectors").is_dir()
    )


def _extract_archive(archive_path: Path, extract_root: Path, *, force: bool) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)
    version_dir_name = archive_path.name.removesuffix(".tar.gz")
    extracted_dir = extract_root / version_dir_name
    if extracted_dir.exists() and not force:
        if _looks_like_franka_description_root(extracted_dir):
            print(f"[INFO] Reusing extracted tree {extracted_dir}", flush=True)
            return extracted_dir
        print(f"[WARN] Existing extracted tree is incomplete, re-extracting: {extracted_dir}", flush=True)
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    print(f"[INFO] Extracting {archive_path} -> {extract_root}", flush=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(extract_root)
    if not _looks_like_franka_description_root(extracted_dir):
        raise RuntimeError(f"Extracted source tree does not look like franka_description: {extracted_dir}")
    return extracted_dir


def _fake_ament_prefix(source_root: Path, work_root: Path) -> Path:
    prefix_root = work_root / ".ament_prefix"
    share_root = prefix_root / "share"
    share_root.mkdir(parents=True, exist_ok=True)
    package_index_dir = share_root / "ament_index" / "resource_index" / "packages"
    package_index_dir.mkdir(parents=True, exist_ok=True)
    (package_index_dir / "franka_description").write_text("", encoding="utf-8")
    target = share_root / "franka_description"
    if target.is_symlink() or target.exists():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.symlink_to(source_root, target_is_directory=True)
    return prefix_root


def _rewrite_package_uris(urdf_text: str, source_root: Path) -> str:
    replacements = {
        "package://franka_description/": f"{source_root.as_posix()}/",
        "package://${description_pkg}/": f"{source_root.as_posix()}/",
    }
    for old, new in replacements.items():
        urdf_text = urdf_text.replace(old, new)
    return urdf_text


def _robot_xacro_path(source_root: Path, robot_type: str) -> Path:
    candidates = (
        source_root / "robots" / robot_type / f"{robot_type}.urdf.xacro",
        source_root / "assets" / "urdf" / "franka_description" / "robots" / robot_type / f"{robot_type}.urdf.xacro",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Unable to locate xacro for robot_type='{robot_type}' under '{source_root}'.")


def _expand_xacro_to_urdf(
    *,
    source_root: Path,
    robot_type: str,
    urdf_path: Path,
    force: bool,
) -> Path:
    if urdf_path.exists() and not force:
        print(f"[INFO] Reusing URDF {urdf_path}", flush=True)
        return urdf_path

    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    xacro_path = _robot_xacro_path(source_root, robot_type)
    work_root = urdf_path.parent / f".xacro_env_{robot_type}"
    prefix_root = _fake_ament_prefix(source_root, work_root)

    env = os.environ.copy()
    env["AMENT_PREFIX_PATH"] = str(prefix_root)
    env["COLCON_PREFIX_PATH"] = str(prefix_root)
    env["ROS_PACKAGE_PATH"] = str(prefix_root / "share")
    stub_root = Path(__file__).resolve().parent.parent / "python_stubs"
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_entries = [str(stub_root)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    interpreter_xacro = Path(sys.executable).with_name("xacro")
    xacro_executable = None
    if interpreter_xacro.exists():
        xacro_executable = str(interpreter_xacro)
    else:
        xacro_executable = shutil.which("xacro")
    if xacro_executable is not None:
        command = [
            xacro_executable,
            str(xacro_path),
            f"robot_type:={robot_type}",
        ]
    elif importlib.util.find_spec("xacro") is not None:
        command = [
            sys.executable,
            "-m",
            "xacro",
            str(xacro_path),
            f"robot_type:={robot_type}",
        ]
    else:
        command = [
            "xacro",
            str(xacro_path),
            f"robot_type:={robot_type}",
        ]
    print(f"[INFO] Expanding xacro for {robot_type} from {xacro_path}", flush=True)
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=source_root, env=env)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "xacro is not available in the current environment. "
            "Install the xacro Python package in the active interpreter or provide a shell xacro executable."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"xacro expansion failed for robot_type='{robot_type}' from '{xacro_path}'. stderr:\n{exc.stderr}"
        ) from exc
    urdf_text = _rewrite_package_uris(result.stdout, source_root)
    urdf_path.write_text(urdf_text, encoding="utf-8")
    return urdf_path


def _run_usd_conversion(
    *,
    urdf_path: Path,
    usd_path: Path,
    force: bool,
    fix_base: bool,
    merge_fixed_joints: bool,
    self_collision: bool,
    joint_stiffness: float,
    joint_damping: float,
) -> Path:
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    if usd_path.exists() and not force:
        print(f"[INFO] Reusing USD {usd_path}", flush=True)
        return usd_path

    # Isaac imports must happen after the app is launched.
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(usd_path.parent),
        usd_file_name=usd_path.name,
        force_usd_conversion=True,
        fix_base=fix_base,
        merge_fixed_joints=merge_fixed_joints,
        self_collision=self_collision,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=float(joint_stiffness),
                damping=float(joint_damping),
            ),
            target_type="position",
            drive_type="force",
        ),
    )
    print(f"[INFO] Converting URDF -> USD: {urdf_path} -> {usd_path}", flush=True)
    converter = UrdfConverter(cfg)
    converted = Path(converter.usd_path).resolve()
    if converted != usd_path.resolve():
        shutil.copy2(converted, usd_path)
    return usd_path


def _source_roots_from_args(args: argparse.Namespace) -> list[tuple[str, str | None, Path, str]]:
    sources: list[tuple[str, str | None, Path, str]] = []
    tags = tuple(dict.fromkeys(args.tags or DEFAULT_TAGS))
    for tag in tags:
        archive_path = args.download_root / f"franka_description-{tag}.tar.gz"
        if not args.skip_download:
            _download_archive(_download_url_for_tag(tag), archive_path, force=args.force_redownload)
        elif not archive_path.is_file():
            raise FileNotFoundError(f"Archive not found and --skip-download was set: {archive_path}")
        extracted_dir = _extract_archive(archive_path, args.download_root, force=args.force_reextract)
        sources.append((f"franka_description_tag_{tag}", tag, extracted_dir, "upstream_tag"))
    if args.include_local_vendor:
        sources.append(("local_vendor", None, LOCAL_VENDOR_SOURCE, "local_vendor"))
    return sources


def main() -> None:
    args = _parse_args()
    robot_types = tuple(
        dict.fromkeys(_canonical_robot_type(value) for value in (args.robot_types or DEFAULT_ROBOT_TYPES))
    )

    app_launcher = None
    simulation_app = None
    if not args.skip_usd:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=True)
        simulation_app = app_launcher.app

    try:
        sources = _source_roots_from_args(args)
        output_root = args.output_root.resolve()
        urdf_root = output_root / "urdf"
        usd_root = output_root / "usd"

        variants: list[PreparedVariant] = []
        for source_label, source_tag, source_root, source_kind in sources:
            for robot_type in robot_types:
                print(
                    f"[INFO] Preparing variant source_label={source_label} source_tag={source_tag} robot_type={robot_type}",
                    flush=True,
                )
                urdf_path = urdf_root / source_label / f"{robot_type}.urdf"
                urdf_path = _expand_xacro_to_urdf(
                    source_root=source_root,
                    robot_type=robot_type,
                    urdf_path=urdf_path,
                    force=args.force_regenerate_urdf,
                )
                usd_path: Path | None = None
                if not args.skip_usd:
                    usd_path = usd_root / source_label / f"{robot_type}.usd"
                    usd_path = _run_usd_conversion(
                        urdf_path=urdf_path,
                        usd_path=usd_path,
                        force=args.force_usd_conversion,
                        fix_base=args.fix_base,
                        merge_fixed_joints=args.merge_fixed_joints,
                        self_collision=args.self_collision,
                        joint_stiffness=args.joint_stiffness,
                        joint_damping=args.joint_damping,
                    )

                variants.append(
                    PreparedVariant(
                        source_label=source_label,
                        source_root=str(source_root),
                        source_kind=source_kind,
                        source_tag=source_tag,
                        robot_type=robot_type,
                        urdf_path=str(urdf_path),
                        usd_path=None if usd_path is None else str(usd_path),
                        metadata={
                            "fix_base": args.fix_base,
                            "merge_fixed_joints": args.merge_fixed_joints,
                            "self_collision": args.self_collision,
                            "joint_stiffness": args.joint_stiffness,
                            "joint_damping": args.joint_damping,
                        },
                    )
                )
                print(
                    f"[INFO] Prepared variant source_label={source_label} robot_type={robot_type} "
                    f"urdf_path={urdf_path} usd_path={usd_path}",
                    flush=True,
                )

        manifest_path = (args.manifest_path or (output_root / "manifest.json")).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "tags": [value for value in args.tags or DEFAULT_TAGS],
            "robot_types": list(robot_types),
            "variants": [asdict(variant) for variant in variants],
        }
        print(f"[INFO] Writing manifest to {manifest_path}", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote asset manifest to {manifest_path}", flush=True)
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
