# PDZ Gripper Slim geometry statistics

## Source and conventions

- Source: `PDZ_Gripper_Slim(3).STEP`
- Source SHA-256:
  `ff72674d78b82c88f2543203720bed56acb4632fc9452189f9a2b72e82a46d5e`
- STEP solids: 29
- Pose: fully open CAD configuration
- Coordinate system: SolidWorks `Flange Mounting Point`
- Dimensions below are axis-aligned in the flange frame, not an oriented
  minimum bounding box.

## Complete assembly

The complete assembly includes the D405, its redesigned bracket, four
heatsinks, robot connector and pucks, 8 mm TPU pads, and timing belt.

| Quantity | Value |
|---|---:|
| X bounds | -84.000000 to 75.050000 mm |
| Y bounds | -82.935059 to 33.964466 mm |
| Z bounds | -5.000000 to 150.500080 mm |
| Overall X dimension | 159.050000 mm |
| Overall Y dimension | 116.899525 mm |
| Overall Z dimension | 155.500080 mm |
| Axis-aligned bounding-box volume | 2,891,192.687 mm^3 (2.891193 L) |
| Convex-hull volume | 1,331,297.892 mm^3 (1.331298 L) |
| Convex-hull surface area | 66,539.211 mm^2 |
| Sum of CAD solid volumes | 374,461.779 mm^3 (374.462 cm^3) |

Removing only the D405 does not change the outer bounds because the camera is
contained inside the envelope established by the heatsinks and mounting
hardware. Without the D405, the convex-hull volume remains 1,331,297.892 mm^3
and the summed solid volume is 335,719.522 mm^3.

## Gripper mechanism envelope

For a mechanism-only comparison, excluding the D405, camera bracket, camera
heatsinks, and robot connector and pucks but retaining the belt, simplified
motor, motor heatsinks, fingers, and tensioner:

| Quantity | Value |
|---|---:|
| Overall dimensions X x Y x Z | 159.050000 x 54.000012 x 145.500080 mm |
| Convex-hull volume | 788,350.197 mm^3 (0.788350 L) |
| Sum of CAD solid volumes | 243,970.083 mm^3 (243.970 cm^3) |

## Method and interpretation

CAD solid volumes are summed from the STEP B-rep solids. Coincident or
overlapping solids, if present, can therefore be counted more than once.

The convex hull was calculated from a tessellation of every included solid
using a 0.1 mm linear tolerance and 0.05 rad angular tolerance, followed by a
3D Qhull calculation. The complete hull used 381,999 input vertices and 569
hull vertices. The hull includes all empty space bridged between external
features; it is an envelope volume, not material volume and not displaced
fluid volume.

## Revision comparison

The same flange-aligned, fully-open, 0.1 mm/0.05 rad calculation was applied to
the CAD exports. These as-exported assemblies contain different component
sets, so this table describes each delivered STEP rather than a strict
part-for-part comparison.

| Revision | Solids | X x Y x Z (mm) | Bounding box (L) | Convex hull (L) | Summed solids (cm^3) |
|---|---:|---:|---:|---:|---:|
| `PDZ_Gripper(3).STEP` | 28 | 165.000 x 122.248 x 151.000 | 3.045806 | 1.422958 | 371.938 |
| `PDZ_Gripper_v2.STEP` | 34 | 187.651 x 151.886 x 145.500 | 4.146967 | 1.825596 | 444.772 |
| `PDZ_Gripper_Slim(1).STEP` | 35 | 159.158 x 151.886 x 155.500 | 3.759030 | 1.620094 | 350.504 |
| `PDZ_Gripper_Slim(2).STEP` | 25 | 159.050 x 116.110 x 155.500 | 2.871673 | 1.279269 | 370.472 |
| `PDZ_Gripper_Slim(3).STEP` | 29 | 159.050 x 116.900 x 155.500 | 2.891193 | 1.331298 | 374.462 |

Relative to Slim(2), the complete Slim(3) export is:

- unchanged in X and Z;
- 0.68% larger in Y and axis-aligned bounding-box volume;
- 4.07% larger in convex-hull volume; and
- 1.08% larger in summed CAD solid volume.

The camera heatsinks establish the slightly larger negative-Y extreme and
expand the complete convex hull. Summed solid volume is only a geometry metric;
it does not account for material density or provide a mass estimate.

Apart from the four heatsinks, Slim(3) also contains small solid-geometry
revisions to the tensioner bracket and camera bracket. Their poses and
axis-aligned outer dimensions are unchanged.

### Normalized mechanism comparison

For a closer mechanism comparison, the camera, camera-specific mounting
hardware, and robot connector hardware were removed. The belt is retained.

| Revision | X x Y x Z (mm) | Convex hull (L) | Summed solids (cm^3) |
|---|---:|---:|---:|
| v2 mechanism | 187.651 x 60.000 x 136.900 | 0.984273 | 313.784 |
| Slim(1) mechanism | 159.158 x 54.000 x 146.950 | 0.775603 | 219.022 |
| Slim(2) mechanism | 159.050 x 54.000 x 145.500 | 0.777690 | 242.200 |
| Slim(3) mechanism | 159.050 x 54.000 x 145.500 | 0.788350 | 243.970 |

On this normalized basis, Slim(3) has identical axis-aligned dimensions to
Slim(2), while the two motor heatsinks and small tensioner revision increase
convex-hull volume by 1.37% and summed solid volume by 0.73%.

A matching normalized first-generation mechanism cannot be extracted reliably
because its robot-side structure and main base are fused into one STEP solid.
