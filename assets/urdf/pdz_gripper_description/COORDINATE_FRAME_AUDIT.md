# PDZ gripper coordinate-frame audit

## Canonical flange convention

The URDF and corrected STEP exports consistently use:

- +X: finger opening direction, toward the positive-X finger and motor side;
- +Y: transverse direction across the guide rods; the D405 is on negative Y;
- +Z: outward from the robot flange toward the fingertips.

This convention is directly visible in every corrected export: finger centres
remain at X = +/-44 mm in the fully-open pose, guide rods remain at +/-Y, the
stepper remains near +X, and the fingertip height increases along +Z.

## Correct flange-coordinate exports

These files use `Flange Mounting Point` and agree on the axis definitions:

- `PDZ_Gripper(2).STEP`
- `PDZ_Gripper(3).STEP`
- `PDZ_Gripper_D405.STEP`
- `PDZ_Gripper_v2.STEP`
- `PDZ_Gripper_Slim(1).STEP`
- `PDZ_Gripper_Slim(2).STEP`
- `PDZ_Gripper_Slim(3).STEP`

The current URDF is generated from `PDZ_Gripper_Slim(3).STEP`.

## Exports using the assembly root instead

These files do not use the flange output coordinate system:

- `PDZ_Gripper.STEP`
- `PDZ_Gripper(1).STEP`
- `PDZ_Gripper_Slim.STEP`

This is not an X/Y swap. Their root orientation is rotated -90 degrees about X
relative to the flange frame:

```text
R_root_from_flange = Rx(-90 deg)

                       [ 1  0  0 ]
                     = [ 0  0  1 ]
                       [ 0 -1  0 ]
```

Therefore:

```text
root +X = flange +X
root +Y = flange +Z
root +Z = flange -Y
```

For the first design family, the complete point transform from the corrected
flange frame to the uncorrected assembly root is:

```text
p_root = Rx(-90 deg) * p_flange
       + [42.697318329, 21.184390913, 104.704867802] mm
```

For the uncorrected Slim export it is:

```text
p_root = Rx(-90 deg) * p_flange
       + [42.697318329, 26.184390913, 107.704867802] mm
```

The translation changed with the mechanical revision, but the orientation
mistake is the same.

## Evidence used

The frame relationship was solved independently from unchanged or traceable
components and then checked against the rest of each assembly:

- base orientation;
- paired finger transforms;
- paired guide-rod positions;
- stepper pose;
- KUKA connector pose where present; and
- D405 pose where present.

All corrected exports preserve X as the opening axis. Changes such as guide
rods moving from Y = +/-19 mm to +/-17 mm and fingers becoming narrower are
mechanical redesigns, not coordinate-axis changes.
