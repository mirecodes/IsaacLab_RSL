# Unitree A2

Source: `robots/a2_description` of [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros)
at commit `cdf67e6eb4c98d73202f29a25e3d098e2aa3b247` (BSD-3-Clause, see `LICENSE`).

| Path | Content |
| --- | --- |
| `meshes/` | Unitree's STL meshes, unchanged |
| `urdf/a2.urdf` | Unitree's URDF without the massless `camera`, `front_lidar` and `rear_lidar` frames |
| `usd/` | Generated, not tracked (the repository allows no USD files); used by `UNITREE_A2_CFG` |

## Generate the USD

Run once after checkout, inside the Isaac Lab container from `/workspace/isaaclab`:

```bash
A=source/isaaclab_assets/data/Robots/Unitree/A2
python scripts/tools/convert_urdf.py $A/urdf/a2.urdf $A/usd --joint_stiffness 0 --joint_damping 0
```

This writes `usd/a2/a2.usda` and records the converter settings in `usd/config.yaml`. The fixed foot joints are
kept (`--merge_joints` off) so the feet stay separate bodies for contact sensing; the drive gains are zero
because the actuator model sets them. The conversion has to run as a standalone script: spawning the URDF
directly (`UrdfFileCfg`) fails under Newton, which does not register the PhysX schemas the importer applies.
