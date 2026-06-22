from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sapien.core as sapien

__all__ = ["PhysxConfig", "setup_scene", "create_viewer", "add_light", "add_base_components"]


@dataclass
class PhysxConfig:
    contact_offset: float = 0.02
    solver_position_iterations: int = 8
    solver_velocity_iterations: int = 1
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0
    enable_ccd: bool = False


def setup_scene(
    time_step: float = 1 / 240,
    shader_type: str = "default",
    physx: PhysxConfig | None = None,
):
    p = physx or PhysxConfig()

    # SAPIEN render resource limits — conservative defaults to handle
    # complex scenes with multiple articulated bodies and textures.
    _MAX_MATERIALS = 50000
    _MAX_TEXTURES = 50000
    sapien.render.set_global_config(max_num_materials=_MAX_MATERIALS, max_num_textures=_MAX_TEXTURES)
    sapien.render.set_camera_shader_dir(shader_type)
    if shader_type == "rt":
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

    sapien.physx.set_shape_config(contact_offset=p.contact_offset, rest_offset=0)
    sapien.physx.set_body_config(
        solver_position_iterations=p.solver_position_iterations, solver_velocity_iterations=p.solver_velocity_iterations
    )
    sapien.physx.set_default_material(
        static_friction=p.static_friction, dynamic_friction=p.dynamic_friction, restitution=p.restitution
    )

    scene_config = sapien.physx.PhysxSceneConfig()
    scene_config.gravity = [0, 0, -9.81]
    scene_config.enable_ccd = p.enable_ccd
    scene_config.enable_pcm = True
    scene_config.enable_tgs = True
    scene_config.enable_friction_every_iteration = True
    sapien.physx.set_scene_config(scene_config)

    scene = sapien.Scene()
    scene.set_timestep(time_step)
    return scene


def create_viewer(scene: sapien.Scene, pose: sapien.Pose):
    viewer = scene.create_viewer()
    viewer.set_camera_pose(pose)
    viewer.control_window.toggle_camera_lines(show=False)
    return viewer


def add_light(
    scene: sapien.Scene,
    num_point_lights: int = 2,
    point_light_shadow: bool = False,
    directional_light_shadow: bool = False,
    shadow_map_size: int = 1024,
):
    directional = scene.add_directional_light(
        [0, 1, -1],
        [0.75, 0.75, 0.75],
        shadow=directional_light_shadow,
        shadow_scale=2.0,
        shadow_map_size=shadow_map_size,
    )
    point_light_pos = [[0, 1, 1], [0, -1, 1], [-1, 0, 1]]
    point_lights = []
    for pos in point_light_pos[:num_point_lights]:
        pl = scene.add_point_light(
            position=pos,
            color=[1, 1, 1],
            shadow=point_light_shadow,
            shadow_map_size=shadow_map_size,
        )
        point_lights.append(pl)
    return directional, point_lights


def add_base_components(scene: sapien.Scene):
    scene.set_ambient_light([0.1, 0.1, 0.1])
    scene.add_ground(altitude=-1.0, render_half_size=[2, 2])

    table_visual = [0.15, 0.30, 0.15]
    table_builder = scene.create_actor_builder()
    table_builder.add_box_visual(half_size=[0.5, 1.0, 0.5], material=table_visual)
    table_builder.add_box_collision(half_size=[0.5, 1.0, 0.5], material=sapien.physx.PhysxMaterial(0.8, 0.8, 0))
    table = table_builder.build_kinematic(name="table")
    table.set_pose(sapien.Pose(p=[0.4, 0, -0.5]))
