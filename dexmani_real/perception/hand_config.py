"""Fixed hand-surface sampling parameters, without mesh or kinematics imports."""

from dataclasses import dataclass

HAND_CLOUD_SAMPLING = "fixed_per_link_area_triangle"
HAND_CLOUD_LINK_SELECTION = "visual_area_gte_0.0001_m2"
HAND_CLOUD_COLOR = "urdf_material_rgb_or_fixed_gray_0.5"


@dataclass(frozen=True)
class HandPointCloudConfig:
    num_points: int = 1024
    seed: int = 0

    def validate(self) -> None:
        if type(self.num_points) is not int or self.num_points <= 0:
            raise ValueError("hand point count must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("hand point seed must be a nonnegative integer")


def resolve_saved_hand_pointcloud_recipe(recipe) -> HandPointCloudConfig:
    """Restore numerical sampling parameters without guessing missing training values."""
    expected = {
        "sampling_method": HAND_CLOUD_SAMPLING,
        "link_selection": HAND_CLOUD_LINK_SELECTION,
        "color_rule": HAND_CLOUD_COLOR,
    }
    if not isinstance(recipe, dict) or recipe.keys() != expected.keys() | {"num_points", "seed"}:
        raise ValueError("saved hand point-cloud recipe is incomplete or has unknown fields")
    if any(recipe[key] != value for key, value in expected.items()):
        raise ValueError("unsupported hand point-cloud numerical recipe")
    config = HandPointCloudConfig(num_points=recipe["num_points"], seed=recipe["seed"])
    config.validate()
    return config
