"""Own the local policy/robot, sensor processes and operator lifecycle."""

import multiprocessing as mp
import os

from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
from dexmani_real.deployment.config import (
    validate_max_running_s,
    validate_num_episodes,
    validate_policy_runtime_compatibility,
)
from dexmani_real.deployment.observation import build_fingertip_runtime
from dexmani_real.deployment.operator import PolicyOperator
from dexmani_real.deployment.runner import PolicyRunner
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.robot.arm_homing import build_policy_home_planner
from dexmani_real.robot.robot import DexManiRobot
from dexmani_real.runtime.processes import shutdown_local_runtime
from dexmani_real.runtime.safety import (
    RunEndReason,
    SafetyState,
    require_transition,
    revoke_motion,
)
from dexmani_real.runtime.supervisor import RuntimeSupervisor
from dexmani_real.sensor.camera.worker import run_camera_worker
from dexmani_real.sensor.pointcloud_worker import PointCloudWorkerConfig, run_pointcloud_worker
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def run_policy_deployment(
    runtime,
    policy_config,
    execute,
    *,
    prefix=None,
    max_running_s=None,
    num_episodes=1,
    recording_config=None,
):
    info = policy_config.info
    cloud_recipe = validate_policy_runtime_compatibility(info, runtime)
    max_running_s = validate_max_running_s(max_running_s)
    num_episodes = validate_num_episodes(num_episodes)
    if recording_config is not None and not execute:
        raise ValueError("recorded evaluation requires execute")
    fields = set(info.observation_fields)
    cloud = "point_cloud" in fields
    # Cloud production needs a camera worker; recording also retains its source RGB-D.
    camera = cloud or "rgb" in fields or recording_config is not None
    points = cloud_recipe.num_points if cloud else runtime.pointcloud.num_points
    camera_calibration = CameraExtrinsics() if cloud else None
    pointcloud_config = (
        PointCloudWorkerConfig.from_runtime(
            runtime,
            pointcloud=cloud_recipe,
            camera_calibration=camera_calibration,
        )
        if cloud
        else None
    )
    ctx = mp.get_context("spawn")
    shared = RuntimeChannels.create(
        prefix=prefix or f"dexmani_policy_{os.getpid()}",
        mp_context=ctx,
        config=RuntimeChannelsConfig.from_runtime(runtime, pointcloud_num_points=points),
    )
    supervisor = RuntimeSupervisor(shared, runtime.safety.readiness_timeouts_s)
    robot = DexManiRobot(shared, runtime, check_services=supervisor.check)
    model = operator = None
    failure = None
    clean = False
    try:
        from dexmani_policy.deployment import load_policy

        model = load_policy(
            policy_config.config, info, device=policy_config.device, seed=policy_config.seed
        )
        fingertip = build_fingertip_runtime(info, runtime)
        model.warmup(samples=5, rgb_hw=(runtime.camera.height, runtime.camera.width))
        robot.connect()
        sensors = []
        if camera:
            sensors.append(
                ctx.Process(name="camera", target=run_camera_worker, args=(shared, runtime.camera))
            )
        if cloud:
            sensors.append(
                ctx.Process(
                    name="pointcloud",
                    target=run_pointcloud_worker,
                    args=(shared, pointcloud_config),
                )
            )
        supervisor.start(sensors)
        require_transition(shared, SafetyState.ARMED)
        operator = PolicyOperator(
            shared,
            runtime,
            build_policy_home_planner(runtime) if execute else None,
            robot=robot,
            execute=execute,
        )
        operator.keyboard.start()
        robot.check_services = lambda: supervisor.check() and operator.keyboard.healthy
        runner = PolicyRunner(
            shared,
            runtime,
            info,
            robot=robot,
            poll_operator=operator.poll,
            model_runtime=model,
            fingertip_runtime=fingertip,
            execute=execute,
            max_running_s=max_running_s,
            num_episodes=num_episodes,
            recording_config=recording_config,
        )
        runner.run()
        clean = bool(shared.quit_requested.value)
    except KeyboardInterrupt as exc:
        shared.estop_request.value = True
        failure = exc
    except Exception as exc:
        failure = exc
        shared.error_state.value = True
        logger.exception("policy session failed")
    finally:
        revoke_motion(shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
        shutdown_clean = shutdown_local_runtime(
            robot,
            supervisor,
            model=model,
            keyboard=operator.keyboard if operator else None,
            timeout_s=runtime.safety.shutdown_timeout_s,
        )
    return int(not clean or failure is not None or not shutdown_clean)
