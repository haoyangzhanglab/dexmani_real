"""Motor trajectory interpolator for XHand joint-space trajectories.

Ref: DexUMI dexumi/real_env/common/motor_trajectory_interpolator.py
"""

from __future__ import annotations

import numbers
from typing import Union

import numpy as np
import scipy.interpolate as si


class MotorTrajectoryInterpolator:
    """Linear trajectory interpolator for hand motor position waypoints.

    Wraps scipy.interpolate.interp1d for smooth linear interpolation along
    the time axis. Supports single-waypoint (hold) and multi-waypoint modes,
    trajectory trimming, and speed-limited waypoint scheduling.

    Ref: DexUMI motor_trajectory_interpolator.py:8-210

    Usage:
        # Multi-waypoint interpolation
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([[0]*12, [1]*12, [2]*12])  # (3, 12)
        interp = MotorTrajectoryInterpolator(times, values)
        pos_at_1_5 = interp(1.5)  # interpolated position at t=1.5

        # Speed-limited waypoint driving
        interp2 = interp.drive_to_waypoint(target, arrival_time, curr_time, max_speed=3.0)

        # Trim to time range
        trimmed = interp.trim(t_start, t_end)
    """

    def __init__(self, times: np.ndarray, values: np.ndarray) -> None:
        """Initialize the motor trajectory interpolator.

        Args:
            times: Array of timestamps for waypoints.
            values: Array of n-dimensional motor values corresponding to timestamps,
                    shape (N, D) where N = len(times).
        """
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        if len(times) < 1:
            raise ValueError("Must provide at least one waypoint")
        if len(values) != len(times):
            raise ValueError(f"Number of values ({len(values)}) must match number of timestamps ({len(times)})")

        if len(times) == 1:
            # Special handling for single waypoint — hold position
            self._single_step = True
            self._times = times
            self._values = values
        else:
            if not np.all(times[1:] >= times[:-1]):
                raise ValueError("Times must be monotonically increasing")
            self._single_step = False
            # Create linear interpolator for all dimensions (axis=0).
            # bounds_error=False + fill_value="extrapolate" allows queries
            # outside [t_start, t_end]; __call__ clamps t to match DexUMI behavior.
            self._interp = si.interp1d(
                times,
                values,
                kind="linear",
                axis=0,
                bounds_error=False,
                fill_value="extrapolate",
            )

    @property
    def times(self) -> np.ndarray:
        """Get interpolation timestamps."""
        if self._single_step:
            return self._times
        return self._interp.x

    @property
    def values(self) -> np.ndarray:
        """Get interpolation values."""
        if self._single_step:
            return self._values
        return self._interp.y

    # ── Trajectory manipulation ──

    def trim(self, start_t: float, end_t: float) -> "MotorTrajectoryInterpolator":
        """Create new interpolator trimmed to [start_t, end_t].

        Waypoints inside the range are preserved; boundary values are
        interpolated at start_t and end_t.

        Args:
            start_t: Start time for trimmed trajectory.
            end_t: End time for trimmed trajectory.

        Returns:
            New MotorTrajectoryInterpolator covering [start_t, end_t].
        """
        if start_t > end_t:
            raise ValueError(f"start_t ({start_t}) must be <= end_t ({end_t})")

        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]

        # Include boundary points
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        all_times = np.unique(all_times)  # remove duplicates

        all_values = self(all_times)
        return MotorTrajectoryInterpolator(times=all_times, values=all_values)

    def drive_to_waypoint(
        self,
        value: np.ndarray,
        time: float,
        curr_time: float,
        max_speed: float = np.inf,
    ) -> "MotorTrajectoryInterpolator":
        """Create new interpolator that drives from current position to a waypoint.

        If the required speed exceeds max_speed, the arrival time is
        automatically extended to respect the speed limit.

        Args:
            value: Target motor values, shape (D,).
            time: Desired arrival time.
            curr_time: Current time (used to compute current position).
            max_speed: Maximum allowed scalar speed (L2 norm distance / time).

        Returns:
            New MotorTrajectoryInterpolator with the waypoint appended.
        """
        if max_speed <= 0:
            raise ValueError("Speed limit must be positive")

        time = max(time, curr_time)
        curr_value = self(curr_time)
        value_dist = float(np.linalg.norm(np.asarray(value) - np.asarray(curr_value)))
        min_duration = value_dist / max_speed
        duration = max(time - curr_time, min_duration)
        last_waypoint_time = curr_time + duration

        # Trim to current position and append new waypoint
        trimmed = self.trim(curr_time, curr_time)
        times = np.append(trimmed.times, [last_waypoint_time], axis=0)
        values = np.append(trimmed.values, np.atleast_2d(value), axis=0)

        return MotorTrajectoryInterpolator(times, values)

    def schedule_waypoint(
        self,
        value: np.ndarray,
        time: float,
        max_speed: float = np.inf,
        curr_time: float | None = None,
        last_waypoint_time: float | None = None,
    ) -> "MotorTrajectoryInterpolator":
        """Schedule a new waypoint while respecting speed limits and timing constraints.

        Handles the case where waypoints arrive out of order or with overlapping
        times — truncates existing trajectory as needed and appends the new
        waypoint with speed-limited duration.

        Args:
            value: Target motor values, shape (D,).
            time: Desired arrival time.
            max_speed: Maximum allowed scalar speed.
            curr_time: Current time (optional, enables time-based truncation).
            last_waypoint_time: Time of last scheduled waypoint (optional,
                requires curr_time; used to enforce ordering).

        Returns:
            New MotorTrajectoryInterpolator with the waypoint scheduled.
        """
        if max_speed <= 0:
            raise ValueError("Speed limit must be positive")
        if last_waypoint_time is not None and curr_time is None:
            raise ValueError("curr_time is required when last_waypoint_time is provided")

        start_time = self.times[0]
        end_time = self.times[-1]

        if curr_time is not None:
            if time <= curr_time:
                # Waypoint is in the past — keep current trajectory
                return self
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)

        # Trim trajectory to relevant range
        trimmed = self.trim(start_time, end_time)

        # Compute speed-limited duration to the new waypoint
        duration = time - end_time
        end_value = trimmed(end_time)
        value_dist = float(np.linalg.norm(np.asarray(value) - np.asarray(end_value)))
        min_duration = value_dist / max_speed
        duration = max(duration, min_duration)
        last_waypoint_arrival = end_time + duration

        times = np.append(trimmed.times, [last_waypoint_arrival], axis=0)
        values = np.append(trimmed.values, np.atleast_2d(value), axis=0)

        return MotorTrajectoryInterpolator(times, values)

    # ── Interpolation call ──

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        """Interpolate motor values at time t.

        Args:
            t: Scalar time or array of times.

        Returns:
            Interpolated motor values at t. For scalar input, returns (D,).
            For array input, returns (len(t), D).
        """
        is_single = isinstance(t, numbers.Number)
        if is_single:
            t_arr = np.array([t])
        else:
            t_arr = np.asarray(t)

        if self._single_step:
            values = np.tile(self._values[0], (len(t_arr), 1))
        else:
            # Clamp to trajectory bounds (ref: DexUMI motor_trajectory_interpolator.py:203-205)
            t_clipped = np.clip(t_arr, self.times[0], self.times[-1])
            values = self._interp(t_clipped)

        if is_single:
            return values[0]
        return values
