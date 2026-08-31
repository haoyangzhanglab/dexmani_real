"""Learned-policy deployment runtime.

Model contracts, the inference worker, and the coordinator reuse the robot
runtime, safety, IPC, and lifecycle machinery. The model
output is a proposal only; the coordinator is the sole robot-action producer.
"""
