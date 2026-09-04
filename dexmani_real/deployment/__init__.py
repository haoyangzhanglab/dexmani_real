"""Learned-policy deployment runtime.

Model contracts, the inference worker, and the policy executor reuse the robot
runtime, safety, IPC, and lifecycle machinery. The model output is a proposal
only; the policy executor is the sole learned-policy action producer.
"""
