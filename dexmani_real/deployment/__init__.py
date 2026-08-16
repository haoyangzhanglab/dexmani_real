"""Learned-policy deployment runtime.

Backend-neutral contracts, the inference worker, and the coordinator that reuse
the A/B-frozen robot runtime, safety, IPC, and lifecycle machinery. The model
output is a proposal only; the coordinator is the sole robot-action producer.
"""
