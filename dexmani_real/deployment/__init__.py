"""Learned-policy deployment runtime.

The synchronous policy runner owns the model and action queue, reusing the
robot runtime, safety, IPC, and lifecycle machinery. Model output is a proposal
until it passes the existing command preparation and publication boundaries.
"""
