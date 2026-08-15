"""Model-repository integrations (execution doc §86).

``dexmani_real/deployment`` must never import from this package; the dependency
direction is integration -> deployment. Each integration encapsulates one model
repository's specific API behind the three deployment Protocols.
"""
