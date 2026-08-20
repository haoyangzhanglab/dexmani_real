# DexMani Real Code Style Guide

This document defines the preferred coding style for DexMani Real.

The goal is not to make the repository maximally abstract or maximally extensible. The goal is to make research code **easy to read, easy to modify, easy to debug, and difficult to misuse**.

The guiding principle is:

> Prefer explicit data flow and simple ownership over framework-like abstraction.

---

## 1. Core Principles

### 1.1 Optimize for the next researcher

Code should make it easy to answer:

1. Where does the data come from?
2. What transforms it?
3. Who owns the state?
4. Where does the side effect happen?
5. What happens when something fails?

A reader should not need to jump through many wrappers, factories, managers, registries, or abstract interfaces to answer these questions.

### 1.2 Prefer simple code over general code

Do not design for hypothetical future requirements.

Prefer:

```text
specific implementation
→ repeated pattern
→ small reusable abstraction
```

over:

```text
generic framework
→ interface
→ adapter
→ factory
→ one actual implementation
```

An abstraction should exist because it removes real duplication or establishes a meaningful boundary.

### 1.3 Keep one source of truth

A concept should have one authoritative definition.

Typical examples:

- configuration defaults
- shared-memory schema
- dataset schema
- coordinate-frame convention
- safety validation
- device ownership
- serialization format

Do not duplicate the same information into multiple modules and keep them synchronized manually.

### 1.4 Architecture should follow data flow

For robotics code, the most useful decomposition is usually:

```text
input
→ representation
→ algorithm
→ validation
→ output / side effect
```

The directory and module structure should help reveal this flow rather than hide it behind infrastructure abstractions.

### 1.5 Thin outside, explicit inside

Entry points, CLIs and integration adapters should be thin.

Core domain modules should contain the actual behavior.

```text
CLI
 └─ parse arguments
 └─ build configuration
 └─ call domain API

domain module
 └─ actual control / planning / recording logic
```

Do not implement substantial algorithms directly inside `examples/` or CLI files.

---

# 2. File-Level Readability

File readability is a first-class requirement.

A Python file should normally have **one primary responsibility**.

A useful default layout is:

```python
"""Short module-level description, when useful."""

from __future__ import annotations

# standard library
import ...

# third-party
import numpy as np

# project
from dexmani_real... import ...

logger = ...

# constants

# dataclasses / lightweight types

# small pure helpers

# primary class or public functions

# private implementation helpers
```

Not every file needs every section.

The important property is that a reader can understand the file **top-to-bottom**.

---

## 2.1 Split by responsibility, not by line count

There is no strict maximum file length.

However, a large file should trigger inspection when it contains several independent responsibilities.

Bad reason to split:

> This file reached 300 lines.

Good reason to split:

> Camera conversion, visualization, networking and calibration are four different responsibilities.

Similarly, do not fragment a coherent 150-line algorithm into six tiny modules merely to reduce file size.

---

## 2.2 Keep abstraction depth shallow

Prefer call paths such as:

```text
run()
→ get_observation()
→ compute_action()
→ validate_action()
→ send_action()
```

Avoid paths such as:

```text
runner
→ manager
→ controller
→ service
→ adapter
→ backend
→ implementation
```

unless those layers correspond to genuinely different responsibilities.

A good research codebase usually has **few conceptual layers with clear ownership**.

---

# 3. Naming

Names should encode semantics rather than implementation mechanics.

## 3.1 Python conventions

Use:

- `snake_case` for variables, functions and modules
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for true constants
- leading `_` for implementation details

Functions should normally be verbs:

```python
compute_fk()
build_point_cloud()
validate_action()
read_episode()
send_command()
```

Data should normally be nouns:

```python
joint_positions
camera_pose
episode
command
observation
```

Avoid vague names such as:

```text
data
info
obj
handler
manager
processor
result
tmp
```

when a more precise domain name is available.

---

## 3.2 Encode units in names

Physical quantities should make units obvious whenever ambiguity is possible.

Prefer:

```python
timeout_s
frequency_hz
position_m
angle_rad
velocity_rad_s
```

over:

```python
timeout
frequency
position
angle
velocity
```

Unsuffixed angles should only be used when the local convention is completely unambiguous.

---

## 3.3 Encode coordinate frames

Frame semantics should be visible in names.

Examples:

```python
T_base_camera
T_world_object
ee_pose_base
points_camera
points_world
```

For transforms, use a single project-wide convention and document it once.

Do not use generic names such as:

```python
transform
pose
matrix
```

when multiple frames are involved.

---

# 4. Imports

Imports should make dependencies easy to inspect.

Use three groups:

```python
# standard library
import time
from pathlib import Path

# third-party
import numpy as np
import torch

# project
from dexmani_real.robot import ...
```

Rules:

- prefer absolute imports across major project modules
- avoid `from module import *`
- use conventional aliases such as `np`
- avoid aliases that hide the original dependency
- keep optional heavyweight dependencies local only when there is a concrete reason
- do not use local imports as a routine solution to circular dependencies

Circular imports usually indicate an architectural problem.

Fix the dependency direction instead of hiding the cycle.

---

# 5. Functions

Functions are the main unit of readability.

## 5.1 One function, one semantic level

A function should generally operate at one level of abstraction.

Bad:

```python
def run():
    # parse device packet
    # compute FK
    # perform collision checking
    # write HDF5
    # render UI
```

Better:

```python
def run():
    observation = read_observation()
    action = compute_action(observation)
    action = safety_gate(action)
    publish_action(action)
```

Implementation details belong in the relevant functions.

---

## 5.2 Prefer pure functions for mathematics

Geometry, transformations, filtering, trajectory calculations and validation should be pure whenever practical.

Prefer:

```python
q_target = compute_target(q_current, command)
```

over:

```python
controller.update_internal_target(command)
q_target = controller.target
```

when persistent state is not actually necessary.

Pure functions are easier to:

- understand
- test
- reuse
- benchmark
- debug

---

## 5.3 Avoid boolean-driven mega-functions

Avoid APIs such as:

```python
process(
    use_filter=True,
    save=True,
    visualize=False,
    replay=False,
    debug=True,
)
```

Several mode flags usually indicate several responsibilities.

Prefer separate functions or a small explicit configuration object.

---

## 5.4 Use early returns

Prefer:

```python
if observation is None:
    return None

if not observation.valid:
    return None

return compute_action(observation)
```

over deeply nested conditional blocks.

---

## 5.5 Keep side effects explicit

Functions performing important side effects should make them visible in their names.

Good:

```python
save_episode()
send_robot_command()
start_camera()
publish_action()
```

Less clear:

```python
process()
update()
handle()
do()
```

---

# 6. Classes

Use a class when there is meaningful **state, ownership or lifecycle**.

Good class candidates:

- robot/device connections
- shared-memory resources
- stateful controllers
- recorder lifecycle
- inference runtime
- long-lived workers

Do not create a class merely to group unrelated functions.

Prefer:

```python
def transform_points(...):
    ...
```

over:

```python
class GeometryUtils:
    @staticmethod
    def transform_points(...):
        ...
```

---

## 6.1 Constructors should be cheap

`__init__` should primarily establish object state.

Avoid surprising operations such as:

- starting processes
- connecting to hardware
- opening cameras
- starting threads
- entering control mode

Prefer explicit lifecycle methods:

```python
camera = Camera(config)
camera.start()

...

camera.stop()
```

This makes resource ownership visible.

---

## 6.2 Recommended method order

Within a class:

```text
__init__

public lifecycle methods
start()
stop()
close()

public domain operations

small public queries

private helpers
```

Do not organize methods chronologically according to when they were added.

---

## 6.3 Avoid pass-through wrappers

This usually adds little value:

```python
class RobotManager:
    def move(self, command):
        return self.robot.move(command)
```

A wrapper should introduce a real semantic boundary such as:

- safety validation
- protocol translation
- resource ownership
- synchronization
- lifecycle management

Otherwise call the underlying object directly.

---

# 7. Dataclasses and Data Structures

Use dataclasses for small structured domain data and configuration.

Prefer explicit data:

```python
@dataclass
class CameraObservation:
    rgb: np.ndarray
    depth: np.ndarray
    timestamp_s: float
```

over dictionaries with implicit fields:

```python
obs["rgb"]
obs["depth"]
obs["timestamp"]
```

For cross-process or storage boundaries, use the representation required by that boundary, but keep its schema centralized.

Do not introduce large nested object graphs into multiprocessing communication.

---

# 8. Comments and Docstrings

Comments should explain information the code itself cannot express.

Good comments explain:

- why an unusual decision exists
- coordinate-frame convention
- units
- safety constraints
- timing assumptions
- concurrency invariants
- hardware-specific behavior
- non-obvious mathematical reasoning

Bad:

```python
# Increment counter
counter += 1
```

Good:

```python
# Use monotonic time because wall-clock adjustments must not trigger a
# false worker timeout.
elapsed_s = time.monotonic() - last_update_s
```

---

## 8.1 Do not narrate the code

Avoid excessive comments such as:

```python
# Create camera
camera = Camera()

# Start camera
camera.start()

# Read image
image = camera.read()
```

Well-named code should carry this information.

---

## 8.2 Docstrings describe contracts

Public APIs should document non-obvious contracts:

- input semantics
- shape
- dtype
- unit
- frame
- return semantics
- important failure behavior

Example:

```python
def transform_points(
    points_camera: np.ndarray,
    T_world_camera: np.ndarray,
) -> np.ndarray:
    """Transform Nx3 points from camera frame to world frame."""
```

Avoid multi-paragraph docstrings for trivial implementation details.

---

## 8.3 Delete dead code

Do not preserve historical implementations as commented code.

Git already stores history.

---

# 9. Configuration

Configuration should remain simple and traceable.

## 9.1 One canonical default

Each configuration value should have one canonical default.

Avoid:

```text
Python default
+ CLI default
+ YAML default
+ fallback default
+ magic default in worker
```

Defaults should be defined once and overridden explicitly.

---

## 9.2 Pass the configuration a component owns

Avoid passing one giant global configuration object everywhere.

Prefer:

```python
camera = Camera(camera_config)
planner = Planner(planning_config)
```

over:

```python
camera = Camera(config)
planner = Planner(config)
```

This makes dependencies explicit.

---

## 9.3 Derived values belong near their owner

If:

```python
period_s = 1.0 / control_hz
```

the component that owns the control loop should normally derive `period_s`.

Do not store every trivial derived quantity as another independent configuration option.

---

# 10. Scripts and Entry Points

Files under `examples/`, `scripts/`, or equivalent entry-point directories should be intentionally boring.

A typical script should contain:

```python
def parse_args():
    ...

def main():
    args = parse_args()
    config = ...
    run(config)

if __name__ == "__main__":
    main()
```

Scripts may perform:

1. argument parsing
2. configuration construction
3. dependency construction
4. lifecycle invocation
5. user-facing output

They should not contain substantial:

- geometry
- control logic
- recording logic
- safety logic
- device protocols

Move reusable behavior into the package.

---

# 11. Error Handling and Logging

Fail at meaningful boundaries.

Examples:

- malformed external input
- incorrect array shape
- NaN/Inf entering a controller
- hardware communication failure
- corrupted episode
- impossible state transition

Do not repeatedly validate trusted values inside every internal helper.

---

## 11.1 Assertions versus exceptions

Use assertions for programmer invariants:

```python
assert len(items) == len(weights)
```

Use explicit exceptions for runtime/user/external input:

```python
if points.shape[-1] != 3:
    raise ValueError(...)
```

---

## 11.2 Avoid broad exception swallowing

Avoid:

```python
try:
    ...
except Exception:
    pass
```

At process or hardware boundaries, a broad catch may be necessary, but preserve context and report the failure.

---

## 11.3 Logging should describe events

Prefer:

```python
logger.info("Camera connected: serial=%s", serial)
logger.warning("Skipping stale observation: age=%.3fs", age_s)
```

Avoid high-frequency logs inside control loops unless rate-limited or explicitly diagnostic.

---

# 12. Robotics and Concurrency

Real-robot software needs stricter ownership rules than ordinary research scripts.

## 12.1 One clear owner for each hardware resource

A hardware connection should have one obvious owner.

Do not pass live SDK objects across multiprocessing boundaries.

Communicate through explicit data structures instead.

---

## 12.2 No important import-time side effects

Importing a module should not:

- connect to a robot
- open a camera
- start a process
- start a thread
- modify robot state

Hardware behavior should begin through an explicit operation.

---

## 12.3 Keep real-time loops narrow

A control loop should primarily do:

```text
read
→ compute
→ validate
→ publish
→ rate control
```

Move unrelated work out of the loop:

- filesystem writes
- visualization
- expensive logging
- blocking network requests
- UI interaction

---

## 12.4 Validate at system boundaries

Important validation belongs where data crosses a boundary:

```text
device → software
process → process
model → robot command
disk → runtime
user input → controller
```

Avoid scattering the same validation throughout the entire call graph.

---

# 13. Directory Structure

Organize code by stable domain responsibility.

A useful conceptual structure is:

```text
package/
├── config/
├── sensor/
├── robot/
├── teleop/
├── planning/
├── policy/
├── recording/
├── deployment/
└── utils/

examples/
docs/
```

This is a guideline, not a permanent architecture contract.

Create a new top-level package only when it represents a stable domain concept.

---

## 13.1 Be careful with `utils/`

`utils/` should not become a dumping ground.

If a helper clearly belongs to:

- geometry
- recording
- planning
- robot communication
- visualization

put it in that domain.

Generic utilities should be genuinely generic.

---

# 14. Avoid Overengineering

The following patterns require a concrete justification before introduction:

- `Base*` classes with only one implementation
- factories with only one product
- registries for a fixed set of internal components
- manager → controller → service chains
- one-line adapter layers
- generic event systems for simple direct communication
- dependency injection frameworks
- complex configuration inheritance
- duplicated protocol/schema classes
- speculative plugin systems

Before introducing abstraction, ask:

> What concrete complexity does this remove today?

If there is no clear answer, keep the implementation explicit.

---

# 15. Refactoring Existing Code

When cleaning an overgrown module, prefer this order:

### Step 1 — reveal the data flow

Rename unclear variables and functions.

### Step 2 — separate side effects from computation

Extract pure geometry/control/data helpers.

### Step 3 — remove duplicate logic

Choose one canonical implementation.

### Step 4 — simplify ownership

Remove unnecessary managers and forwarding wrappers.

### Step 5 — split genuine responsibilities

Only now split files or modules.

Do not begin by creating a new framework around already complicated code.

---

# 16. AI-Assisted Coding Rules

When using Claude Code, Codex, or another coding agent:

1. Read the relevant call path before editing.
2. Prefer the smallest coherent change.
3. Do not refactor unrelated code.
4. Do not introduce an abstraction merely because it appears cleaner in isolation.
5. Reuse existing domain concepts before inventing new ones.
6. Preserve public behavior unless change is requested.
7. Keep hardware operations explicit.
8. Verify the focused change offline whenever possible.
9. Inspect the final diff for accidental complexity.
10. Report validation that was not performed.

---

# 17. Review Checklist

Before accepting new code, ask:

### Readability

- Can I understand the main data flow quickly?
- Are names domain-specific?
- Does each function operate at one semantic level?
- Does this file have one coherent responsibility?

### Architecture

- Is ownership obvious?
- Did we introduce another source of truth?
- Did we add a wrapper without adding semantics?
- Is the dependency direction simple?

### Robotics

- Are units and coordinate frames clear?
- Are side effects explicit?
- Are hardware resources clearly owned?
- Is validation performed at the appropriate boundary?
- Can failures leave the system in an unsafe state?

### Research maintainability

- Can a new student modify this six months later?
- Can the important computation be tested without hardware?
- Is configuration traceable?
- Is there a simpler implementation with the same behavior?

If the simpler implementation is equally correct, prefer the simpler implementation.