"""Connection state mixin — shared by XArm7, XHand, and SimRobotInterface.

Provides default ``connected_flag`` / ``error_state`` / ``last_error_message`` /
``last_action_code`` attributes and default ``is_connected()`` / ``is_error()`` /
``clear_error()`` implementations. Hardware drivers override the methods to
include device-specific health checks.
"""

from __future__ import annotations

from typing import Any


class ConnectionStateMixin:
    """Mixin that provides common connection-state bookkeeping.

    Subclasses must call ``super().__init__()`` (or set the attributes
    manually) and may override the ``is_connected`` / ``is_error`` methods
    to add hardware-specific checks.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.connected_flag: bool = False
        self.error_state: bool = False
        self.last_error_message: str = ""
        self.last_action_code: int | None = None

    def is_connected(self) -> bool:
        """Return True if the device is connected and not in an error state.

        Override in subclasses to add hardware-specific checks
        (e.g. ``self.arm is not None``).
        """
        return self.connected_flag and not self.error_state

    def is_error(self) -> bool:
        """Return True if the device should NOT receive commands.

        Returns True when the device is not connected (``connected_flag=False``)
        OR when an explicit error has been flagged (``error_state=True``).
        This inclusive check is intentional: the safety gate must block commands
        both before connection and after a hardware fault.

        Override in subclasses to add hardware-specific checks
        (e.g. ``self.arm.error_code != 0``).
        """
        return not self.connected_flag or self.error_state

    def clear_error(self) -> bool:
        """Clear the error state. Returns current connection status.

        Override in subclasses to also clear hardware-level errors.
        """
        self.error_state = False
        self.last_error_message = ""
        return self.is_connected()
