# ABOUTME: Per-channel dye-dose skew state for the dilution disruption.
# ABOUTME: Factor is bounded to (0, 1.0]; only channels with role == "dye" may be skewed.

from rig_service.config import RigConfig


class DisruptError(Exception):
    """Invalid skew configuration (unknown channel, non-dye channel, or out-of-bound factor)."""


class Disruptions:
    """Per-channel dye-dose skew state. Factor 1.0 == no skew; only dye channels skew."""

    def __init__(self, config: RigConfig) -> None:
        self._channels = config.channels
        self._skews: dict[str, float] = {}

    def set_skew(self, channel_name: str, factor: float) -> None:
        """DisruptError unless the channel exists with role == 'dye' and
        0 < factor <= 1.0 (both bounds checked; 1.0 accepted and clears the entry)."""
        channel = self._channels.get(channel_name)
        if channel is None:
            raise DisruptError(f"unknown channel {channel_name!r}")
        if channel.role != "dye":
            raise DisruptError(
                f"channel {channel_name!r} has role {channel.role!r}; only dye channels may be skewed"
            )
        if factor <= 0.0:
            raise DisruptError(f"factor must be > 0 (got {factor})")
        if factor > 1.0:
            raise DisruptError(f"factor must be <= 1.0 (got {factor})")
        if factor == 1.0:
            self._skews.pop(channel_name, None)
        else:
            self._skews[channel_name] = factor

    def clear(self) -> None:
        """Drop all skews."""
        self._skews.clear()

    def factor(self, channel_name: str) -> float:
        """Return the skew factor for a channel; 1.0 for any unset channel."""
        return self._skews.get(channel_name, 1.0)

    def executed_ml(self, channel_name: str, requested_ml: float) -> float:
        """Return the executed dose for a channel: requested_ml × factor(channel_name)."""
        return requested_ml * self.factor(channel_name)

    @property
    def factors(self) -> dict[str, float]:
        """Snapshot copy of ACTIVE skews only (factor != 1.0); mutating it changes nothing."""
        return dict(self._skews)
