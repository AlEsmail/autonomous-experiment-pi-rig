# ABOUTME: Per-cycle camera-lock watchdog: verifies v4l2 controls still hold the locked
# ABOUTME: values after a possible power-cycle reset, and flags blank-LAB drift beyond threshold.

from shared.color import delta_e


class CameraLock:
    def __init__(self, camera, controls: dict[str, int]) -> None:
        self._camera = camera
        self._controls = dict(controls)

    def apply(self) -> None:
        self._camera.apply_controls(self._controls)

    def verify(self) -> list[str]:
        current = self._camera.read_controls(self._controls.keys())
        messages = []
        for name, locked_value in self._controls.items():
            actual_value = current[name]
            if actual_value != locked_value:
                messages.append(
                    f"{name} mismatch: expected {locked_value}, got {actual_value}"
                )
        return messages


def blank_drift_alarm(
    blank_lab: tuple[float, float, float],
    reference_lab: tuple[float, float, float],
    threshold_delta_e: float,
) -> str | None:
    drift = delta_e(blank_lab, reference_lab)
    if drift <= threshold_delta_e:
        return None
    return f"blank LAB drift {drift:.2f} exceeds threshold {threshold_delta_e:.2f}"
