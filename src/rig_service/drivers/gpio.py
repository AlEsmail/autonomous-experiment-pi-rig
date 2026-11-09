# ABOUTME: Real relay driver for the Raspberry Pi: drives low-trigger relay boards via gpiozero,
# ABOUTME: holding every pin HIGH (relay off) from construction. Refuses to load off-Pi.

from __future__ import annotations


class GpioRelayDriver:
    def __init__(self, gpios: list[int]) -> None:
        try:
            from gpiozero import DigitalOutputDevice
        except ImportError as exc:
            raise RuntimeError(
                "GpioRelayDriver requires a Raspberry Pi with gpiozero installed; "
                "use FakeRelayDriver (RIG_DRIVER=fake) for development"
            ) from exc

        try:
            # active_high=False: relays are low-trigger, so device.on() drives the pin
            # LOW (relay energized) and initial_value=False leaves the pin HIGH (off).
            self._devices = {
                gpio: DigitalOutputDevice(gpio, active_high=False, initial_value=False)
                for gpio in gpios
            }
        except Exception as exc:
            raise RuntimeError(
                f"GpioRelayDriver could not claim GPIO pins (is this a Raspberry Pi?): {exc}"
            ) from exc

    def all_off(self) -> None:
        for device in self._devices.values():
            device.off()

    def set(self, gpio: int, on: bool) -> None:
        device = self._devices[gpio]
        if on:
            device.on()
        else:
            device.off()

    def relays_on(self) -> set[int]:
        return {gpio for gpio, device in self._devices.items() if device.value}
