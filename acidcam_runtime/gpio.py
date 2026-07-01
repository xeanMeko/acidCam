from __future__ import annotations

import sys

_GPIO = None


class GpioOutput:
    def __init__(self, pin: int | None, numbering: str = "BOARD", dry_run: bool = False, disabled: bool = False):
        self.pin = pin
        self.numbering = numbering.upper()
        self.dry_run = dry_run
        self.disabled = disabled
        self.state: bool | None = None
        self.gpio = None
        if self.disabled:
            return
        if self.pin is None:
            raise ValueError("--gpio-pin is required unless --gpio-dry-run or --no-gpio is set")
        if self.dry_run:
            print(f"GPIO dry-run: {self.numbering} pin {self.pin} would be driven active-high")
            self.write(False)
            return
        global _GPIO
        if _GPIO is None:
            import Jetson.GPIO as GPIO

            _GPIO = GPIO
        self.gpio = _GPIO
        self.gpio.setmode(getattr(self.gpio, self.numbering))
        self.gpio.setup(self.pin, self.gpio.OUT, initial=self.gpio.LOW)
        self.state = False

    def write(self, passed: bool) -> None:
        passed = bool(passed)
        if self.disabled:
            self.state = passed
            return
        if self.state == passed:
            return
        self.state = passed
        if self.dry_run:
            print(f"GPIO dry-run: pin {self.pin} -> {'HIGH' if passed else 'LOW'}")
            return
        if self.gpio is not None:
            self.gpio.output(self.pin, self.gpio.HIGH if passed else self.gpio.LOW)

    def fail_safe_low(self) -> None:
        try:
            self.write(False)
        except Exception as exc:
            print(f"Warning: could not set GPIO LOW: {exc}", file=sys.stderr)

    def cleanup(self) -> None:
        self.fail_safe_low()
        if self.gpio is not None and self.pin is not None:
            self.gpio.cleanup(self.pin)
