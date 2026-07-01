from __future__ import annotations

import sys
from abc import ABC, abstractmethod

_GPIO = None


class OutputBase(ABC):
    """
    Common interface for output devices.

    write(True)  -> active / pass / relay ON
    write(False) -> inactive / fail-safe / relay OFF
    """

    state: bool | None

    @abstractmethod
    def write(self, passed: bool) -> None:
        pass

    def fail_safe_low(self) -> None:
        try:
            self.write(False)
        except Exception as exc:
            print(f"Warning: could not set output LOW/OFF: {exc}", file=sys.stderr)

    @abstractmethod
    def cleanup(self) -> None:
        pass


class GpioOutput(OutputBase):
    def __init__(
        self,
        pin: int | None,
        numbering: str = "BOARD",
        dry_run: bool = False,
        disabled: bool = False,
    ):
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

    def cleanup(self) -> None:
        self.fail_safe_low()

        if self.gpio is not None and self.pin is not None:
            self.gpio.cleanup(self.pin)


class UsbRelayOutput(OutputBase):
    """
    Output class for LCUS-1 / ARCELI USB relay module.

    Protocol:
        ON  = A0 01 01 A2
        OFF = A0 01 00 A1
    """

    RELAY_ON = bytes([0xA0, 0x01, 0x01, 0xA2])
    RELAY_OFF = bytes([0xA0, 0x01, 0x00, 0xA1])

    def __init__(
        self,
        port: str | None,
        baudrate: int = 9600,
        dry_run: bool = False,
        disabled: bool = False,
    ):
        self.port = port
        self.baudrate = baudrate
        self.dry_run = dry_run
        self.disabled = disabled
        self.state: bool | None = None
        self.serial = None

        if self.disabled:
            return

        if self.port is None:
            raise ValueError("--usb-relay-port is required unless --usb-relay-dry-run or --no-output is set")

        if self.dry_run:
            print(f"USB relay dry-run: {self.port} at {self.baudrate} baud")
            self.write(False)
            return

        import serial

        self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
        self.write(False)

    def write(self, passed: bool) -> None:
        passed = bool(passed)

        if self.disabled:
            self.state = passed
            return

        if self.state == passed:
            return

        self.state = passed

        command = self.RELAY_ON if passed else self.RELAY_OFF

        if self.dry_run:
            print(f"USB relay dry-run: {self.port} -> {'ON' if passed else 'OFF'}")
            return

        if self.serial is not None:
            self.serial.write(command)
            self.serial.flush()

    def cleanup(self) -> None:
        self.fail_safe_low()

        if self.serial is not None:
            self.serial.close()
            self.serial = None