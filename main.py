# Compatibility shim for legacy tests expecting functions in main.
# New architecture implements logic in os_ai_core.tools.computer.

import os
import sys
import glob
from typing import Optional, Tuple

# Ensure all workspace package sources are importable when running directly
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _src_dir in glob.glob(os.path.join(_ROOT, "packages", "*", "src")):
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

from os_ai_core.tools import computer as _computer  # real implementation
import os_ai_os.api as _os_api
from os_ai_os.platform.drivers import PlatformDrivers
from os_ai_os.ports.types import Capabilities, Size

try:
    import pyautogui  # type: ignore
except Exception:
    pyautogui = None  # type: ignore


class _NoOpOverlay:
    def highlight(self, x: int, y: int, *, radius: Optional[int] = None, duration: Optional[float] = None) -> None:
        return None

    def process_events(self) -> None:
        return None


class _NoOpSound:
    def play_click(self) -> None:
        return None

    def play_done(self) -> None:
        return None


class _AlwaysGrantedPermissions:
    def has_input_access(self) -> bool:
        return True

    def ensure_input_access(self) -> None:
        return None

    def has_screen_recording(self) -> bool:
        return True

    def ensure_screen_recording(self) -> None:
        return None


class _CompatMouse:
    def __init__(self, pag):
        self._pag = pag

    def position(self) -> Tuple[int, int]:
        if not hasattr(self._pag, "position"):
            return 0, 0
        x, y = self._pag.position()
        return int(x), int(y)

    def move_to(self, x: int, y: int, *, duration_ms: int = 0) -> None:
        self._pag.moveTo(int(x), int(y), duration=max(0.0, float(duration_ms) / 1000.0))

    def click(self, *, button: str = "left", clicks: int = 1) -> None:
        self._pag.click(button=button, clicks=int(clicks), interval=0.05)

    def down(self, *, button: str = "left") -> None:
        self._pag.mouseDown(button=button)

    def up(self, *, button: str = "left") -> None:
        self._pag.mouseUp(button=button)

    def scroll(self, *, dx: int = 0, dy: int = 0) -> None:
        if dy:
            self._pag.scroll(int(dy))
        if dx:
            try:
                self._pag.hscroll(int(dx))
            except AttributeError:
                self._pag.keyDown("shift")
                try:
                    self._pag.scroll(int(dx))
                finally:
                    self._pag.keyUp("shift")

    def drag(self, start: Tuple[int, int], end: Tuple[int, int], *, steps: int = 1, delay_ms: int = 0) -> None:
        raise NotImplementedError("legacy shim does not route drag() directly")


class _CompatKeyboard:
    def __init__(self, pag):
        self._pag = pag

    def press_enter(self) -> None:
        press_enter_mac()

    def press_combo(self, keys: Tuple[str, ...]) -> None:
        if not keys:
            return
        if len(keys) == 1:
            self._pag.press(keys[0])
        else:
            self._pag.hotkey(*keys)

    def key_down(self, key: str) -> None:
        self._pag.keyDown(key)

    def key_up(self, key: str) -> None:
        self._pag.keyUp(key)

    def type_text(self, text: str, *, wpm: int = 180) -> None:
        self._pag.write(text, interval=0.02)


class _CompatScreen:
    def __init__(self, pag):
        self._pag = pag

    def size(self) -> Size:
        if not hasattr(self._pag, "size"):
            return Size(width=1920, height=1080)
        w, h = self._pag.size()
        return Size(width=int(w), height=int(h))

    def screenshot(self, region: Optional[Tuple[int, int, int, int]] = None):
        if not hasattr(self._pag, "screenshot"):
            from PIL import Image  # type: ignore

            return Image.new("RGB", (1920, 1080), (0, 0, 0))
        if region is None:
            return self._pag.screenshot()
        return self._pag.screenshot(region=region)


def _compat_drivers() -> PlatformDrivers | None:
    pag = globals().get("pyautogui")
    if pag is None:
        return None
    return PlatformDrivers(
        mouse=_CompatMouse(pag),
        keyboard=_CompatKeyboard(pag),
        screen=_CompatScreen(pag),
        overlay=_NoOpOverlay(),
        permissions=_AlwaysGrantedPermissions(),
        sound=_NoOpSound(),
        capabilities=Capabilities(),
    )


def press_enter_mac():
    try:
        if pyautogui is not None:
            pyautogui.press("enter")  # type: ignore
    except Exception:
        pass


def handle_computer_action(action, params):  # type: ignore
    original_drivers = _os_api._drivers  # type: ignore[attr-defined]
    compat = _compat_drivers()

    try:
        if compat is not None:
            _os_api._drivers = compat  # type: ignore[attr-defined]
            _computer._GEOMETRY_READY = False  # type: ignore[attr-defined]
        res = _computer.handle_computer_action(action, params)
        try:
            globals()["LAST_SCREENSHOT_PATH"] = getattr(_computer, "LAST_SCREENSHOT_PATH", "")
        except Exception:
            pass
        return res
    finally:
        _os_api._drivers = original_drivers  # type: ignore[attr-defined]
        _computer._GEOMETRY_READY = False  # type: ignore[attr-defined]


if __name__ == "__main__":
    from os_ai_cli.main import main as cli_main
    try:
        code = cli_main()
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
        code = 130
    raise SystemExit(code)
