from __future__ import annotations

from typing import Optional, Tuple

import pyautogui

from os_ai_os.config import PYAUTO_FAILSAFE, PYAUTO_PAUSE_SECONDS
from os_ai_os.platform.drivers import PlatformDrivers
from os_ai_os.ports.mouse import Mouse
from os_ai_os.ports.keyboard import Keyboard
from os_ai_os.ports.screen import Screen
from os_ai_os.ports.overlay import Overlay
from os_ai_os.ports.sound import Sound
from os_ai_os.ports.permissions import Permissions
from os_ai_os.ports.types import Capabilities, Size

from .keyboard import press_enter_mac
from .overlay import highlight_position, process_overlay_events
from .sound import play_click_sound, play_done_sound

pyautogui.PAUSE = PYAUTO_PAUSE_SECONDS
pyautogui.FAILSAFE = PYAUTO_FAILSAFE


def _normalize_combo_keys(keys: Tuple[str, ...]) -> Tuple[str, ...]:
    """Normalize incoming combo keys to PyAutoGUI names and split joined combos.

    - Supports inputs like ("cmd", "k") or ("cmd+k",)
    - Maps aliases: cmd->command, ctrl/control->ctrl, alt/option->option
    - Normalizes case to lowercase
    """
    mapping = {
        "cmd": "command",
        "command": "command",
        "ctrl": "ctrl",
        "control": "ctrl",
        "alt": "option",
        "option": "option",
        "shift": "shift",
        "enter": "enter",
        "return": "enter",
        "esc": "esc",
        "escape": "esc",
        "tab": "tab",
        "space": "space",
        "backspace": "backspace",
        "delete": "delete",
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
    }
    flat: list[str] = []
    for raw in keys:
        try:
            token = (raw or "").strip()
        except Exception:
            token = str(raw)
        if not token:
            continue
        parts = [p for p in token.split("+") if p]
        for p in parts:
            name = mapping.get(p.lower(), p.lower())
            flat.append(name)
    return tuple(flat)


class DarwinMouse:
    def position(self) -> Tuple[int, int]:
        x, y = pyautogui.position()
        return int(x), int(y)

    def move_to(self, x: int, y: int, *, duration_ms: int = 0) -> None:
        dur = max(0.0, float(duration_ms) / 1000.0)
        pyautogui.moveTo(int(x), int(y), duration=dur)

    def click(self, *, button: str = "left", clicks: int = 1) -> None:
        pyautogui.click(button=button, clicks=int(clicks), interval=0.05)

    def down(self, *, button: str = "left") -> None:
        pyautogui.mouseDown(button=button)

    def up(self, *, button: str = "left") -> None:
        pyautogui.mouseUp(button=button)

    def scroll(self, *, dx: int = 0, dy: int = 0) -> None:
        if dy:
            pyautogui.scroll(int(dy))
        if dx:
            try:
                pyautogui.hscroll(int(dx))
            except AttributeError:
                pyautogui.keyDown("shift")
                try:
                    pyautogui.scroll(int(dx))
                finally:
                    pyautogui.keyUp("shift")

    def drag(self, start: Tuple[int, int], end: Tuple[int, int], *, steps: int = 1, delay_ms: int = 0) -> None:
        sx, sy = int(start[0]), int(start[1])
        ex, ey = int(end[0]), int(end[1])
        pyautogui.moveTo(sx, sy)
        pyautogui.mouseDown(button="left")
        if steps <= 1:
            pyautogui.moveTo(ex, ey)
        else:
            for i in range(1, int(steps) + 1):
                nx = int(round(sx + (ex - sx) * (i / float(steps))))
                ny = int(round(sy + (ey - sy) * (i / float(steps))))
                pyautogui.moveTo(nx, ny)
                if delay_ms > 0:
                    pyautogui.sleep(max(0.0, float(delay_ms) / 1000.0))
        pyautogui.mouseUp(button="left")


class DarwinKeyboard:
    def press_enter(self) -> None:
        press_enter_mac()

    def press_combo(self, keys: Tuple[str, ...]) -> None:
        if not keys:
            return
        norm = _normalize_combo_keys(keys)
        if not norm:
            return
        if len(norm) == 1:
            # Special handling for Enter to use reliable Quartz event path
            if norm[0] in ("enter", "return"):
                press_enter_mac()
            else:
                pyautogui.press(norm[0])
            return
        pyautogui.hotkey(*norm)

    def key_down(self, key: str) -> None:
        norm = _normalize_combo_keys((key,))
        if norm:
            pyautogui.keyDown(norm[0])

    def key_up(self, key: str) -> None:
        norm = _normalize_combo_keys((key,))
        if norm:
            pyautogui.keyUp(norm[0])

    def type_text(self, text: str, *, wpm: int = 180) -> None:
        interval = 0.02
        try:
            cps = max(1.0, float(wpm) * 5.0 / 60.0)
            interval = max(0.0, 1.0 / cps)
        except Exception:
            pass
        pyautogui.write(text, interval=interval)


class DarwinScreen:
    def size(self) -> Size:
        w, h = pyautogui.size()
        return Size(width=int(w), height=int(h))

    def screenshot(self, region: Optional[Tuple[int, int, int, int]] = None):
        if region is not None:
            x, y, w, h = region
            return pyautogui.screenshot(region=(int(x), int(y), int(w), int(h)))
        return pyautogui.screenshot()


class DarwinOverlay:
    def highlight(self, x: int, y: int, *, radius: Optional[int] = None, duration: Optional[float] = None) -> None:
        highlight_position(int(x), int(y), radius=radius, duration=duration)

    def process_events(self) -> None:
        process_overlay_events()


class DarwinSound:
    def play_click(self) -> None:
        play_click_sound()

    def play_done(self) -> None:
        play_done_sound()


class DarwinPermissions:
    def has_input_access(self) -> bool:
        return True

    def ensure_input_access(self) -> None:
        return None

    def has_screen_recording(self) -> bool:
        return True

    def ensure_screen_recording(self) -> None:
        return None


def _detect_scale() -> float:
    try:
        from AppKit import NSScreen
        ms = NSScreen.mainScreen()
        if ms is not None:
            return float(ms.backingScaleFactor())
    except Exception:
        pass
    return 1.0


def make_drivers() -> PlatformDrivers:
    caps = Capabilities(
        supports_synthetic_input=True,
        supports_click_through_overlay=True,
        supports_smooth_move=True,
        dpi_scale=_detect_scale(),
        screen_recording_available=True,
    )
    return PlatformDrivers(
        mouse=DarwinMouse(),
        keyboard=DarwinKeyboard(),
        screen=DarwinScreen(),
        overlay=DarwinOverlay(),
        permissions=DarwinPermissions(),
        sound=DarwinSound(),
        capabilities=caps,
    )

