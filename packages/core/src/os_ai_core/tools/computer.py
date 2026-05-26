from __future__ import annotations

import base64
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

from os_ai_core.config import (
    LOGGER_NAME,
    COORD_X_SCALE,
    COORD_Y_SCALE,
    COORD_X_OFFSET,
    COORD_Y_OFFSET,
    POST_MOVE_VERIFY,
    POST_MOVE_TOLERANCE_PX,
    POST_MOVE_CORRECTION_DURATION,
    VIRTUAL_DISPLAY_ENABLED,
    VIRTUAL_DISPLAY_WIDTH_PX,
    VIRTUAL_DISPLAY_HEIGHT_PX,
    SCREENSHOT_MODE,
    SCREENSHOT_FORMAT,
    SCREENSHOT_JPEG_QUALITY,
)
from os_ai_os.api import get_drivers
from os_ai_os.config import (
    DEFAULT_MOVE_SPEED_PPS,
    DEFAULT_DRAG_SPEED_PPS,
    MIN_MOVE_DURATION,
    MAX_MOVE_DURATION,
    PREMOVE_HIGHLIGHT_DEFAULT_DURATION,
)


LAST_SCREENSHOT_PATH: str = ""

SCREEN_W = 1
SCREEN_H = 1
MODEL_DISPLAY_W = 1
MODEL_DISPLAY_H = 1
DYNAMIC_X_SCALE = 1.0
DYNAMIC_Y_SCALE = 1.0
MODEL_CONTENT_W = 1
MODEL_CONTENT_H = 1
MODEL_LB_OFFSET_X = 0
MODEL_LB_OFFSET_Y = 0
CONTENT_X_SCALE = 1.0
CONTENT_Y_SCALE = 1.0
_GEOMETRY_READY = False


def _drivers():
    return get_drivers()


def _duration_ms(seconds: float) -> int:
    return max(0, int(round(float(seconds) * 1000.0)))


def _is_input_fail_safe(exc: BaseException) -> bool:
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    return "failsafe" in name or "fail-safe" in text or "fail safe" in text


def _ensure_geometry(*, refresh: bool = False) -> None:
    global SCREEN_W, SCREEN_H
    global MODEL_DISPLAY_W, MODEL_DISPLAY_H, DYNAMIC_X_SCALE, DYNAMIC_Y_SCALE
    global MODEL_CONTENT_W, MODEL_CONTENT_H, MODEL_LB_OFFSET_X, MODEL_LB_OFFSET_Y
    global CONTENT_X_SCALE, CONTENT_Y_SCALE, _GEOMETRY_READY

    if _GEOMETRY_READY and not refresh:
        return

    size = _drivers().screen.size()
    SCREEN_W = max(1, int(size.width))
    SCREEN_H = max(1, int(size.height))

    if (SCREENSHOT_MODE or "downscale").lower() == "native":
        MODEL_DISPLAY_W = SCREEN_W
        MODEL_DISPLAY_H = SCREEN_H
        DYNAMIC_X_SCALE = 1.0
        DYNAMIC_Y_SCALE = 1.0
    elif VIRTUAL_DISPLAY_ENABLED:
        try:
            vd_w = int(VIRTUAL_DISPLAY_WIDTH_PX)
        except Exception:
            vd_w = SCREEN_W
        try:
            vd_h = int(VIRTUAL_DISPLAY_HEIGHT_PX)
        except Exception:
            vd_h = 0
        if vd_w <= 0:
            vd_w = SCREEN_W
        if vd_h <= 0:
            vd_h = max(1, int(round(float(SCREEN_H) * float(vd_w) / float(SCREEN_W))))
        MODEL_DISPLAY_W = vd_w
        MODEL_DISPLAY_H = vd_h
        try:
            DYNAMIC_X_SCALE = float(SCREEN_W) / float(MODEL_DISPLAY_W)
            DYNAMIC_Y_SCALE = float(SCREEN_H) / float(MODEL_DISPLAY_H)
        except Exception:
            DYNAMIC_X_SCALE = 1.0
            DYNAMIC_Y_SCALE = 1.0
    else:
        MODEL_DISPLAY_W = SCREEN_W
        MODEL_DISPLAY_H = SCREEN_H
        DYNAMIC_X_SCALE = 1.0
        DYNAMIC_Y_SCALE = 1.0

    try:
        if (SCREENSHOT_MODE or "downscale").lower() == "downscale" and VIRTUAL_DISPLAY_ENABLED:
            screen_aspect = float(SCREEN_W) / float(SCREEN_H)
            model_aspect = float(MODEL_DISPLAY_W) / float(MODEL_DISPLAY_H)
            if screen_aspect > model_aspect:
                MODEL_CONTENT_W = int(MODEL_DISPLAY_W)
                MODEL_CONTENT_H = max(1, int(round(MODEL_DISPLAY_W / screen_aspect)))
                MODEL_LB_OFFSET_X = 0
                MODEL_LB_OFFSET_Y = int((int(MODEL_DISPLAY_H) - MODEL_CONTENT_H) / 2)
            else:
                MODEL_CONTENT_H = int(MODEL_DISPLAY_H)
                MODEL_CONTENT_W = max(1, int(round(MODEL_DISPLAY_H * screen_aspect)))
                MODEL_LB_OFFSET_Y = 0
                MODEL_LB_OFFSET_X = int((int(MODEL_DISPLAY_W) - MODEL_CONTENT_W) / 2)
        else:
            MODEL_CONTENT_W = int(MODEL_DISPLAY_W)
            MODEL_CONTENT_H = int(MODEL_DISPLAY_H)
            MODEL_LB_OFFSET_X = 0
            MODEL_LB_OFFSET_Y = 0
    except Exception:
        MODEL_CONTENT_W = int(MODEL_DISPLAY_W)
        MODEL_CONTENT_H = int(MODEL_DISPLAY_H)
        MODEL_LB_OFFSET_X = 0
        MODEL_LB_OFFSET_Y = 0

    try:
        CONTENT_X_SCALE = float(SCREEN_W) / float(MODEL_CONTENT_W)
        CONTENT_Y_SCALE = float(SCREEN_H) / float(MODEL_CONTENT_H)
    except Exception:
        CONTENT_X_SCALE = float(SCREEN_W) / float(max(1, int(MODEL_DISPLAY_W)))
        CONTENT_Y_SCALE = float(SCREEN_H) / float(max(1, int(MODEL_DISPLAY_H)))

    _GEOMETRY_READY = True


def _mouse_position() -> Tuple[int, int]:
    return _drivers().mouse.position()


def _move_to(x: int, y: int, duration_seconds: float = 0.0) -> None:
    _drivers().mouse.move_to(int(x), int(y), duration_ms=_duration_ms(duration_seconds))


def _press_combo(keys: List[str] | Tuple[str, ...]) -> None:
    clean = tuple(k for k in keys if isinstance(k, str) and k.strip())
    if clean:
        _drivers().keyboard.press_combo(clean)


def _press_single_key(key: str) -> None:
    if key in ("enter", "return"):
        _drivers().keyboard.press_enter()
        return
    _press_combo((key,))


def _key_down(key: str) -> None:
    _drivers().keyboard.key_down(key)


def _key_up(key: str) -> None:
    _drivers().keyboard.key_up(key)


def _type_text(text: str) -> None:
    _drivers().keyboard.type_text(text, wpm=600)


def _compute_duration_to(
    target_x: int,
    target_y: int,
    params: Dict[str, Any],
    *,
    default: float,
    speed_pps: float,
) -> float:
    try:
        if "duration" in params or "move_duration" in params:
            val = float(params.get("duration", params.get("move_duration")))
            return max(MIN_MOVE_DURATION, min(MAX_MOVE_DURATION, val))
        cx, cy = _mouse_position()
        dist = ((target_x - cx) ** 2 + (target_y - cy) ** 2) ** 0.5
        dur = dist / float(speed_pps)
        return max(MIN_MOVE_DURATION, min(MAX_MOVE_DURATION, dur))
    except Exception:
        return default


def computer_tool_handler(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    action = args.get("action") or args.get("type")
    if not action:
        return [{"type": "text", "text": "error: missing 'action'"}]
    try:
        args.setdefault("coordinate_space", "auto")
    except Exception:
        pass
    return handle_computer_action(action, args)


def _capture_driver_image():
    try:
        return _drivers().screen.screenshot()
    except Exception:
        return None


def _find_project_root(start_dir: str) -> str:
    if getattr(sys, "frozen", False):
        if sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            return os.path.join(base, "OS AI")
        if sys.platform == "darwin":
            return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "OS AI")
        return os.path.join(os.path.expanduser("~"), ".local", "share", "os-ai")

    cur = os.path.abspath(start_dir)
    sentinel_files = {"pyproject.toml", "requirements.txt", "Makefile"}
    for _ in range(8):
        try:
            entries = set(os.listdir(cur))
        except Exception:
            entries = set()
        if entries & sentinel_files or os.path.isdir(os.path.join(cur, "screenshots")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start_dir)


def b64_image_from_screenshot() -> Dict[str, Any]:
    _ensure_geometry(refresh=True)
    img = _capture_driver_image()
    if img is None:
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": ""},
        }

    try:
        from PIL import Image  # type: ignore

        target_w, target_h = int(MODEL_DISPLAY_W), int(MODEL_DISPLAY_H)
        if (SCREENSHOT_MODE or "downscale").lower() == "downscale":
            if img.width != SCREEN_W or img.height != SCREEN_H:
                img = img.resize((SCREEN_W, SCREEN_H), resample=getattr(Image, "LANCZOS", None) or Image.BILINEAR)
            content = img.resize((MODEL_CONTENT_W, MODEL_CONTENT_H), resample=getattr(Image, "LANCZOS", None) or Image.BILINEAR)
            canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            canvas.paste(content, (MODEL_LB_OFFSET_X, MODEL_LB_OFFSET_Y))
            img = canvas
        elif img.width != target_w or img.height != target_h:
            resample = getattr(Image, "LANCZOS", getattr(Image, "BILINEAR", None))
            img = img.resize((target_w, target_h), resample=resample)
    except Exception:
        pass

    from io import BytesIO

    buf = BytesIO()
    fmt = (SCREENSHOT_FORMAT or "PNG").upper()
    media_type = "image/png" if fmt == "PNG" else "image/jpeg"
    try:
        try:
            save_root = _find_project_root(os.path.dirname(__file__))
            save_dir = os.path.join(save_root, "screenshots")
            os.makedirs(save_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            ms = int((time.time() - int(time.time())) * 1000)
            ext = "jpg" if fmt == "JPEG" else "png"
            file_path = os.path.join(save_dir, f"screenshot_{ts}_{ms:03d}.{ext}")
            if fmt == "JPEG":
                img_to_save = img.convert("RGB")
                img_to_save.save(file_path, format="JPEG", quality=int(SCREENSHOT_JPEG_QUALITY or 85))
            else:
                img.save(file_path, format="PNG")
            logging.getLogger(LOGGER_NAME).info("Saved screenshot: %s", file_path)
            global LAST_SCREENSHOT_PATH
            LAST_SCREENSHOT_PATH = file_path
        except Exception:
            pass

        if fmt == "JPEG":
            img_enc = img.convert("RGB")
            img_enc.save(buf, format="JPEG", quality=int(SCREENSHOT_JPEG_QUALITY or 85))
        else:
            img.save(buf, format="PNG")
    except Exception:
        try:
            img.save(buf, format="PNG")
            media_type = "image/png"
        except Exception:
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": ""},
            }

    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def clamp_xy(x: int, y: int) -> Tuple[int, int]:
    _ensure_geometry()
    return max(0, min(x, SCREEN_W - 1)), max(0, min(y, SCREEN_H - 1))


def _apply_calibration(x: int, y: int) -> Tuple[int, int]:
    try:
        cx = int(round(x * float(COORD_X_SCALE) + float(COORD_X_OFFSET)))
        cy = int(round(y * float(COORD_Y_SCALE) + float(COORD_Y_OFFSET)))
        return cx, cy
    except Exception:
        return x, y


def _to_screen_xy(x: int, y: int, *, coordinate_space: str | None = None) -> Tuple[int, int]:
    _ensure_geometry()
    try:
        space = (coordinate_space or "screen").lower()
    except Exception:
        space = "screen"
    sx, sy = int(x), int(y)
    if space == "auto":
        try:
            if int(sx) > int(MODEL_DISPLAY_W) or int(sy) > int(MODEL_DISPLAY_H):
                space = "screen"
            else:
                space = "model"
        except Exception:
            space = "screen"
    if space == "model":
        try:
            sx_adj = float(sx) - float(MODEL_LB_OFFSET_X)
            sy_adj = float(sy) - float(MODEL_LB_OFFSET_Y)
            sx_adj = max(0.0, min(sx_adj, float(MODEL_CONTENT_W) - 1.0))
            sy_adj = max(0.0, min(sy_adj, float(MODEL_CONTENT_H) - 1.0))
            sx = int(round(sx_adj * float(CONTENT_X_SCALE)))
            sy = int(round(sy_adj * float(CONTENT_Y_SCALE)))
        except Exception:
            try:
                sx = int(round(float(sx) * float(DYNAMIC_X_SCALE)))
                sy = int(round(float(sy) * float(DYNAMIC_Y_SCALE)))
            except Exception:
                pass
    sx, sy = _apply_calibration(sx, sy)
    return clamp_xy(sx, sy)


def parse_key_combo(combo: str) -> List[str]:
    is_mac = sys.platform == "darwin"
    meta_key = "command" if is_mac else "win"
    alt_key = "option" if is_mac else "alt"
    mapping = {
        "cmd": meta_key,
        "command": meta_key,
        "super": meta_key,
        "meta": meta_key,
        "ctrl": "ctrl",
        "control": "ctrl",
        "alt": alt_key,
        "option": alt_key,
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
    keys: List[str] = []
    for k in combo.lower().split("+"):
        k = k.strip()
        if k:
            keys.append(mapping.get(k, k))
    return keys


def _keys_from_fallback_text(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = [t for t in text.replace("+", " ").split() if t]
    if not tokens:
        return []
    mapped: List[str] = []
    for t in tokens:
        mapped.extend(parse_key_combo(t))
    return mapped


def _with_modifiers(mods: List[str], action_fn: Callable[[], Any]) -> Any:
    mods = [m for m in mods if m]
    try:
        for m in mods:
            _key_down(m)
        return action_fn()
    finally:
        for m in reversed(mods):
            try:
                _key_up(m)
            except Exception:
                pass


def _modifiers_from_params(params: Dict[str, Any]) -> List[str]:
    raw_mods = params.get("modifiers") or []
    if isinstance(raw_mods, str):
        raw_mods = [s.strip() for s in raw_mods.split("+") if s.strip()]
    return parse_key_combo("+".join(raw_mods)) if raw_mods else []


def _move_to_coord(coord: Any, params: Dict[str, Any], *, default: float, speed_pps: float) -> Tuple[int, int]:
    coord_space = params.get("coordinate_space")
    x, y = _to_screen_xy(int(coord[0]), int(coord[1]), coordinate_space=coord_space)
    dur = _compute_duration_to(x, y, params, default=default, speed_pps=speed_pps)
    _move_to(x, y, dur)
    return x, y


def handle_computer_action(action: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    logger = logging.getLogger(LOGGER_NAME)
    _ensure_geometry()

    if action == "screenshot":
        return [b64_image_from_screenshot()]

    if action == "mouse_move":
        x, y = params.get("coordinate", [0, 0])
        x, y = _to_screen_xy(int(x), int(y), coordinate_space=params.get("coordinate_space"))
        dur = _compute_duration_to(x, y, params, default=0.35, speed_pps=DEFAULT_MOVE_SPEED_PPS)
        try:
            try:
                _drivers().overlay.highlight(x, y, duration=PREMOVE_HIGHLIGHT_DEFAULT_DURATION)
            except Exception:
                pass
            _move_to(x, y, dur)
            try:
                _drivers().overlay.process_events()
            except Exception:
                pass
            if POST_MOVE_VERIFY:
                try:
                    ax, ay = _mouse_position()
                    dx, dy = abs(ax - x), abs(ay - y)
                    if dx > POST_MOVE_TOLERANCE_PX or dy > POST_MOVE_TOLERANCE_PX:
                        _move_to(x, y, max(0.0, POST_MOVE_CORRECTION_DURATION))
                except Exception:
                    pass
        except Exception as exc:
            if _is_input_fail_safe(exc):
                logger.warning("Input fail-safe triggered during move; skipping move")
                return [{"type": "text", "text": "move skipped: fail-safe"}]
            raise
        if os.environ.get("SCREENSHOT_AFTER_ACTIONS") == "1":
            return [b64_image_from_screenshot()]
        return [{"type": "text", "text": "ok"}]

    if action in ("left_click", "double_click", "triple_click", "right_click", "middle_click"):
        coord = params.get("coordinate")
        clicks = 1
        button = "left"
        if action == "double_click":
            clicks = 2
        if action == "triple_click":
            clicks = 3
        if action == "right_click":
            button = "right"
        if action == "middle_click":
            button = "middle"
        modifiers = _modifiers_from_params(params)
        try:
            if coord:
                _move_to_coord(coord, params, default=0.30, speed_pps=DEFAULT_MOVE_SPEED_PPS)

            def _do() -> None:
                _drivers().mouse.click(clicks=clicks, button=button)

            _with_modifiers(modifiers, _do)
        except Exception as exc:
            if _is_input_fail_safe(exc):
                logger.warning("Input fail-safe triggered during click; skipping click")
                return [{"type": "text", "text": "click skipped: fail-safe"}]
            raise
        try:
            _drivers().sound.play_click()
        except Exception:
            pass
        return [{"type": "text", "text": f"done: {action}"}]

    if action in ("left_mouse_down", "left_mouse_up"):
        coord = params.get("coordinate")
        modifiers = _modifiers_from_params(params)
        try:
            if coord:
                _move_to_coord(coord, params, default=0.30, speed_pps=DEFAULT_MOVE_SPEED_PPS)

            def _do() -> None:
                if action == "left_mouse_down":
                    _drivers().mouse.down(button="left")
                else:
                    _drivers().mouse.up(button="left")

            _with_modifiers(modifiers, _do)
        except Exception as exc:
            if _is_input_fail_safe(exc):
                logger.warning("Input fail-safe triggered during mouse down/up; skipping")
                return [{"type": "text", "text": f"{action} skipped: fail-safe"}]
            raise
        return [{"type": "text", "text": f"done: {action}"}]

    if action == "left_click_drag":
        start = (
            params.get("start")
            or params.get("from")
            or params.get("source")
            or params.get("start_coordinate")
            or params.get("from_coordinate")
        )
        end = (
            params.get("end")
            or params.get("to")
            or params.get("target")
            or params.get("end_coordinate")
            or params.get("to_coordinate")
        )
        full_path = params.get("path")
        if not (start and end):
            return [{"type": "text", "text": "drag skipped: missing start/end"}]
        coord_space = params.get("coordinate_space")
        x1, y1 = _to_screen_xy(int(start[0]), int(start[1]), coordinate_space=coord_space)
        x2, y2 = _to_screen_xy(int(end[0]), int(end[1]), coordinate_space=coord_space)
        hold_before_ms = int(params.get("hold_before_ms", 50))
        hold_after_ms = int(params.get("hold_after_ms", 50))
        steps = max(1, int(params.get("steps", 1)))
        step_delay = max(0.0, float(params.get("step_delay", 0.0)))
        modifiers = _modifiers_from_params(params)
        move_dur = _compute_duration_to(x1, y1, params, default=0.30, speed_pps=DEFAULT_MOVE_SPEED_PPS)

        def _do_drag() -> None:
            time.sleep(max(0.0, hold_before_ms / 1000.0))
            _drivers().mouse.down(button="left")
            try:
                if full_path and len(full_path) > 2:
                    for pt in full_path[1:]:
                        px, py = _to_screen_xy(int(pt[0]), int(pt[1]), coordinate_space=coord_space)
                        cx, cy = _mouse_position()
                        dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                        seg_dur = max(0.01, dist / float(DEFAULT_DRAG_SPEED_PPS))
                        _move_to(px, py, seg_dur)
                        if step_delay > 0:
                            time.sleep(step_delay)
                elif steps <= 1:
                    drag_dur = _compute_duration_to(x2, y2, params, default=0.40, speed_pps=DEFAULT_DRAG_SPEED_PPS)
                    _move_to(x2, y2, drag_dur)
                else:
                    for i in range(1, steps + 1):
                        nx = int(round(x1 + (x2 - x1) * (i / float(steps))))
                        ny = int(round(y1 + (y2 - y1) * (i / float(steps))))
                        step_dur = _compute_duration_to(nx, ny, params, default=0.05, speed_pps=DEFAULT_DRAG_SPEED_PPS)
                        _move_to(nx, ny, step_dur)
                        if step_delay > 0:
                            time.sleep(step_delay)
                time.sleep(max(0.0, hold_after_ms / 1000.0))
            finally:
                _drivers().mouse.up(button="left")

        try:
            _move_to(x1, y1, move_dur)
            _with_modifiers(modifiers, _do_drag)
        except Exception as exc:
            if _is_input_fail_safe(exc):
                logger.warning("Input fail-safe triggered during drag; skipping drag")
                return [{"type": "text", "text": "drag skipped: fail-safe"}]
            raise
        return [{"type": "text", "text": f"done: {action}"}]

    if action == "type":
        text = params.get("text", "")
        try:
            has_non_ascii = any(ord(c) > 127 for c in text)
        except Exception:
            has_non_ascii = False
        looks_multiline = "\n" in str(text)
        looks_codey = any(tok in str(text) for tok in ("()", "{}", "[]", "'", '"', "=>", ": "))
        prefer_paste = has_non_ascii or looks_multiline or looks_codey
        if prefer_paste:
            try:
                import pyperclip  # type: ignore
                from os_ai_core.config import (
                    RESTORE_CLIPBOARD_AFTER_PASTE,
                    PASTE_COPY_DELAY_SECONDS,
                    PASTE_POST_DELAY_SECONDS,
                )

                try:
                    prev_clip = pyperclip.paste()
                except Exception:
                    prev_clip = None
                try:
                    pyperclip.copy(text)
                    time.sleep(PASTE_COPY_DELAY_SECONDS)
                    modifier = "command" if sys.platform == "darwin" else "ctrl"
                    _press_combo((modifier, "v"))
                    time.sleep(PASTE_POST_DELAY_SECONDS)
                finally:
                    if RESTORE_CLIPBOARD_AFTER_PASTE and prev_clip is not None:
                        try:
                            pyperclip.copy(prev_clip)
                        except Exception:
                            pass
                return [{"type": "text", "text": f"pasted {len(text)} chars via clipboard"}]
            except Exception:
                pass
        _type_text(str(text))
        return [{"type": "text", "text": "done: type"}]

    if action in ("key", "hold_key"):
        combo = params.get("key") or params.get("keys") or params.get("combo") or ""
        derived_from_text = False
        try:
            if isinstance(combo, str):
                norm_keys = [k for k in parse_key_combo(combo) if isinstance(k, str) and k.strip()]
            elif isinstance(combo, (list, tuple)):
                tmp: List[str] = []
                for v in combo:
                    if isinstance(v, str):
                        tmp.extend(parse_key_combo(v))
                    else:
                        s = str(v).strip()
                        if s:
                            tmp.append(s)
                norm_keys = [k for k in tmp if isinstance(k, str) and k.strip()]
            else:
                norm_keys = []
        except Exception:
            norm_keys = []

        if not norm_keys:
            fallback_text = params.get("text") or params.get("character")
            if isinstance(fallback_text, str) and fallback_text:
                maybe_keys = _keys_from_fallback_text(fallback_text)
                if maybe_keys:
                    norm_keys = maybe_keys
                    derived_from_text = True
                else:
                    _type_text(fallback_text)
                    return [{"type": "text", "text": f"typed: {len(fallback_text)} chars"}]
            if not norm_keys:
                combo_raw = combo if isinstance(combo, str) else str(combo)
                return [{"type": "text", "text": f"error: missing key combo (raw='{combo_raw}')"}]

        pressed_label = "+".join(norm_keys)
        if action == "hold_key":
            if len(norm_keys) < 2:
                return [{"type": "text", "text": "error: hold_key needs modifiers+key"}]
            try:
                for k in norm_keys[:-1]:
                    _key_down(k)
                _press_single_key(norm_keys[-1])
            finally:
                for k in reversed(norm_keys[:-1]):
                    _key_up(k)
        else:
            if len(norm_keys) == 1:
                _press_single_key(norm_keys[0])
            else:
                simple_non_mods = {"enter", "tab", "esc", "space", "backspace", "delete"}
                if derived_from_text and len(set(norm_keys)) == 1 and norm_keys[0] in simple_non_mods:
                    key = norm_keys[0]
                    count = len(norm_keys)
                    for _ in range(count):
                        _press_single_key(key)
                    pressed_label = f"{key} x{count}"
                else:
                    _press_combo(norm_keys)
        return [{"type": "text", "text": f"pressed: {pressed_label}"}]

    if action == "scroll":
        coord = params.get("coordinate")
        direction = (params.get("scroll_direction") or "down").lower()
        amount = int(params.get("scroll_amount", 1))
        try:
            if coord:
                _move_to_coord(coord, params, default=0.25, speed_pps=DEFAULT_MOVE_SPEED_PPS)
            if direction in ("down", "up"):
                clicks = -abs(amount) if direction == "down" else abs(amount)
                _drivers().mouse.scroll(dy=clicks)
            elif direction in ("left", "right"):
                clicks = -abs(amount) if direction == "left" else abs(amount)
                _drivers().mouse.scroll(dx=clicks)
        except Exception as exc:
            if _is_input_fail_safe(exc):
                return [{"type": "text", "text": "scroll skipped: fail-safe"}]
            raise
        return [{"type": "text", "text": "ok"}]

    if action == "wait":
        sec = float(params.get("seconds", 0.2))
        time.sleep(sec)
        return [{"type": "text", "text": "ok"}]

    return [{"type": "text", "text": f"error: unknown action '{action}'"}]


def computer_tool_handler_batch(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not args.get("_openai_batch"):
        return computer_tool_handler(args)

    actions = args.get("_openai_actions", [])
    if not actions:
        return [b64_image_from_screenshot()]

    cancel_token = args.get("_cancel_token")
    logger = logging.getLogger(LOGGER_NAME)
    for i, action_args in enumerate(actions):
        if cancel_token is not None and cancel_token.is_cancelled:
            logger.info("Batch cancelled at action %d/%d", i + 1, len(actions))
            break

        action_name = action_args.get("action", "")
        if action_name == "screenshot":
            continue

        logger.debug("Batch action %d/%d: %s", i + 1, len(actions), action_name)
        try:
            result = handle_computer_action(action_name, {**action_args, "coordinate_space": "auto"})
            for block in result:
                if isinstance(block, dict) and block.get("text", "").startswith("error:"):
                    logger.warning("Batch action %d failed: %s", i + 1, block.get("text"))
        except Exception as exc:
            logger.warning("Batch action %d exception: %s", i + 1, exc)

    return [b64_image_from_screenshot()]
