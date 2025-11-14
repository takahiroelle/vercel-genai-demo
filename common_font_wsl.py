# common_font_wsl.py
import os, glob, sys
import matplotlib
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

def _register_font(path: str):
    try:
        font_manager.fontManager.addfont(path)
        name = FontProperties(fname=path).get_name()
        matplotlib.rcParams["font.family"] = [name]
        matplotlib.rcParams["font.sans-serif"] = [name]
        matplotlib.rcParams["font.serif"] = [name]
        matplotlib.rcParams["axes.unicode_minus"] = False
        print(f"[font] registered {path} as {name}", file=sys.stderr)
        return name
    except Exception:
        matplotlib.rcParams["axes.unicode_minus"] = False
        print(f"[font] failed to register {path}", file=sys.stderr)
        return None

def set_font_from_file(font_path: str):
    if not font_path:
        return None
    font_path = os.path.expanduser(os.path.expandvars(font_path))
    if os.path.exists(font_path):
        return _register_font(font_path)
    return None

def _try_known_paths(paths):
    for raw in paths:
        candidate = os.path.expanduser(os.path.expandvars(raw))
        if os.path.exists(candidate):
            name = _register_font(candidate)
            if name:
                return name
    return None

def set_font_auto(manual_path: str = None, bundled_dir: str = "fonts"):
    env_path = os.environ.get("LOCAL_JP_FONT", "").strip()
    if env_path:
        print(f"[font] trying LOCAL_JP_FONT={env_path}", file=sys.stderr)
        name = set_font_from_file(env_path)
        if name: return name
    if manual_path:
        print(f"[font] trying manual path {manual_path}", file=sys.stderr)
        name = set_font_from_file(manual_path)
        if name: return name
    linux_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansJP-Regular.otf",
    ]
    name = _try_known_paths(linux_candidates)
    if name: return name
    win_fonts = "/mnt/c/Windows/Fonts"
    if os.path.isdir(win_fonts):
        patterns = [
            "YuGoth*.ttc","YUGOTH*.ttc","meiryo*.ttc","meiryo*.ttf",
            "msgothic.ttc","MSMINCHO.TTC","msmincho.ttc",
            "NotoSansCJK*.otf","NotoSansCJK*.ttc","NotoSansJP*.otf","NotoSansJP*.ttf"
        ]
        cands = []
        for pat in patterns:
            cands += glob.glob(os.path.join(win_fonts, pat))
        for p in cands:
            name = _register_font(p)
            if name: return name
    for ext in ("*.otf","*.ttf","*.ttc"):
        for p in glob.glob(os.path.join(bundled_dir, ext)):
            name = _register_font(p)
            if name: return name
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None
