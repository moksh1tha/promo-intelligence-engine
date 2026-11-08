"""Timestamp helpers. Screenplays use MM:SS.cc; outputs use HH:MM:SS.cc."""


def parse(ts: str) -> float:
    """Parse 'MM:SS.cc' or 'HH:MM:SS.cc' into seconds (rounded to 1/100 s)."""
    total = 0.0
    for part in ts.strip().split(":"):
        total = total * 60 + float(part)
    return round(total, 2)


def fmt(seconds: float) -> str:
    """Format seconds as HH:MM:SS.cc."""
    cs = int(round(seconds * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h:02d}:{m:02d}:{s:02d}.{c:02d}"


def fmt_short(seconds: float) -> str:
    """Format seconds as MM:SS.cc (screenplay style)."""
    cs = int(round(seconds * 100))
    m, rem = divmod(cs, 6000)
    s, c = divmod(rem, 100)
    return f"{m:02d}:{s:02d}.{c:02d}"
