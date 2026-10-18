"""Chat avatars: flat SVG faces drawn for this app (assets/)."""
from pathlib import Path

ASSETS = Path(__file__).parent.parent / "assets"
USER_AVATAR = str(ASSETS / "user-avatar.svg")  # dollar-eyes face
ASSISTANT_AVATAR = str(ASSETS / "assistant-avatar.svg")  # light-yellow robot helper
AVATARS = {"user": USER_AVATAR, "assistant": ASSISTANT_AVATAR}
