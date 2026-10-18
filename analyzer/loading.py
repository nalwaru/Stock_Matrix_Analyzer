"""Self-contained HTML/CSS 'working' animation: the robot helper transmitting a signal.

No scripts and no external requests; the robot is the assistant avatar drawn inline, and the
signal is a set of rings that ripple out of its antenna."""
from pathlib import Path

WAVE_COUNT = 3
_WAVE_DELAY = 0.6  # seconds between rings
_ROBOT = (Path(__file__).parent.parent / "assets" / "assistant-avatar.svg").read_text()
_ROBOT_INNER = _ROBOT[_ROBOT.index(">") + 1: _ROBOT.rindex("</svg>")]
_STYLE = """
<style>
.rb-wrap{width:110px;height:110px;margin:2px 0 6px}
.rb-robot{animation:rb-bob 1.6s ease-in-out infinite}
.rb-wave{fill:none;stroke:#E0A113;stroke-width:2.5;opacity:0;transform-box:fill-box;
         transform-origin:center;animation:rb-ping 1.8s ease-out infinite}
@keyframes rb-ping{0%{opacity:.9;transform:scale(.4)}100%{opacity:0;transform:scale(3.4)}}
@keyframes rb-bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-2px)}}
@media (prefers-reduced-motion: reduce){.rb-robot,.rb-wave{animation:none}.rb-wave{opacity:.5}}
</style>
"""


def transmitting_html() -> str:
    waves = "".join(
        f'<circle class="rb-wave" cx="32" cy="5" r="6" style="animation-delay:{i * _WAVE_DELAY:.1f}s"/>'
        for i in range(WAVE_COUNT)
    )
    return (f'{_STYLE}<div class="rb-wrap" role="img" aria-label="Working on your question">'
            f'<svg viewBox="-16 -24 96 96" width="110" height="110">{waves}'
            f'<g class="rb-robot">{_ROBOT_INNER}</g></svg></div>')
