"""Self-contained HTML/CSS 'working' animation: a money bag with gold coins popping out.

No scripts and no external requests; coins are drawn with CSS so they look the same everywhere."""

COIN_COUNT = 6
# (horizontal drift in px, rotation in degrees) per coin: fanned out, not in lockstep
_COINS = [(-46, -200), (-24, 160), (-6, -120), (14, 220), (34, -180), (54, 140)]
_STYLE = """
<style>
.mb-wrap{position:relative;width:170px;height:104px;margin:2px 0 6px}
.mb-bag{position:absolute;left:10px;bottom:0;font-size:58px;line-height:1;transform-origin:50% 92%;
        animation:mb-wobble 1.2s ease-in-out infinite}
.mb-coin{position:absolute;left:48px;bottom:46px;width:22px;height:22px;border-radius:50%;
         background:#f4be34;border:2px solid #cc8c14;box-sizing:border-box;color:#6e4600;
         font:700 13px/18px Arial,sans-serif;text-align:center;opacity:0;
         animation:mb-pop 1.8s ease-out infinite}
@keyframes mb-wobble{0%,100%{transform:rotate(-4deg) scale(1)}50%{transform:rotate(4deg) scale(1.06)}}
@keyframes mb-pop{0%{opacity:0;transform:translate(0,0) scale(.5) rotate(0)}
                  15%{opacity:1}
                  100%{opacity:0;transform:translate(var(--dx),-84px) scale(1) rotate(var(--rot))}}
@media (prefers-reduced-motion: reduce){.mb-bag,.mb-coin{animation:none}.mb-coin{display:none}}
</style>
"""


def money_bag_html() -> str:
    coins = "".join(
        f'<span class="mb-coin" style="--dx:{dx}px;--rot:{rot}deg;animation-delay:{i * 0.3:.1f}s">$</span>'
        for i, (dx, rot) in enumerate(_COINS)
    )
    return (f'{_STYLE}<div class="mb-wrap" role="img" aria-label="Working on your question">'
            f'<span class="mb-bag">\U0001F4B0</span>{coins}</div>')
