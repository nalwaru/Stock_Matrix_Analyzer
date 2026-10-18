import re

from analyzer.loading import COIN_COUNT, money_bag_html


def test_animation_has_a_money_bag_and_several_coins_that_move():
    html = money_bag_html()
    assert "\U0001F4B0" in html  # money bag
    assert html.count('class="mb-coin"') == COIN_COUNT >= 5
    assert "@keyframes mb-pop" in html and "@keyframes mb-wobble" in html
    assert "animation" in html and "infinite" in html


def test_each_coin_has_its_own_delay_and_direction_so_they_do_not_move_in_lockstep():
    html = money_bag_html()
    delays = re.findall(r"animation-delay:([\d.]+)s", html)
    drift = re.findall(r"--dx:(-?\d+)px", html)
    assert len(set(delays)) == COIN_COUNT and len(set(drift)) >= 3


def test_respects_reduced_motion_and_is_labelled_for_screen_readers():
    html = money_bag_html()
    assert "prefers-reduced-motion" in html
    assert 'role="img"' in html and "aria-label=" in html


def test_is_self_contained_html_without_scripts_or_external_requests():
    html = money_bag_html()
    assert "<script" not in html and "http" not in html


def test_coins_are_drawn_with_css_not_an_emoji_font():
    assert "\U0001FA99" not in money_bag_html()  # coin emoji varies by platform; CSS coin does not
