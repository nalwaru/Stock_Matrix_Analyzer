import re

from analyzer.loading import WAVE_COUNT, transmitting_html


def test_animation_shows_the_light_yellow_robot_with_pulsing_signal_waves():
    html = transmitting_html()
    assert "#FFF1A8" in html  # the robot helper's light-yellow body
    assert html.count('class="rb-wave"') == WAVE_COUNT >= 3
    assert "@keyframes rb-ping" in html and "@keyframes rb-bob" in html
    assert "animation" in html and "infinite" in html


def test_each_wave_starts_at_a_different_time_so_the_signal_ripples():
    delays = re.findall(r"animation-delay:([\d.]+)s", transmitting_html())
    assert len(delays) == WAVE_COUNT and len(set(delays)) == WAVE_COUNT


def test_respects_reduced_motion_and_is_labelled_for_screen_readers():
    html = transmitting_html()
    assert "prefers-reduced-motion" in html
    assert 'role="img"' in html and "aria-label=" in html


def test_is_self_contained_html_without_scripts_or_external_requests():
    html = transmitting_html()
    assert "<script" not in html and "http" not in html


def test_no_money_bag_or_coins_while_working():
    html = transmitting_html()
    assert "\U0001F4B0" not in html and "mb-coin" not in html
