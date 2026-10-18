import xml.etree.ElementTree as ET
from pathlib import Path

from analyzer import avatars


def test_avatar_files_exist_and_are_valid_svg():
    for path in (avatars.USER_AVATAR, avatars.ASSISTANT_AVATAR):
        assert Path(path).is_file(), path
        assert ET.parse(path).getroot().tag.endswith("svg")


def test_roles_map_to_their_avatar():
    assert avatars.AVATARS == {"user": avatars.USER_AVATAR, "assistant": avatars.ASSISTANT_AVATAR}


def test_assistant_robot_is_light_yellow_not_blue_grey():
    text = Path(avatars.ASSISTANT_AVATAR).read_text().lower()
    assert "#fff1a8" in text and "#cfd8e6" not in text


def test_busy_avatar_is_the_robot_with_signal_rings_that_animate_forever():
    text = Path(avatars.ASSISTANT_AVATAR_BUSY).read_text()
    assert ET.parse(avatars.ASSISTANT_AVATAR_BUSY).getroot().tag.endswith("svg")
    assert "#FFF1A8" in text
    assert text.count("<animate") >= 6 and text.count('repeatCount="indefinite"') == text.count("<animate")
    assert "<script" not in text and "href" not in text.lower().replace("xlink", "")
