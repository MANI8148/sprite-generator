"""Tests for dataset captioning module (roadmap: Phase 0 — Caption/tag sprite dataset)."""

import random
import pytest

from backend.modules.captioning import (
    generate_caption,
    generate_captions_batch,
    generate_caption_from_hf_item,
    CLASS_VOCAB,
    ACTION_VOCAB,
    DIRECTION_VOCAB,
)


class TestCaptionGeneration:
    def test_returns_string(self):
        caption = generate_caption()
        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_includes_class_name(self):
        caption = generate_caption(cls="enemy")
        assert "enemy" in caption.lower()

    def test_includes_action(self):
        caption = generate_caption(action="attack")
        assert "attack" in caption.lower()

    def test_includes_direction(self):
        caption = generate_caption(direction="left")
        assert "left" in caption.lower()

    def test_unknown_class_falls_back(self):
        caption = generate_caption(cls="unknown_thing_xyz")
        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_unknown_action_falls_back(self):
        caption = generate_caption(action="nonexistent_action")
        assert isinstance(caption, str)

    def test_unknown_direction_falls_back(self):
        caption = generate_caption(direction="diagonal_xyz")
        assert isinstance(caption, str)

    def test_short_templates(self):
        caption = generate_caption(use_short=True)
        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_all_vocab_classes_produce_valid_captions(self):
        for cls in CLASS_VOCAB:
            caption = generate_caption(cls=cls)
            assert cls in caption.lower()

    def test_all_vocab_actions_produce_valid_captions(self):
        for act in ACTION_VOCAB[:5]:
            caption = generate_caption(action=act)
            assert act in caption.lower()

    def test_all_vocab_directions_produce_valid_captions(self):
        for dire in DIRECTION_VOCAB:
            caption = generate_caption(direction=dire)
            assert dire in caption.lower()

    def test_custom_templates(self):
        custom = ["custom template {cls} {act} {dir}"]
        caption = generate_caption(templates=custom)
        assert caption == "custom template character idle front"

    def test_multiple_templates_rotation(self):
        custom = ["tmpl_a_{cls}", "tmpl_b_{act}"]
        seen = set()
        for _ in range(50):
            caption = generate_caption(templates=custom)
            seen.add(caption[:6])
        assert len(seen) == 2
        assert "tmpl_a" in seen
        assert "tmpl_b" in seen

    def test_seed_does_not_affect_randomness_across_calls(self):
        caps = set()
        for _ in range(20):
            caps.add(generate_caption(cls="character"))
        assert len(caps) > 1


class TestBatchGeneration:
    def test_batch_returns_list(self):
        metadata = [
            {"class": "character", "action": "idle", "direction": "front"},
            {"class": "enemy", "action": "attack", "direction": "left"},
        ]
        result = generate_captions_batch(metadata)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(c, str) for c in result)

    def test_batch_with_seed(self):
        metadata = [
            {"class": "character", "action": "idle", "direction": "front"},
            {"class": "enemy", "action": "walk", "direction": "right"},
        ]
        result1 = generate_captions_batch(metadata, seed=42)
        result2 = generate_captions_batch(metadata, seed=42)
        assert result1 == result2

    def test_batch_different_seeds(self):
        metadata = [
            {"class": "character", "action": "idle", "direction": "front"},
            {"class": "enemy", "action": "walk", "direction": "right"},
        ]
        result1 = generate_captions_batch(metadata, seed=1)
        result2 = generate_captions_batch(metadata, seed=2)
        assert result1 != result2

    def test_batch_with_short_templates(self):
        metadata = [{"class": "character", "action": "idle", "direction": "front"}]
        result = generate_captions_batch(metadata, use_short=True)
        assert len(result) == 1
        assert isinstance(result[0], str)

    def test_empty_batch(self):
        result = generate_captions_batch([])
        assert result == []

    def test_batch_missing_fields(self):
        metadata = [{}, {"class": "enemy"}, {"action": "walk", "direction": "left"}]
        result = generate_captions_batch(metadata)
        assert len(result) == 3
        assert all(isinstance(c, str) for c in result)


class TestHFItemCaptioning:
    def test_hf_item_with_all_fields(self):
        item = {"class": "wizard", "action": "cast", "direction": "front"}
        caption = generate_caption_from_hf_item(item)
        assert "wizard" in caption.lower()
        assert "cast" in caption.lower()

    def test_hf_item_missing_fields(self):
        item = {"class": "goblin"}
        caption = generate_caption_from_hf_item(item)
        assert "goblin" in caption.lower()

    def test_hf_item_empty_dict(self):
        caption = generate_caption_from_hf_item({})
        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_hf_item_custom_field_names(self):
        item = {"type": "dragon", "anim": "fly", "facing": "side"}
        caption = generate_caption_from_hf_item(
            item,
            cls_field="type",
            action_field="anim",
            direction_field="facing",
        )
        assert "dragon" in caption.lower()
        assert "fly" in caption.lower()

    def test_hf_item_short_mode(self):
        item = {"class": "knight", "action": "block", "direction": "back"}
        caption = generate_caption_from_hf_item(item, use_short=True)
        assert "knight" in caption.lower()
        assert "block" in caption.lower()
        assert "back" in caption.lower()

    def test_hf_item_unknown_class(self):
        item = {"class": "mythical_beast_42", "action": "roar", "direction": "front"}
        caption = generate_caption_from_hf_item(item)
        assert isinstance(caption, str)
        assert len(caption) > 0
