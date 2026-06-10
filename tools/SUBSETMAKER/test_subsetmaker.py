"""
Tests for the SubsetMaker core logic (non-GUI functions).
"""

import json
import os
import random
import textwrap
import pytest
import tempfile
import shutil

# ── import functions under test ────────────────────────────────────────────
from subsetmaker import (
    parse_label_file,
    scan_dataset,
    filter_and_remap_label,
    filter_label_file,
    create_subset,
    find_splits,
    images_dir_for_split,
    labels_dir_for_split,
    load_class_names,
    _write_data_yaml,
    check_dataset,
    fix_missing_labels,
    fix_orphan_labels,
    load_yaml_file,
    yaml_from_json,
    remap_labels,
    remap_yaml_classes,
    split_dataset as do_split_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def flat_dataset(tmp_path):
    """
    A minimal YOLO dataset with flat layout (no train/val split).

    images/
        img0.jpg  → labels: class 0
        img1.jpg  → labels: class 1
        img2.jpg  → labels: classes 0 and 1
    labels/
        img0.txt
        img1.txt
        img2.txt
    data.yaml
    """
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    # Create dummy image files (JPEG magic bytes: FF D8 FF E0)
    for name in ("img0.jpg", "img1.jpg", "img2.jpg"):
        (images_dir / name).write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    # Label files (YOLO format: class_id cx cy w h)
    (labels_dir / "img0.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (labels_dir / "img1.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    (labels_dir / "img2.txt").write_text("0 0.3 0.3 0.1 0.1\n1 0.7 0.7 0.1 0.1\n")

    # data.yaml
    (tmp_path / "data.yaml").write_text("nc: 2\nnames: [cat, dog]\n")

    return tmp_path


@pytest.fixture()
def split_dataset(tmp_path):
    """
    A YOLO dataset with train/val splits.
    """
    for split in ("train", "val"):
        img_dir = tmp_path / "images" / split
        lbl_dir = tmp_path / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        (img_dir / f"{split}_img0.jpg").write_bytes(b"\x00" * 50)
        (lbl_dir / f"{split}_img0.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        (img_dir / f"{split}_img1.jpg").write_bytes(b"\x00" * 50)
        (lbl_dir / f"{split}_img1.txt").write_text("1 0.5 0.5 0.2 0.2\n")

    (tmp_path / "data.yaml").write_text("nc: 2\nnames: [car, bus]\n")
    return tmp_path


# ---------------------------------------------------------------------------
# parse_label_file
# ---------------------------------------------------------------------------

def test_parse_label_file_basic(tmp_path):
    lbl = tmp_path / "test.txt"
    lbl.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n2 0.7 0.7 0.05 0.05\n")
    assert parse_label_file(str(lbl)) == [0, 1, 2]


def test_parse_label_file_empty(tmp_path):
    lbl = tmp_path / "empty.txt"
    lbl.write_text("")
    assert parse_label_file(str(lbl)) == []


def test_parse_label_file_skips_bad_lines(tmp_path):
    lbl = tmp_path / "bad.txt"
    lbl.write_text("0 0.5 0.5 0.2 0.2\nbadline\n\n1 0.1 0.2 0.3 0.4\n")
    assert parse_label_file(str(lbl)) == [0, 1]


# ---------------------------------------------------------------------------
# filter_and_remap_label
# ---------------------------------------------------------------------------

def test_filter_and_remap_label_keeps_selected(tmp_path):
    lbl = tmp_path / "lbl.txt"
    lbl.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n2 0.7 0.7 0.05 0.05\n")
    result = filter_and_remap_label(str(lbl), {0, 2}, {0: 0, 2: 1})
    lines = result.strip().split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("0 ")
    assert lines[1].startswith("1 ")  # remapped from 2 → 1


def test_filter_and_remap_label_no_remap(tmp_path):
    lbl = tmp_path / "lbl.txt"
    lbl.write_text("3 0.5 0.5 0.2 0.2\n5 0.3 0.3 0.1 0.1\n")
    result = filter_and_remap_label(str(lbl), {3, 5}, {3: 3, 5: 5})
    assert "3 " in result
    assert "5 " in result


def test_filter_and_remap_label_removes_unselected(tmp_path):
    lbl = tmp_path / "lbl.txt"
    lbl.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    result = filter_and_remap_label(str(lbl), {0}, {0: 0})
    assert "0 " in result
    assert "1 " not in result


# ---------------------------------------------------------------------------
# scan_dataset
# ---------------------------------------------------------------------------

def test_scan_dataset_flat(flat_dataset):
    data = scan_dataset(str(flat_dataset), "")
    # class 0 appears in img0 and img2
    assert 0 in data
    assert len(data[0]) == 2
    # class 1 appears in img1 and img2
    assert 1 in data
    assert len(data[1]) == 2


def test_scan_dataset_split(split_dataset):
    data = scan_dataset(str(split_dataset), "train")
    assert 0 in data
    assert 1 in data
    for img_path in data[0]:
        assert "train" in img_path


def test_scan_dataset_missing_dir():
    result = scan_dataset("/nonexistent/path", "train")
    assert result == {}


# ---------------------------------------------------------------------------
# find_splits
# ---------------------------------------------------------------------------

def test_find_splits_with_subdirs(split_dataset):
    splits = find_splits(str(split_dataset))
    assert "train" in splits
    assert "val" in splits


def test_find_splits_flat(flat_dataset):
    splits = find_splits(str(flat_dataset))
    # Flat dataset has images/ but no subdirs, so returns [""]
    assert splits == [""]


# ---------------------------------------------------------------------------
# load_class_names
# ---------------------------------------------------------------------------

def test_load_class_names(flat_dataset):
    names = load_class_names(str(flat_dataset))
    assert names == ["cat", "dog"]


def test_load_class_names_missing_yaml(tmp_path):
    names = load_class_names(str(tmp_path))
    assert names == []


# ---------------------------------------------------------------------------
# create_subset (integration)
# ---------------------------------------------------------------------------

def test_create_subset_basic(flat_dataset, tmp_path):
    output_dir = tmp_path / "subset"
    random.seed(0)
    summary = create_subset(
        dataset_dir=str(flat_dataset),
        split="",
        selected_class_ids=[0],
        max_per_class=10,
        output_dir=str(output_dir),
        class_names=["cat", "dog"],
        remap_ids=True,
    )
    assert "cat" in summary
    assert summary["cat"] == 2  # img0 and img2 have class 0

    # Output should have images and labels
    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    assert out_images.is_dir()
    assert out_labels.is_dir()

    copied_images = list(out_images.iterdir())
    assert len(copied_images) == 2


def test_create_subset_max_per_class(flat_dataset, tmp_path):
    output_dir = tmp_path / "subset"
    random.seed(0)
    summary = create_subset(
        dataset_dir=str(flat_dataset),
        split="",
        selected_class_ids=[0],
        max_per_class=1,  # Only 1 image allowed
        output_dir=str(output_dir),
        class_names=["cat", "dog"],
        remap_ids=True,
    )
    assert summary["cat"] == 1
    out_images = output_dir / "images"
    assert len(list(out_images.iterdir())) == 1


def test_create_subset_multi_class(flat_dataset, tmp_path):
    output_dir = tmp_path / "subset"
    random.seed(0)
    summary = create_subset(
        dataset_dir=str(flat_dataset),
        split="",
        selected_class_ids=[0, 1],
        max_per_class=10,
        output_dir=str(output_dir),
        class_names=["cat", "dog"],
        remap_ids=True,
    )
    assert "cat" in summary
    assert "dog" in summary
    # All 3 images should be included
    out_images = output_dir / "images"
    assert len(list(out_images.iterdir())) == 3


def test_create_subset_label_filtering(flat_dataset, tmp_path):
    """Label files in subset should only contain selected class annotations."""
    output_dir = tmp_path / "subset"
    random.seed(0)
    create_subset(
        dataset_dir=str(flat_dataset),
        split="",
        selected_class_ids=[0],  # Only cat
        max_per_class=10,
        output_dir=str(output_dir),
        class_names=["cat", "dog"],
        remap_ids=False,
    )
    # img2 has both class 0 and 1; after filtering only class 0 should remain
    label_path = output_dir / "labels" / "img2.txt"
    if label_path.exists():
        with open(label_path) as f:
            content = f.read()
        lines = [ln.strip() for ln in content.strip().split("\n") if ln.strip()]
        for line in lines:
            cls_id = int(line.split()[0])
            assert cls_id == 0, f"Unexpected class id {cls_id} in filtered label"


def test_create_subset_remap_ids(flat_dataset, tmp_path):
    """When remap_ids=True, class IDs in output labels should start from 0."""
    output_dir = tmp_path / "subset"
    random.seed(0)
    create_subset(
        dataset_dir=str(flat_dataset),
        split="",
        selected_class_ids=[1],  # Only dog (class id 1)
        max_per_class=10,
        output_dir=str(output_dir),
        class_names=["cat", "dog"],
        remap_ids=True,  # Should remap 1 → 0
    )
    out_labels = output_dir / "labels"
    for lbl_file in out_labels.iterdir():
        with open(lbl_file) as f:
            content = f.read().strip()
        if content:
            for line in content.split("\n"):
                cls_id = int(line.split()[0])
                assert cls_id == 0, f"Expected remapped id 0, got {cls_id}"


def test_create_subset_data_yaml_written(flat_dataset, tmp_path):
    output_dir = tmp_path / "subset"
    create_subset(
        dataset_dir=str(flat_dataset),
        split="",
        selected_class_ids=[0],
        max_per_class=10,
        output_dir=str(output_dir),
        class_names=["cat", "dog"],
        remap_ids=True,
    )
    assert (output_dir / "data.yaml").exists()


def test_create_subset_with_split(split_dataset, tmp_path):
    output_dir = tmp_path / "subset"
    random.seed(0)
    summary = create_subset(
        dataset_dir=str(split_dataset),
        split="train",
        selected_class_ids=[0],
        max_per_class=10,
        output_dir=str(output_dir),
        class_names=["car", "bus"],
        remap_ids=True,
    )
    assert "car" in summary
    out_images = output_dir / "images" / "train"
    assert out_images.is_dir()
    assert len(list(out_images.iterdir())) >= 1


# ---------------------------------------------------------------------------
# check_dataset
# ---------------------------------------------------------------------------

def test_check_dataset_clean(flat_dataset):
    """A fully-paired dataset should report no issues."""
    result = check_dataset(str(flat_dataset), "")
    assert result["missing_labels"] == []
    assert result["orphan_labels"] == []


def test_check_dataset_missing_label(tmp_path):
    """Image without a matching label file is reported as missing_labels."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    (images_dir / "img0.jpg").write_bytes(b"\x00" * 10)
    # No label file for img0.jpg

    result = check_dataset(str(tmp_path), "")
    assert len(result["missing_labels"]) == 1
    assert "img0.jpg" in result["missing_labels"][0]
    assert result["orphan_labels"] == []


def test_check_dataset_orphan_label(tmp_path):
    """Label file without a matching image is reported as orphan_labels."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    (labels_dir / "ghost.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    # No image for ghost.txt

    result = check_dataset(str(tmp_path), "")
    assert result["missing_labels"] == []
    assert len(result["orphan_labels"]) == 1
    assert "ghost.txt" in result["orphan_labels"][0]


def test_check_dataset_split(split_dataset):
    """Split layout check should only look inside the given split folder."""
    result = check_dataset(str(split_dataset), "train")
    assert result["missing_labels"] == []
    assert result["orphan_labels"] == []


def test_check_dataset_missing_dir(tmp_path):
    """Non-existent dataset dir should return empty lists, not raise."""
    result = check_dataset(str(tmp_path / "does_not_exist"), "")
    assert result["missing_labels"] == []
    assert result["orphan_labels"] == []


def test_check_dataset_uppercase_extension_not_orphan(tmp_path):
    """Image with uppercase extension (e.g. .JPG) must not be reported as orphan."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    # Image has uppercase extension – common when files come from Windows cameras
    (images_dir / "001.JPG").write_bytes(b"\x00" * 10)
    (labels_dir / "001.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    result = check_dataset(str(tmp_path), "")
    assert result["missing_labels"] == []
    assert result["orphan_labels"] == []


def test_check_dataset_uppercase_extension_split_not_orphan(tmp_path):
    """Same as above but for the split (non-recursive) code path."""
    images_dir = tmp_path / "images" / "train"
    labels_dir = tmp_path / "labels" / "train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (images_dir / "frame_001.PNG").write_bytes(b"\x00" * 10)
    (labels_dir / "frame_001.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    result = check_dataset(str(tmp_path), "train")
    assert result["missing_labels"] == []
    assert result["orphan_labels"] == []


def test_check_dataset_zone_identifier_ignored(tmp_path):
    """Zone.Identifier metadata files must be silently skipped."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    (images_dir / "cat.jpg").write_bytes(b"\x00" * 10)
    (labels_dir / "cat.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    # Windows Zone.Identifier sidecar files must not affect results
    (images_dir / "cat.jpg:Zone.Identifier").write_bytes(b"")
    (labels_dir / "cat.txt:Zone.Identifier").write_bytes(b"")

    result = check_dataset(str(tmp_path), "")
    assert result["missing_labels"] == []
    assert result["orphan_labels"] == []


# ---------------------------------------------------------------------------
# fix_missing_labels
# ---------------------------------------------------------------------------

def test_fix_missing_labels_creates_files(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    img = images_dir / "img0.jpg"
    img.write_bytes(b"\x00" * 10)

    missing = [str(img)]
    count = fix_missing_labels(missing, str(tmp_path), "")
    assert count == 1
    assert (labels_dir / "img0.txt").exists()
    assert (labels_dir / "img0.txt").read_text() == ""


def test_fix_missing_labels_skips_existing(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    img = images_dir / "img0.jpg"
    img.write_bytes(b"\x00" * 10)
    existing_lbl = labels_dir / "img0.txt"
    existing_lbl.write_text("0 0.5 0.5 0.1 0.1\n")

    count = fix_missing_labels([str(img)], str(tmp_path), "")
    assert count == 0
    # Existing label must remain untouched
    assert existing_lbl.read_text() == "0 0.5 0.5 0.1 0.1\n"


# ---------------------------------------------------------------------------
# fix_orphan_labels
# ---------------------------------------------------------------------------

def test_fix_orphan_labels_deletes_files(tmp_path):
    lbl = tmp_path / "ghost.txt"
    lbl.write_text("0 0.5 0.5 0.1 0.1\n")
    count = fix_orphan_labels([str(lbl)])
    assert count == 1
    assert not lbl.exists()


def test_fix_orphan_labels_ignores_missing(tmp_path):
    """Passing a non-existent path should not raise and return count 0."""
    count = fix_orphan_labels([str(tmp_path / "nonexistent.txt")])
    assert count == 0


# ---------------------------------------------------------------------------
# load_yaml_file
# ---------------------------------------------------------------------------

def test_load_yaml_file_list_names(tmp_path):
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text("nc: 3\nnames: [cat, dog, bird]\n")
    result = load_yaml_file(str(yaml_file))
    assert result["nc"] == 3
    assert result["names"] == ["cat", "dog", "bird"]


def test_load_yaml_file_dict_names(tmp_path):
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text("nc: 2\nnames:\n  0: cat\n  1: dog\n")
    result = load_yaml_file(str(yaml_file))
    assert result["nc"] == 2
    assert result["names"] == ["cat", "dog"]


def test_load_yaml_file_missing_file(tmp_path):
    result = load_yaml_file(str(tmp_path / "nonexistent.yaml"))
    assert result["names"] == []
    assert result["nc"] == 0
    assert result["raw"] == {}


def test_load_yaml_file_raw_contents(tmp_path):
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text("nc: 1\nnames: [person]\ntrain: images/train\n")
    result = load_yaml_file(str(yaml_file))
    assert "train" in result["raw"]
    assert result["raw"]["train"] == "images/train"


def test_load_yaml_file_nc_inferred_from_names(tmp_path):
    """When nc is absent, it should be inferred from the length of names."""
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text("names: [a, b, c]\n")
    result = load_yaml_file(str(yaml_file))
    assert result["nc"] == 3


def test_load_yaml_file_block_sequence_many_names(tmp_path):
    """YAML with names in block-sequence format (57 entries, as reported in bug)."""
    names_57 = [
        "MISSING", "C1_OK", "C1_NOTOK", "C2_OK", "C2_NOTOK", "C3_OK", "C3_NOTOK",
        "LED1_OK", "LED1_NOTOK", "LED10_OK", "LED10_NOTOK",
        "LED2_OK", "LED2_NOTOK", "LED3_OK", "LED3_NOTOK",
        "LED4_OK", "LED4_NOTOK", "LED5_OK", "LED5_NOTOK",
        "LED6_OK", "LED6_NOTOK", "LED7_OK", "LED7_NOTOK",
        "LED8_OK", "LED8_NOTOK", "LED9_OK", "LED9_NOTOK",
        "R10_OK", "R10_NOTOK", "R11_OK", "R11_NOTOK",
        "R12_OK", "R12_NOTOK", "R13_OK", "R13_NOTOK",
        "R2_OK", "R2_NOTOK", "R3_OK", "R3_NOTOK",
        "R4_OK", "R4_NOTOK", "R5_OK", "R5_NOTOK",
        "R6_OK", "R6_NOTOK", "R7_OK", "R7_NOTOK",
        "R8_OK", "R8_NOTOK", "R9_OK", "R9_NOTOK",
        "RP1_OK", "RP1_NOTOK", "U1_OK", "U1_NOTOK", "U2_OK", "U2_NOTOK",
    ]
    names_block = "\n".join(f"- {n}" for n in names_57)
    yaml_content = (
        "train: my_dataset/images/train\n"
        "val: my_dataset/images/val\n"
        "nc: 57\n"
        "names:\n"
        f"{names_block}\n"
    )
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text(yaml_content)

    result = load_yaml_file(str(yaml_file))
    assert result["nc"] == 57, f"Expected nc=57, got {result['nc']}"
    assert len(result["names"]) == 57, f"Expected 57 names, got {len(result['names'])}"
    assert result["names"][0] == "MISSING"
    assert result["names"][-1] == "U2_NOTOK"

    # Also verify load_class_names reads the same data
    names = load_class_names(str(tmp_path))
    assert len(names) == 57
    assert names[0] == "MISSING"
    assert names[-1] == "U2_NOTOK"


def test_write_data_yaml_uses_block_sequence(tmp_path):
    """_write_data_yaml must write names as a YAML block sequence, not Python repr."""
    kept = {0: "cat", 1: "dog", 2: "bird"}
    _write_data_yaml(str(tmp_path), str(tmp_path / "out"), kept)

    yaml_path = tmp_path / "out" / "data.yaml"
    content = yaml_path.read_text()

    # Each name must appear on its own line prefixed by "- "
    lines = content.splitlines()
    name_lines = [ln for ln in lines if ln.startswith("- ")]
    assert len(name_lines) == 3
    assert any("cat" in ln for ln in name_lines)
    assert any("dog" in ln for ln in name_lines)
    assert any("bird" in ln for ln in name_lines)

    # Round-trip: load_yaml_file must parse it back correctly
    result = load_yaml_file(str(yaml_path))
    assert result["nc"] == 3
    assert result["names"] == ["cat", "dog", "bird"]


def test_write_data_yaml_escapes_special_chars(tmp_path):
    """Names with YAML-special characters must be quoted in the output."""
    kept = {0: "plain", 1: ":special", 2: "has: colon", 3: "backslash\\end"}
    _write_data_yaml(str(tmp_path), str(tmp_path / "out"), kept)

    yaml_path = tmp_path / "out" / "data.yaml"
    content = yaml_path.read_text()

    # Round-trip must preserve names unchanged
    result = load_yaml_file(str(yaml_path))
    assert result["nc"] == 4
    assert result["names"] == ["plain", ":special", "has: colon", "backslash\\end"]


# ---------------------------------------------------------------------------
# yaml_from_json
# ---------------------------------------------------------------------------

def test_yaml_from_json_coco_format(tmp_path):
    """COCO annotation JSON with 'categories' is correctly parsed."""
    json_file = tmp_path / "coco.json"
    json_file.write_text(json.dumps({
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
            {"id": 3, "name": "bird"},
        ]
    }))
    result = yaml_from_json(str(json_file))
    assert result == {1: "cat", 2: "dog", 3: "bird"}


def test_yaml_from_json_names_list(tmp_path):
    """JSON with a 'names' list is correctly parsed."""
    json_file = tmp_path / "names.json"
    json_file.write_text(json.dumps({"names": ["cat", "dog", "bird"]}))
    result = yaml_from_json(str(json_file))
    assert result == {0: "cat", 1: "dog", 2: "bird"}


def test_yaml_from_json_names_dict(tmp_path):
    """JSON with a 'names' dict is correctly parsed."""
    json_file = tmp_path / "names_dict.json"
    json_file.write_text(json.dumps({"names": {"0": "cat", "1": "dog"}}))
    result = yaml_from_json(str(json_file))
    assert result == {0: "cat", 1: "dog"}


def test_yaml_from_json_plain_list(tmp_path):
    """A plain JSON list of strings is treated as ordered class names."""
    json_file = tmp_path / "list.json"
    json_file.write_text(json.dumps(["alpha", "beta", "gamma"]))
    result = yaml_from_json(str(json_file))
    assert result == {0: "alpha", 1: "beta", 2: "gamma"}


def test_yaml_from_json_invalid_raises(tmp_path):
    """JSON that doesn't match any supported format raises ValueError."""
    json_file = tmp_path / "bad.json"
    json_file.write_text(json.dumps({"foo": "bar"}))
    with pytest.raises(ValueError):
        yaml_from_json(str(json_file))


def test_yaml_from_json_roundtrip(tmp_path):
    """yaml_from_json + _write_data_yaml produces a valid, parseable data.yaml."""
    json_file = tmp_path / "classes.json"
    json_file.write_text(json.dumps({"names": ["person", "car", "truck"]}))

    class_map = yaml_from_json(str(json_file))
    out_dir = tmp_path / "out"
    _write_data_yaml(str(tmp_path), str(out_dir), class_map)

    yaml_path = out_dir / "data.yaml"
    assert yaml_path.exists()
    result = load_yaml_file(str(yaml_path))
    assert result["nc"] == 3
    assert result["names"] == ["person", "car", "truck"]



# ---------------------------------------------------------------------------
# remap_labels
# ---------------------------------------------------------------------------

def _make_label_dir(tmp_path, files: dict[str, str]) -> str:
    """Create a labels directory with the given {filename: content} mapping."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for name, content in files.items():
        (labels_dir / name).write_text(content)
    return str(labels_dir)


def test_remap_labels_swap_inplace(tmp_path):
    """Swap class 0 ↔ 1 modifies files in-place."""
    labels_dir = _make_label_dir(tmp_path, {
        "a.txt": "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n",
        "b.txt": "1 0.7 0.7 0.1 0.1\n",
    })
    written = remap_labels(labels_dir, {0: 1, 1: 0})
    assert written == 2  # both files changed
    assert parse_label_file(os.path.join(labels_dir, "a.txt")) == [1, 0]
    assert parse_label_file(os.path.join(labels_dir, "b.txt")) == [0]


def test_remap_labels_to_output_dir(tmp_path):
    """Writing to a separate output_dir leaves the source untouched."""
    labels_dir = _make_label_dir(tmp_path, {
        "a.txt": "2 0.5 0.5 0.2 0.2\n",
    })
    out_dir = str(tmp_path / "out")
    remap_labels(labels_dir, {2: 0}, output_dir=out_dir)
    # source unchanged
    assert parse_label_file(os.path.join(labels_dir, "a.txt")) == [2]
    # output has remapped class
    assert parse_label_file(os.path.join(out_dir, "a.txt")) == [0]


def test_remap_labels_no_change_skipped_inplace(tmp_path):
    """Files with no matching classes are not rewritten when in-place."""
    labels_dir = _make_label_dir(tmp_path, {
        "unchanged.txt": "3 0.5 0.5 0.2 0.2\n",
    })
    written = remap_labels(labels_dir, {0: 1})  # class 3 is not remapped
    assert written == 0


def test_remap_labels_merge_classes(tmp_path):
    """Mapping two classes to the same ID effectively merges them."""
    labels_dir = _make_label_dir(tmp_path, {
        "a.txt": "1 0.1 0.1 0.1 0.1\n3 0.5 0.5 0.1 0.1\n",
    })
    remap_labels(labels_dir, {1: 0, 3: 0})
    assert parse_label_file(os.path.join(labels_dir, "a.txt")) == [0, 0]


def test_remap_labels_bbox_coords_preserved(tmp_path):
    """Remapping only changes the class column; bounding-box coords are intact."""
    labels_dir = _make_label_dir(tmp_path, {
        "a.txt": "2 0.12 0.34 0.56 0.78\n",
    })
    remap_labels(labels_dir, {2: 5})
    content = open(os.path.join(labels_dir, "a.txt")).read().strip()
    assert content == "5 0.12 0.34 0.56 0.78"


def test_remap_labels_empty_file(tmp_path):
    """Empty label files are handled without errors."""
    labels_dir = _make_label_dir(tmp_path, {"empty.txt": ""})
    written = remap_labels(labels_dir, {0: 1})
    assert written == 0  # nothing to remap, nothing written


# ---------------------------------------------------------------------------
# remap_yaml_classes
# ---------------------------------------------------------------------------

def _make_yaml(tmp_path, name: str, content: str) -> str:
    """Write a YAML file in tmp_path and return its path."""
    p = tmp_path / name
    p.write_text(content)
    return str(p)


def test_remap_yaml_classes_swap_inplace(tmp_path):
    """Swapping classes 0 and 1 in the YAML exchanges their names."""
    yaml_path = _make_yaml(tmp_path, "data.yaml",
                           "nc: 3\nnames:\n- cat\n- dog\n- bird\n")
    remap_yaml_classes(yaml_path, {0: 1, 1: 0}, yaml_path)
    data = load_yaml_file(yaml_path)
    assert data["names"] == ["dog", "cat", "bird"]
    assert data["nc"] == 3


def test_remap_yaml_classes_free_remap_to_output(tmp_path):
    """Free renumber writes updated YAML to a separate output path."""
    yaml_path = _make_yaml(tmp_path, "data.yaml",
                           "nc: 3\nnames:\n- cat\n- dog\n- bird\n")
    out_yaml = str(tmp_path / "out" / "data.yaml")
    remap_yaml_classes(yaml_path, {1: 3, 2: 1}, out_yaml)

    # Source unchanged
    src_data = load_yaml_file(yaml_path)
    assert src_data["names"] == ["cat", "dog", "bird"]

    # Output has remapped classes
    out_data = load_yaml_file(out_yaml)
    # old 0->0:cat, old 1->3:dog, old 2->1:bird  => {0:cat,1:bird,3:dog}
    assert out_data["names"][0] == "cat"
    assert out_data["names"][1] == "bird"
    assert out_data["nc"] == 3


def test_remap_yaml_classes_preserves_train_val_paths(tmp_path):
    """train/val/test paths from the original YAML are kept in the output."""
    yaml_path = _make_yaml(tmp_path, "data.yaml",
                           "nc: 2\nnames:\n- a\n- b\ntrain: images/train\nval: images/val\n")
    remap_yaml_classes(yaml_path, {0: 1, 1: 0}, yaml_path)
    data = load_yaml_file(yaml_path)
    assert data["raw"].get("train") == "images/train"
    assert data["raw"].get("val") == "images/val"


def test_remap_yaml_classes_missing_file_is_noop(tmp_path):
    """If the YAML file does not exist, remap_yaml_classes does nothing."""
    out_path = str(tmp_path / "nonexistent_out.yaml")
    # Should not raise
    remap_yaml_classes(str(tmp_path / "missing.yaml"), {0: 1}, out_path)
    assert not os.path.isfile(out_path)



def _make_flat_dataset(tmp_path, n_images: int = 10) -> None:
    """Create a flat dataset with *n_images* images and matching label files."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for i in range(n_images):
        (images_dir / f"img{i:03d}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
        (labels_dir / f"img{i:03d}.txt").write_text(f"{i % 3} 0.5 0.5 0.2 0.2\n")
    (tmp_path / "data.yaml").write_text("nc: 3\nnames: [a, b, c]\n")


def test_split_dataset_basic(tmp_path):
    """split_dataset creates train/val structure with correct counts."""
    src = tmp_path / "src"
    src.mkdir()
    _make_flat_dataset(src, n_images=10)

    out = tmp_path / "out"
    counts = do_split_dataset(str(src), "", str(out), train_pct=80, seed=0)

    assert counts["train"] + counts["val"] == 10
    assert counts["train"] == 8
    assert counts["val"] == 2

    # Verify directory structure
    assert (out / "images" / "train").is_dir()
    assert (out / "images" / "val").is_dir()
    assert (out / "labels" / "train").is_dir()
    assert (out / "labels" / "val").is_dir()

    train_imgs = list((out / "images" / "train").iterdir())
    val_imgs = list((out / "images" / "val").iterdir())
    assert len(train_imgs) == 8
    assert len(val_imgs) == 2


def test_split_dataset_total_images_preserved(tmp_path):
    """All source images end up in train or val, no image is lost."""
    src = tmp_path / "src"
    src.mkdir()
    n = 20
    _make_flat_dataset(src, n_images=n)

    out = tmp_path / "out"
    counts = do_split_dataset(str(src), "", str(out), train_pct=70, seed=1)

    total_out = (
        len(list((out / "images" / "train").iterdir()))
        + len(list((out / "images" / "val").iterdir()))
    )
    assert total_out == n
    assert counts["train"] + counts["val"] == n


def test_split_dataset_labels_copied(tmp_path):
    """Each image in train/val has a corresponding label file copied."""
    src = tmp_path / "src"
    src.mkdir()
    _make_flat_dataset(src, n_images=6)

    out = tmp_path / "out"
    do_split_dataset(str(src), "", str(out), train_pct=50, seed=42)

    for subset in ("train", "val"):
        img_dir = out / "images" / subset
        lbl_dir = out / "labels" / subset
        for img_file in img_dir.iterdir():
            stem = img_file.stem
            assert (lbl_dir / f"{stem}.txt").is_file(), (
                f"Missing label for {img_file.name} in {subset}"
            )


def test_split_dataset_reproducible(tmp_path):
    """Same seed produces the same split."""
    src = tmp_path / "src"
    src.mkdir()
    _make_flat_dataset(src, n_images=10)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    do_split_dataset(str(src), "", str(out1), train_pct=80, seed=7)
    do_split_dataset(str(src), "", str(out2), train_pct=80, seed=7)

    names1 = sorted(f.name for f in (out1 / "images" / "train").iterdir())
    names2 = sorted(f.name for f in (out2 / "images" / "train").iterdir())
    assert names1 == names2


def test_split_dataset_yaml_copied(tmp_path):
    """data.yaml from source is copied to the output directory."""
    src = tmp_path / "src"
    src.mkdir()
    _make_flat_dataset(src, n_images=4)

    out = tmp_path / "out"
    do_split_dataset(str(src), "", str(out), train_pct=75, seed=0)

    assert (out / "data.yaml").is_file()


def test_split_dataset_from_split_source(tmp_path):
    """split_dataset works when source is a named split (e.g. 'train')."""
    src = tmp_path / "src"
    src.mkdir()
    img_dir = src / "images" / "train"
    lbl_dir = src / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in range(8):
        (img_dir / f"img{i}.jpg").write_bytes(b"\x00" * 10)
        (lbl_dir / f"img{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.1 0.1\n")

    out = tmp_path / "out"
    counts = do_split_dataset(str(src), "train", str(out), train_pct=75, seed=0)

    assert counts["train"] + counts["val"] == 8
    assert (out / "images" / "train").is_dir()
    assert (out / "images" / "val").is_dir()


def test_split_dataset_missing_images_dir(tmp_path):
    """split_dataset raises FileNotFoundError when the images directory is missing."""
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"

    with pytest.raises(FileNotFoundError):
        do_split_dataset(str(src), "", str(out), train_pct=80)


def test_split_dataset_no_images(tmp_path):
    """split_dataset raises ValueError when the images directory is empty."""
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    out = tmp_path / "out"

    with pytest.raises(ValueError):
        do_split_dataset(str(src), "", str(out), train_pct=80)
