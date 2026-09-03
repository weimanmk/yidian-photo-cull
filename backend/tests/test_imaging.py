from pathlib import Path

from PIL import Image

from photocull.imaging import discover_images, load_image, raw_companion_preview


def test_discover_images_prefers_raw_over_same_stem_jpeg(tmp_path: Path) -> None:
    (tmp_path / "DSC00001.ARW").touch()
    (tmp_path / "DSC00001.JPG").touch()
    (tmp_path / "DSC00002.JPG").touch()
    (tmp_path / "DSC00003.PNG").touch()

    discovered = discover_images(tmp_path, recursive=False)

    assert [path.name for path in discovered] == [
        "DSC00001.ARW",
        "DSC00002.JPG",
        "DSC00003.PNG",
    ]


def test_discover_images_only_pairs_files_in_same_directory(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    jpeg_dir = tmp_path / "jpeg"
    raw_dir.mkdir()
    jpeg_dir.mkdir()
    (raw_dir / "DSC00001.ARW").touch()
    (jpeg_dir / "DSC00001.JPG").touch()

    discovered = discover_images(tmp_path, recursive=True)

    assert {path.relative_to(tmp_path).as_posix() for path in discovered} == {
        "raw/DSC00001.ARW",
        "jpeg/DSC00001.JPG",
    }


def test_load_image_uses_full_size_raw_companion_jpeg(tmp_path: Path) -> None:
    raw_path = tmp_path / "DSC00001.ARW"
    jpeg_path = tmp_path / "DSC00001.JPG"
    raw_path.touch()
    Image.new("RGB", (12, 8), (10, 20, 30)).save(jpeg_path)

    assert raw_companion_preview(raw_path) == jpeg_path
    image = load_image(raw_path)

    assert image.size == (12, 8)
    assert image.getpixel((0, 0)) == (10, 20, 30)
