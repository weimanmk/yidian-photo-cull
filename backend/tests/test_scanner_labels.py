from photocull.project_store import ProjectStore
from photocull.scanner import ScannerService


def test_manual_label_keeps_group_best_ids_in_sync(tmp_path) -> None:
    scanner = ScannerService(ProjectStore(tmp_path))
    scanner._results = {
        "project_id": "manual-labels",
        "photos": [
            {"id": "a", "group_id": "group-1", "category": "selected", "is_best_pick": True, "stars": 4},
            {"id": "b", "group_id": "group-1", "category": "duplicate", "is_best_pick": False, "stars": 0},
        ],
        "groups": [{"id": "group-1", "photo_ids": ["a", "b"], "best_photo_ids": ["a"]}],
        "summary": {"people": 1, "elapsed_seconds": 2.0},
    }

    assert scanner.label_photo("b", "selected", 3) is True
    assert scanner._results["groups"][0]["best_photo_ids"] == ["a", "b"]
    assert scanner._results["summary"]["selected"] == 2
    assert scanner._results["summary"]["coverage_protected"] == 0

    assert scanner.label_photo("a", "duplicate", 0) is True
    assert scanner._results["groups"][0]["best_photo_ids"] == ["b"]
    saved, _ = scanner.projects.load("manual-labels")
    assert saved["groups"][0]["best_photo_ids"] == ["b"]


def test_manual_rating_lock_is_persisted_and_updates_semantic_summary(tmp_path) -> None:
    scanner = ScannerService(ProjectStore(tmp_path))
    scanner._results = {
        "schema_version": 2,
        "rating_migration_status": "native",
        "lightroom_ready": True,
        "project_id": "manual-ratings",
        "photos": [
            {
                "id": "a",
                "group_id": "group-1",
                "category": "duplicate",
                "is_best_pick": False,
                "stars": 0,
                "rating_tier": "waste",
                "rating_origin": "ai",
                "rating_reason": "redundant_reject",
                "rating_locked": False,
                "coverage_keys": [],
            }
        ],
        "groups": [{"id": "group-1", "photo_ids": ["a"], "best_photo_ids": []}],
        "rating_policy": {"required_coverage_keys": 0, "unresolved_coverage_keys": 0},
        "summary": {"people": 0, "elapsed_seconds": 1.0},
    }

    assert scanner.rate_photo("a", 2, locked=True) is True

    photo = scanner._results["photos"][0]
    assert photo["stars"] == 2
    assert photo["rating_tier"] == "coverage"
    assert photo["rating_origin"] == "manual"
    assert photo["rating_reason"] == "manual_override"
    assert photo["rating_locked"] is True
    assert scanner._results["groups"][0]["best_photo_ids"] == ["a"]
    assert scanner._results["summary"]["stars_2"] == 1
    assert scanner._results["summary"]["selected"] == 1

    saved, _ = scanner.projects.load("manual-ratings")
    assert saved["photos"][0]["rating_locked"] is True
