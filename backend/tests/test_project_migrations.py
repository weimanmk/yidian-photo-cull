import json

from photocull.project_migrations import migrate_project_payload
from photocull.project_store import ProjectStore


def v1_payload() -> dict:
    return {
        "results": {
            "project_id": "legacy-project",
            "photos": [
                {
                    "id": "photo-a",
                    "filename": "photo-a.jpg",
                    "group_id": "group-1",
                    "stars": 3,
                    "category": "selected",
                }
            ],
            "groups": [{"id": "group-1", "photo_ids": ["photo-a"]}],
            "summary": {"total": 1, "selected": 1},
        },
        "files": {"photo-a": "C:/photos/photo-a.jpg"},
    }


def test_v1_project_is_marked_rescan_required_without_semantic_fields() -> None:
    original = v1_payload()

    migrated = migrate_project_payload(original)

    assert migrated["results"]["schema_version"] == 2
    assert migrated["results"]["rating_migration_status"] == "rescan_required"
    assert migrated["results"]["lightroom_ready"] is False
    assert migrated["results"]["photos"][0]["rating_origin"] == "legacy"
    assert migrated["results"]["photos"][0]["rating_tier"] == "legacy"
    assert migrated["results"]["photos"][0]["stars"] == 3
    assert "schema_version" not in original["results"]


def test_project_store_rewrites_migrated_payload_once(tmp_path) -> None:
    store = ProjectStore(tmp_path)
    path = tmp_path / "legacy-project.json"
    path.write_text(json.dumps(v1_payload(), ensure_ascii=False), encoding="utf-8")

    first_results, first_files = store.load("legacy-project")
    first_serialized = path.read_text(encoding="utf-8")
    second_results, second_files = store.load("legacy-project")

    assert first_results == second_results
    assert first_files == second_files
    assert path.read_text(encoding="utf-8") == first_serialized
    assert second_results["rating_migration_status"] == "rescan_required"
