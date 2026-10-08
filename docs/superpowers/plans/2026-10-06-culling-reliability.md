# Culling reliability implementation plan

> For agentic workers: use test-driven-development and independent domain implementation, then review the integrated branch.

**Goal:** Implement the three reliability improvements approved in the preceding repository review.

**Architecture:** Keep the existing offline analysis pipeline and manual rating authority. Introduce explicit rating eligibility, globally coherent duplicate layers, quality-aware deterministic identity assignment, and persistent manual identity corrections. Audit final star selections independently of old category labels.

**Tech Stack:** Python 3.12, NumPy, FastAPI, SQLite, Electron, React and TypeScript.

**Spec:** The user's approved three-item review immediately preceding “你去实现”.

## Global constraints

- Do not change or delete source photos, upload images, retrain weights or claim a new real-image benchmark without data.
- Preserve manual locked ratings and explicit low-quality coverage exceptions with review warnings.
- Keep CPU/offline operation and existing project compatibility. No new model downloads or production dependencies required.
- Keep evaluation model hashes factual; the historical report does not validate the changed pipeline.

## Review focus

- All-bad groups must not create automatic 3-star photos, including frozen-model fallback.
- Manual 0-star and 3-star locks must survive scans and dominate duplicate/coverage decisions.
- Cross-group duplicates and transitive chains must not produce silent leaks or false merges.
- Large group photos, ambiguous low-quality faces, reordered inputs and manual identity corrections must preserve reliable identities.
- Existing saved projects and final star selections must retain correct coverage, provenance and audit scope.

## Task 1: Quality eligibility

Files: rating_policy.py, rating_types.py if needed, test_rating_policy.py; result inspector labels.
- [x] Write and run failing tests for all-bad groups, severe issues during budget fill, flagged coverage exceptions, and manual locks.
- [x] Separate decodable, primary eligible, ordinary reserve eligible and coverage-only candidates; allow empty primary groups.
- [x] Deduplicate group seeds using global cluster IDs, with manual locks taking precedence.
- [x] Run policy, coverage and pipeline tests.

## Task 2: Global duplicate layers and evaluation

Files: near_duplicates.py, duplicate and semantic audit scripts, evaluation tests, REAL_SET_REPORT.md and ALGORITHM.md sections owned by this task.
Interface: build_duplicate_layers(groups) retains DuplicateLayers fields and returns global strict IDs and existing local beat IDs.
- [x] Write/run failing cross-group and transitive conflict tests.
- [x] Implement conservative global candidate retrieval and complete cluster compatibility; constrain changed moments without missing identical copies.
- [x] Audit minimum-stars selections, add independent labeled pair evaluation and provenance (commit, model SHA-256, parameters).
- [x] Mark historical report provenance mismatch without fabricating new real-photo metrics; run targeted tests.

## Task 3: Detection and identity stability

Files: face_engine.py, face_quality.py, identity.py and corresponding tests.
Interface: FaceEngine.analyze and IdentityClusterer.assign remain compatible; unknown identity may remain None.
- [x] Write/run failing large-group, ambiguous-face and input-order tests.
- [x] Remove silent 24/12 face truncation, add conservative larger-resolution detection retry for crowded/small-face scenes.
- [x] Seed identity clusters deterministically with quality evidence, prevent ambiguous weak faces from contaminating centroids, retain same-photo cannot-link.
- [x] Run face quality, identity, quality and grouping tests.

## Task 4: Persistent identity correction

Files: new identity correction helper/store, scanner.py, api.py, schemas.py, frontend API/types/results components and tests.
Interface: project-scoped correction endpoint returns updated scan results; face references use photo ID and face ID; merge/split corrections survive reload and rescan.
- [x] Write/run failing correction persistence, merge conflict and split tests.
- [x] Add validated merge/split commands and persisted face-level overrides; apply after automatic clustering and before grouping on rescan.
- [x] Add accessible result-workspace UI for merge and splitting selected faces; preserve stars and clearly request a rescan for reranking if needed.
- [x] Run endpoint/service tests, UI tests and typecheck.

## Task 5: Integration

- [x] Run backend suite, frontend suite, typecheck, build and UI contract checks.
- [x] Review diff independently, fix evidenced defects, and rerun affected gates.
- [x] Commit on fix/culling-reliability and provide reviewable branch/PR if remote push is available; no merge or release.

## Progress and rulings

- User explicitly requested implementation of the previously presented changes; no repeat design approval is needed.
- Work occurs in the clean task-specific clone on a feature branch. Additional worktree is unnecessary.
- Participant roster/reference-face enrollment is conditional future functionality, not a prerequisite for these anonymous identity fixes; no claim of roster-level coverage will be added.

## Final validation

Implementation and independent review are complete. Backend: 276 passed, 3 skipped, 11 failures (9 unavailable model integrations; 2 Lightroom fixture hash failures reproduced at baseline). Frontend: 3 passed; TypeScript build and UI contract passed. No new real-image accuracy claim. See `docs/CULLING_RELIABILITY.md` for evidence and limitations. Branch delivery is a reviewable draft, with no merge or release.
