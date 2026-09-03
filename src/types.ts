export type ViewKey = 'home' | 'cull' | 'results' | 'settings'
export type ScanState = 'idle' | 'discovering' | 'analyzing' | 'identifying' | 'personalizing' | 'grouping' | 'ranking' | 'vlm_reviewing' | 'completed' | 'cancelled' | 'failed'
export type GroupingPreset = 'cautious' | 'balanced' | 'aggressive'
export type PhotoStars = 0 | 1 | 2 | 3
export type RatingTier = 'waste' | 'valuable' | 'coverage' | 'primary'
export type RatingOrigin = 'ai' | 'coverage' | 'manual' | 'legacy'
export type RatingReason =
  | 'primary_rank'
  | 'person_stage_gap'
  | 'person_stage_reserve'
  | 'unique_moment'
  | 'technical_reject'
  | 'redundant_reject'
  | 'manual_override'
  | 'legacy_score'

export interface SemanticRatingFields {
  stars: PhotoStars
  rating_tier: RatingTier
  rating_origin: RatingOrigin
  rating_reason: RatingReason | string
  rating_locked: boolean
  needs_review: boolean
  coverage_keys: string[]
}

export interface ModelStatus {
  available: boolean
  backend: string
  providers: string[]
  provider_source?: 'actual' | 'configured'
  cuda_preload_error?: string | null
  model_dir: string | null
  missing_models: string[]
  eye_model?: { available: boolean; name: string; path: string | null }
  expression_model?: { available: boolean; name: string; path: string | null; labels: string[] }
  landmark_3d_model?: { available: boolean; name: string; path: string | null; role: string }
  face_quality_model?: { available: boolean; name: string; path: string | null; role: string }
}

export interface HealthResponse {
  status: 'ok'
  version: string
  offline: boolean
  face_ai: ModelStatus
  body_ai?: {
    available: boolean
    backend: string
    providers?: string[]
    provider_source?: 'actual' | 'configured'
    cuda_preload_error?: string | null
    detector: { available: boolean; name: string; path: string | null; fallbacks: string[] }
    reid_model: { available: boolean; name: string; path: string | null; role: string }
    errors: string[]
  }
  pose_ai?: {
    available: boolean
    backend: string
    model: string
    path: string | null
    landmarks: number
    world_coordinates: boolean
    telemetry: boolean
    runtime: string
    error: string | null
  }
  depth_ai?: {
    available: boolean
    backend: string
    providers?: string[]
    provider_source?: 'actual' | 'configured'
    cuda_preload_error?: string | null
    model: string
    path: string | null
    relative_depth: boolean
    metric_depth: boolean
    local_only: boolean
    roles: string[]
    error: string | null
  }
  scene_ai?: ModelStatus
  preference_ai?: {
    available: boolean
    version: string
    ranking_strength: number
    blend_weight?: number
    selection_filter_enabled?: boolean
  }
  vlm_ai?: VlmRuntimeStatus
}

export interface VlmRuntimeStatus {
  enabled: boolean
  available: boolean
  configured: boolean
  running: boolean
  managed: boolean
  backend: string
  server_url: string
  model_id: string
  quantization: string
  context_size: number
  gpu_layers: number
  prompt_version: string
  error: string | null
}

export interface ScanStatus {
  status: ScanState
  phase: string
  message: string
  processed: number
  total: number
  progress: number
  current_file: string
  elapsed_seconds: number
  eta_seconds: number | null
  error: string | null
  project_id: string | null
  cache_hits?: number
  cache_misses?: number
}

export interface FaceResult {
  face_id: string
  person_id: string | null
  confidence: number
  bbox: [number, number, number, number]
  area_ratio: number
  eye_state: 'Open' | 'Partial' | 'Closed' | 'Unknown'
  open_probability: number | null
  sharpness: number
  profile: boolean
  smile_score: number
  high_res_sharpness?: number
  eye_sharpness?: number
  yaw?: number
  pitch?: number
  roll?: number
  occlusion_risk?: number
  expression?: string
  expression_confidence?: number | null
  expression_score?: number
  fiqa_score?: number | null
}

export interface BodyResult {
  bbox: [number, number, number, number]
  confidence: number
  area_ratio: number
  detector: string
  reid_available: boolean
}

export interface PoseResult {
  bbox: [number, number, number, number]
  detection_confidence: number
  presence_confidence: number
  area_ratio: number
  visibility: number
  foreground_score: number | null
  landmarks: [number, number, number][]
  model: string
  world_3d_available: boolean
}

export interface DepthResult {
  subject_depth: number | null
  background_depth: number | null
  foreground_separation: number
  subject_focus_score: number | null
  background_blur_score: number | null
  occlusion_risk: number
  subject_confidence: number
  model: string
}

export interface QualityMetrics {
  sharpness: number
  sharpness_score: number
  face_sharpness_score: number
  exposure_score: number
  composition_score: number
  eye_score: number
  contrast_score: number
  brightness: number
  highlight_clip: number
  shadow_clip: number
  tenengrad_score?: number
  motion_blur_score?: number
  subject_sharpness_score?: number
  depth_focus_score?: number
  depth_background_blur_score?: number
  depth_separation_score?: number
  depth_occlusion_risk?: number
  depth_subject_confidence?: number
  noise_score?: number
  eye_sharpness_score?: number
  face_quality_score?: number
  learned_face_quality_score?: number
  min_face_score?: number
  bad_face_count?: number
  expression_score?: number
  technical_score?: number
  group_ranking_score?: number
  preference_score?: number
  preference_threshold?: number
}

export type PhotoCategory = 'selected' | 'duplicate' | 'blurred' | 'closed_eyes' | 'exposure' | 'rejected'

export interface PhotoResult extends SemanticRatingFields {
  id: string
  filename: string
  relative_path: string
  width: number
  height: number
  capture_time: string | null
  thumbnail_url: string
  image_url: string
  score: number
  strict_duplicate_cluster_id: string
  beat_id: string
  category: PhotoCategory
  issues: string[]
  metrics: QualityMetrics
  faces: FaceResult[]
  bodies?: BodyResult[]
  poses?: PoseResult[]
  depth?: DepthResult | null
  person_ids: string[]
  group_id: string
  is_best_pick: boolean
  rank_in_group: number
  selection_reasons: string[]
  vlm_rank?: number | null
  vlm_confidence?: number | null
  vlm_reasons?: string[]
  stage_id: string
  stage_label: string
  coverage_protected: boolean
  coverage_person_ids: string[]
  coverage_original_category: PhotoCategory | null
}

export interface VlmGroupDecision {
  model_id: string
  prompt_version: string
  best_photo_id: string
  confidence: number
  best_reasons: string[]
  applied: boolean
  changed_winner: boolean
  cached?: boolean
  reason?: string
}

export interface PhotoGroup {
  id: string
  photo_ids: string[]
  best_photo_ids: string[]
  person_ids: string[]
  size: number
  confidence: number
  scene_reason: string
  vlm_decision?: VlmGroupDecision | null
  stage_id?: string
  stage_label?: string
  coverage_protected?: boolean
}

export interface CoverageStage {
  id: string
  label: string
  photo_count: number
  person_count: number
  start_time: string | null
  end_time: string | null
}

export interface CoverageReport {
  enabled: boolean
  stage_source: 'disabled' | 'folder' | 'time'
  window_minutes: number
  stages: CoverageStage[]
  eligible_people: number
  required_cells: number
  already_covered_cells: number
  protected_photos: number
  protected_cells: number
  unresolved_cells: number
}

export interface ScanSummary {
  total: number
  selected: number
  duplicates: number
  issues: number
  groups: number
  people: number
  coverage_protected: number
  coverage_stages: number
  coverage_required_cells: number
  coverage_unresolved_cells: number
  primary_duplicate_leaks: number
  primary: number
  coverage: number
  valuable: number
  waste: number
  stars_0: number
  stars_1: number
  stars_2: number
  stars_3: number
  elapsed_seconds: number
}

export interface ScanResults {
  schema_version?: 1 | 2
  rating_migration_status?: 'native' | 'migrated' | 'legacy'
  lightroom_ready?: boolean
  project_id: string
  project_name: string
  source_name: string
  created_at: string
  photos: PhotoResult[]
  groups: PhotoGroup[]
  coverage?: CoverageReport
  summary: ScanSummary
  engine?: {
    version: string
    coverage_guard?: CoverageReport
    vlm_ai?: VlmRuntimeStatus & {
      applied?: boolean
      candidate_groups?: number
      reviewed_groups?: number
      applied_groups?: number
      changed_winners?: number
      reason?: string
    }
  }
}

export interface LightroomPluginHeartbeat {
  schema_version: 1
  plugin_id: string
  plugin_version: string
  sdk_version: string
  timestamp: string
  running: boolean
  last_error: string | null
}

export interface LightroomBackendStatus {
  bridge_root: string
  plugin_heartbeat: LightroomPluginHeartbeat | null
  queue: {
    inbox: number
    processing: number
    outbox: number
    quarantine: number
  }
}

export interface LightroomReceiptCounts {
  total: number
  new: number
  update: number
  unchanged: number
  protected: number
  invalid: number
  catalog_added: number
  pending_rating: number
  verified: number
  rolled_back: number
}

export type LightroomOperationStatus =
  | 'created'
  | 'waiting_for_plugin'
  | 'preflighting'
  | 'awaiting_confirmation'
  | 'executing'
  | 'verifying'
  | 'pending_rating'
  | 'complete'
  | 'failed'
  | 'quarantined'
  | 'manual_recovery_required'
  | 'rollback_preflight'
  | 'rollback_awaiting_confirmation'
  | 'rolling_back'
  | 'rolled_back'

export interface LightroomOperation {
  schema_version: 1
  id: string
  project_id: string
  project_name: string
  status: LightroomOperationStatus
  created_at: string
  updated_at: string
  plan_hash: string
  preflight_request_id: string
  execute_request_id: string | null
  item_count: number
  can_execute: boolean
  counts: LightroomReceiptCounts | null
  catalog_name: string | null
  catalog_identity_hash: string | null
  baseline_hash: string | null
  error_code: string | null
  error_message: string | null
}

export interface ExportPlan {
  operation_id: string
  plan_hash: string
  project_id: string
  destination: string
  minimum_stars: 1 | 2 | 3
  created_at: string
  version: number
  copy_count: number
  skip_count: number
  conflict_count: number
  invalid_count: number
  items: unknown[]
}

export interface ExportReceipt {
  operation_id: string
  plan_hash: string
  destination: string
  copied: number
  skipped: number
  conflicts: number
  invalid: number
  verification_passed: boolean
  completed_at: string
  items: unknown[]
}

export interface ProjectSummary {
  id: string
  name: string
  source_name: string
  created_at: string
  total: number
  selected: number
  stars_3?: number
}

export interface EngineSettings {
  grouping_preset: GroupingPreset
  keep_per_group: number
  coverage_enabled: boolean
  coverage_window_minutes: number
  face_identity_threshold: number
  use_gpu: boolean
  recursive: boolean
  jpeg_preview_quality: number
  vlm_enabled: boolean
  vlm_server_url: string
  vlm_executable_path: string
  vlm_model_path: string
  vlm_mmproj_path: string
  vlm_model_id: string
  vlm_quantization: string
  vlm_context_size: number
  vlm_gpu_layers: number
  vlm_max_groups: number
  vlm_max_candidates: number
  vlm_ambiguity_margin: number
  vlm_min_confidence: number
  vlm_timeout_seconds: number
}
