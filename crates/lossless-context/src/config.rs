use serde::{Deserialize, Serialize};

/// Lossless Context Management configuration.
///
/// Defaults match the upstream TypeScript `openclaw.plugin.json` schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LcmConfig {
    pub enabled: bool,
    /// Fraction of context window that triggers compaction (0.0–1.0).
    pub context_threshold: f64,
    /// Number of recent raw messages protected from compaction.
    pub fresh_tail_count: usize,
    /// Minimum raw messages needed to form a leaf summary.
    pub leaf_min_fanout: usize,
    /// Minimum same-depth summaries needed for condensation.
    pub condensed_min_fanout: usize,
    /// Hard minimum for condensation (used in budget sweep).
    pub condensed_min_fanout_hard: usize,
    /// Max depth for incremental compaction (0 = leaf only, -1 = unlimited).
    pub incremental_max_depth: i32,
    /// Max source tokens per leaf compaction chunk.
    pub leaf_chunk_tokens: usize,
    /// Target token count for leaf summaries.
    pub leaf_target_tokens: usize,
    /// Target token count for condensed summaries.
    pub condensed_target_tokens: usize,
    /// Max tokens per expand_query answer.
    pub max_expand_tokens: usize,
    /// File size threshold for externalization (tokens).
    pub large_file_token_threshold: usize,
    /// Override model for summarization (cheaper model).
    pub compaction_model: Option<String>,
    /// Override model for expand_query tool.
    pub expansion_model: Option<String>,
    /// Database file path override.  Default: `{workspace}/lcm.db`.
    pub database_path: Option<String>,
    /// Disable automatic compaction after each turn.
    pub autocompact_disabled: bool,
}

impl Default for LcmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            context_threshold: 0.75,
            fresh_tail_count: 32,
            leaf_min_fanout: 8,
            condensed_min_fanout: 4,
            condensed_min_fanout_hard: 2,
            incremental_max_depth: 0,
            leaf_chunk_tokens: 20_000,
            leaf_target_tokens: 1_200,
            condensed_target_tokens: 2_000,
            max_expand_tokens: 4_000,
            large_file_token_threshold: 25_000,
            compaction_model: None,
            expansion_model: None,
            database_path: None,
            autocompact_disabled: false,
        }
    }
}

/// Rough token estimate: ~4 characters per token.
pub fn estimate_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}
