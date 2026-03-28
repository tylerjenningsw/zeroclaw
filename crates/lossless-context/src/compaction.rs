//! Compaction engine — leaf pass, condensation pass, and budget sweep.
//!
//! Ported from upstream lossless-claw `compaction.ts`.

use crate::config::{estimate_tokens, LcmConfig};
use crate::prompts;
use crate::store::conversation::ConversationStore;
use crate::store::summary::SummaryStore;
use anyhow::Result;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Callback type for LLM summarization calls.
///
/// Takes (system_prompt, user_prompt, temperature, max_tokens) → Result<String>.
pub type SummarizeFn = Arc<
    dyn Fn(String, String, f64, usize) -> Pin<Box<dyn Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
>;

/// Result of a compaction pass.
#[derive(Debug)]
pub struct CompactionResult {
    pub summary_id: Option<String>,
    pub tokens_before: usize,
    pub tokens_after: usize,
    pub level: &'static str, // "normal", "aggressive", "fallback", "skipped"
}

pub struct CompactionEngine {
    config: LcmConfig,
    conv_store: Arc<ConversationStore>,
    sum_store: Arc<SummaryStore>,
}

impl CompactionEngine {
    pub fn new(
        config: LcmConfig,
        conv_store: Arc<ConversationStore>,
        sum_store: Arc<SummaryStore>,
    ) -> Self {
        Self {
            config,
            conv_store,
            sum_store,
        }
    }

    /// Run a single leaf pass: summarize the oldest raw messages outside the fresh tail.
    pub async fn leaf_pass(
        &self,
        conversation_id: i64,
        summarize: &SummarizeFn,
        previous_context: Option<&str>,
    ) -> Result<CompactionResult> {
        let items = self.sum_store.get_context_items(conversation_id)?;
        if items.is_empty() {
            return Ok(CompactionResult {
                summary_id: None,
                tokens_before: 0,
                tokens_after: 0,
                level: "skipped",
            });
        }

        // Find oldest contiguous raw-message chunk outside fresh tail
        let tail_start = items.len().saturating_sub(self.config.fresh_tail_count);
        let mut chunk_start: Option<usize> = None;
        let mut chunk_end: usize = 0;
        let mut chunk_tokens: usize = 0;
        let mut chunk_msg_ids: Vec<i64> = Vec::new();

        for (i, item) in items.iter().enumerate() {
            if i >= tail_start {
                break;
            }
            if item.item_type != "message" {
                if !chunk_msg_ids.is_empty() {
                    break; // stop at first non-message
                }
                continue;
            }
            if chunk_start.is_none() {
                chunk_start = Some(i);
            }
            chunk_end = i;
            if let Some(msg_id) = item.message_id {
                chunk_msg_ids.push(msg_id);
                // Estimate tokens from the message
                let messages = self.conv_store.get_messages(conversation_id)?;
                if let Some(msg) = messages.iter().find(|m| m.message_id == msg_id) {
                    chunk_tokens += msg.token_count as usize;
                }
            }
            if chunk_tokens >= self.config.leaf_chunk_tokens {
                break;
            }
        }

        if chunk_msg_ids.len() < self.config.leaf_min_fanout {
            return Ok(CompactionResult {
                summary_id: None,
                tokens_before: chunk_tokens,
                tokens_after: chunk_tokens,
                level: "skipped",
            });
        }

        // Build source text from messages
        let messages = self.conv_store.get_messages(conversation_id)?;
        let source_parts: Vec<String> = chunk_msg_ids
            .iter()
            .filter_map(|id| {
                messages.iter().find(|m| m.message_id == *id).map(|m| {
                    format!("[{}] {}: {}", m.created_at, m.role, m.content)
                })
            })
            .collect();
        let source_text = source_parts.join("\n\n");
        let input_tokens = estimate_tokens(&source_text);

        // Three-level escalation
        let (content, level) = self
            .summarize_with_escalation(
                &source_text,
                input_tokens,
                0,
                false,
                previous_context,
                summarize,
            )
            .await?;

        // Get time range
        let earliest = messages
            .iter()
            .find(|m| chunk_msg_ids.contains(&m.message_id))
            .map(|m| m.created_at.as_str());
        let latest = messages
            .iter()
            .rev()
            .find(|m| chunk_msg_ids.contains(&m.message_id))
            .map(|m| m.created_at.as_str());

        let model = self
            .config
            .compaction_model
            .as_deref()
            .unwrap_or("unknown");

        let summary_id = self.sum_store.insert_leaf_summary(
            conversation_id,
            &content,
            model,
            &chunk_msg_ids,
            earliest,
            latest,
        )?;

        // Replace context range
        let start_ord = items[chunk_start.unwrap()].ordinal;
        let end_ord = items[chunk_end].ordinal;
        self.sum_store.replace_context_range_with_summary(
            conversation_id,
            start_ord,
            end_ord,
            &summary_id,
        )?;

        let tokens_after = estimate_tokens(&content);

        Ok(CompactionResult {
            summary_id: Some(summary_id),
            tokens_before: input_tokens,
            tokens_after,
            level,
        })
    }

    /// Run a condensation pass at a given depth.
    pub async fn condensation_pass(
        &self,
        conversation_id: i64,
        target_depth: i64,
        summarize: &SummarizeFn,
        previous_context: Option<&str>,
    ) -> Result<CompactionResult> {
        let items = self.sum_store.get_context_items(conversation_id)?;
        let fanout = if target_depth == 0 {
            self.config.leaf_min_fanout
        } else {
            self.config.condensed_min_fanout
        };

        // Find contiguous summaries at target_depth
        let mut chunk_start: Option<usize> = None;
        let mut chunk_end: usize = 0;
        let mut chunk_summary_ids: Vec<String> = Vec::new();
        let mut chunk_tokens: usize = 0;

        let summaries = self.sum_store.get_summaries(conversation_id)?;

        for (i, item) in items.iter().enumerate() {
            if item.item_type != "summary" {
                if !chunk_summary_ids.is_empty() {
                    break;
                }
                continue;
            }
            if let Some(ref sid) = item.summary_id {
                if let Some(s) = summaries.iter().find(|s| &s.summary_id == sid) {
                    if s.depth != target_depth {
                        if !chunk_summary_ids.is_empty() {
                            break;
                        }
                        continue;
                    }
                    if chunk_start.is_none() {
                        chunk_start = Some(i);
                    }
                    chunk_end = i;
                    chunk_summary_ids.push(sid.clone());
                    chunk_tokens += s.token_count as usize;
                    if chunk_tokens >= self.config.leaf_chunk_tokens {
                        break;
                    }
                }
            }
        }

        if chunk_summary_ids.len() < fanout {
            return Ok(CompactionResult {
                summary_id: None,
                tokens_before: chunk_tokens,
                tokens_after: chunk_tokens,
                level: "skipped",
            });
        }

        // Build source text from summaries
        let source_parts: Vec<String> = chunk_summary_ids
            .iter()
            .filter_map(|sid| {
                summaries.iter().find(|s| &s.summary_id == sid).map(|s| {
                    let time_range = match (&s.earliest_at, &s.latest_at) {
                        (Some(e), Some(l)) => format!("[{e} - {l}]\n"),
                        _ => String::new(),
                    };
                    format!("{time_range}{}", s.content)
                })
            })
            .collect();
        let source_text = source_parts.join("\n\n");
        let input_tokens = estimate_tokens(&source_text);

        let new_depth = target_depth + 1;
        let (content, level) = self
            .summarize_with_escalation(
                &source_text,
                input_tokens,
                new_depth,
                true,
                previous_context,
                summarize,
            )
            .await?;

        let earliest = summaries
            .iter()
            .find(|s| chunk_summary_ids.contains(&s.summary_id))
            .and_then(|s| s.earliest_at.as_deref());
        let latest = summaries
            .iter()
            .rev()
            .find(|s| chunk_summary_ids.contains(&s.summary_id))
            .and_then(|s| s.latest_at.as_deref());

        let model = self
            .config
            .compaction_model
            .as_deref()
            .unwrap_or("unknown");

        let summary_id = self.sum_store.insert_condensed_summary(
            conversation_id,
            new_depth,
            &content,
            model,
            &chunk_summary_ids,
            earliest,
            latest,
        )?;

        // Replace context range
        let start_ord = items[chunk_start.unwrap()].ordinal;
        let end_ord = items[chunk_end].ordinal;
        self.sum_store.replace_context_range_with_summary(
            conversation_id,
            start_ord,
            end_ord,
            &summary_id,
        )?;

        let tokens_after = estimate_tokens(&content);

        Ok(CompactionResult {
            summary_id: Some(summary_id),
            tokens_before: input_tokens,
            tokens_after,
            level,
        })
    }

    /// Full sweep: repeated leaf passes then condensation passes.
    pub async fn full_sweep(
        &self,
        conversation_id: i64,
        token_budget: usize,
        summarize: &SummarizeFn,
    ) -> Result<Vec<CompactionResult>> {
        let threshold = (token_budget as f64 * self.config.context_threshold) as usize;
        let mut results = Vec::new();

        // Phase 1: Leaf passes
        let mut prev_content: Option<String> = None;
        loop {
            let result = self
                .leaf_pass(
                    conversation_id,
                    summarize,
                    prev_content.as_deref(),
                )
                .await?;

            if result.summary_id.is_none() {
                break;
            }

            if result.tokens_after >= result.tokens_before {
                results.push(result);
                break;
            }

            // Track previous summary content for continuity
            if let Some(ref sid) = result.summary_id {
                if let Some(s) = self.sum_store.get_summary(sid)? {
                    prev_content = Some(s.content);
                }
            }

            results.push(result);

            // Check if we're under threshold
            let total = self.estimate_context_tokens(conversation_id)?;
            if total <= threshold {
                break;
            }
        }

        // Phase 2: Condensation passes (bottom-up by depth)
        let mut prev_tokens = self.estimate_context_tokens(conversation_id)?;
        if prev_tokens > threshold {
            for depth in 0..10 {
                let result = self
                    .condensation_pass(conversation_id, depth, summarize, None)
                    .await?;

                if result.summary_id.is_none() {
                    continue; // try next depth
                }

                results.push(result);

                let current = self.estimate_context_tokens(conversation_id)?;
                if current <= threshold || current >= prev_tokens {
                    break;
                }
                prev_tokens = current;
            }
        }

        Ok(results)
    }

    /// Estimate total context tokens for a conversation.
    fn estimate_context_tokens(&self, conversation_id: i64) -> Result<usize> {
        let items = self.sum_store.get_context_items(conversation_id)?;
        let messages = self.conv_store.get_messages(conversation_id)?;
        let summaries = self.sum_store.get_summaries(conversation_id)?;

        let mut total: usize = 0;
        for item in &items {
            match item.item_type.as_str() {
                "message" => {
                    if let Some(msg_id) = item.message_id {
                        if let Some(m) = messages.iter().find(|m| m.message_id == msg_id) {
                            total += m.token_count as usize;
                        }
                    }
                }
                "summary" => {
                    if let Some(ref sid) = item.summary_id {
                        if let Some(s) = summaries.iter().find(|s| &s.summary_id == sid) {
                            total += s.token_count as usize;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(total)
    }

    /// Three-level escalation: normal → aggressive → deterministic fallback.
    async fn summarize_with_escalation(
        &self,
        source_text: &str,
        input_tokens: usize,
        depth: i64,
        is_condensed: bool,
        previous_context: Option<&str>,
        summarize: &SummarizeFn,
    ) -> Result<(String, &'static str)> {
        let target = prompts::resolve_target_tokens(
            input_tokens,
            false,
            is_condensed,
            self.config.condensed_target_tokens,
        );

        // Level 1: Normal
        let prompt = prompts::build_prompt_for_depth(
            depth,
            source_text,
            target,
            false,
            previous_context,
        );
        match summarize(
            prompts::SUMMARIZER_SYSTEM_PROMPT.to_string(),
            prompt,
            0.2,
            target * 2,
        )
        .await
        {
            Ok(result) if !result.trim().is_empty() => {
                let result_tokens = estimate_tokens(&result);
                if result_tokens < input_tokens {
                    return Ok((result, "normal"));
                }
                // Normal didn't compress enough, try aggressive
            }
            Ok(_) => {} // empty result, try aggressive
            Err(e) => {
                tracing::warn!(error = %e, "Normal summarization failed, trying aggressive");
            }
        }

        // Level 2: Aggressive
        let aggressive_target = prompts::resolve_target_tokens(
            input_tokens,
            true,
            is_condensed,
            self.config.condensed_target_tokens,
        );
        let prompt = prompts::build_prompt_for_depth(
            depth,
            source_text,
            aggressive_target,
            true,
            previous_context,
        );
        match summarize(
            prompts::SUMMARIZER_SYSTEM_PROMPT.to_string(),
            prompt,
            0.1,
            aggressive_target * 2,
        )
        .await
        {
            Ok(result) if !result.trim().is_empty() => {
                let result_tokens = estimate_tokens(&result);
                if result_tokens < input_tokens {
                    return Ok((result, "aggressive"));
                }
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, "Aggressive summarization failed, using fallback");
            }
        }

        // Level 3: Deterministic fallback
        let fallback = prompts::deterministic_fallback(source_text, aggressive_target);
        Ok((fallback, "fallback"))
    }
}
