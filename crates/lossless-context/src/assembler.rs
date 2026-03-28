//! Context assembler — builds the provider message list from SQLite.
//!
//! Combines summaries (XML-wrapped) + fresh tail messages, fitted to a token budget.

use crate::config::estimate_tokens;
use crate::prompts;
use crate::store::conversation::ConversationStore;
use crate::store::summary::SummaryStore;
use anyhow::Result;
use std::sync::Arc;

/// A resolved context item ready for assembly.
#[derive(Debug, Clone)]
pub struct ResolvedItem {
    pub role: String,
    pub content: String,
    pub tokens: usize,
    pub is_summary: bool,
    pub summary_id: Option<String>,
}

/// Result of context assembly.
pub struct AssemblyResult {
    /// Resolved items in order (summaries + fresh tail).
    pub items: Vec<ResolvedItem>,
    /// Total estimated tokens of assembled context.
    pub estimated_tokens: usize,
    /// System prompt addition for LCM tool guidance (if summaries present).
    pub system_prompt_addition: Option<String>,
    pub raw_message_count: usize,
    pub summary_count: usize,
}

pub struct ContextAssembler {
    conv_store: Arc<ConversationStore>,
    sum_store: Arc<SummaryStore>,
    fresh_tail_count: usize,
}

impl ContextAssembler {
    pub fn new(
        conv_store: Arc<ConversationStore>,
        sum_store: Arc<SummaryStore>,
        fresh_tail_count: usize,
    ) -> Self {
        Self {
            conv_store,
            sum_store,
            fresh_tail_count,
        }
    }

    /// Assemble the context for an LLM call.
    ///
    /// Returns resolved items (summaries as XML-wrapped user messages + raw messages)
    /// fitted to `token_budget`.  The fresh tail is always included.
    pub fn assemble(
        &self,
        conversation_id: i64,
        token_budget: usize,
    ) -> Result<AssemblyResult> {
        let items = self.sum_store.get_context_items(conversation_id)?;
        if items.is_empty() {
            return Ok(AssemblyResult {
                items: vec![],
                estimated_tokens: 0,
                system_prompt_addition: None,
                raw_message_count: 0,
                summary_count: 0,
            });
        }

        let messages = self.conv_store.get_messages(conversation_id)?;
        let summaries = self.sum_store.get_summaries(conversation_id)?;

        // Resolve all items
        let mut resolved: Vec<ResolvedItem> = Vec::with_capacity(items.len());
        let mut summary_count = 0;
        let mut raw_message_count = 0;

        for item in &items {
            match item.item_type.as_str() {
                "message" => {
                    if let Some(msg_id) = item.message_id {
                        if let Some(msg) = messages.iter().find(|m| m.message_id == msg_id) {
                            resolved.push(ResolvedItem {
                                role: msg.role.clone(),
                                content: msg.content.clone(),
                                tokens: msg.token_count as usize,
                                is_summary: false,
                                summary_id: None,
                            });
                            raw_message_count += 1;
                        }
                    }
                }
                "summary" => {
                    if let Some(ref sid) = item.summary_id {
                        if let Some(s) = summaries.iter().find(|s| &s.summary_id == sid) {
                            let xml = format_summary_xml(s);
                            let tokens = estimate_tokens(&xml);
                            resolved.push(ResolvedItem {
                                role: "user".to_string(),
                                content: xml,
                                tokens,
                                is_summary: true,
                                summary_id: Some(s.summary_id.clone()),
                            });
                            summary_count += 1;
                        }
                    }
                }
                _ => {}
            }
        }

        // Split into fresh tail and evictable
        let tail_start = resolved.len().saturating_sub(self.fresh_tail_count);
        let fresh_tail = &resolved[tail_start..];
        let evictable = &resolved[..tail_start];

        // Fresh tail always included
        let tail_tokens: usize = fresh_tail.iter().map(|r| r.tokens).sum();

        // Fill remaining budget from evictable (newest first)
        let remaining_budget = token_budget.saturating_sub(tail_tokens);
        let mut evict_tokens: usize = 0;
        let mut evict_cutoff = evictable.len(); // include everything by default

        for (i, item) in evictable.iter().enumerate().rev() {
            if evict_tokens + item.tokens > remaining_budget {
                evict_cutoff = i + 1;
                break;
            }
            evict_tokens += item.tokens;
            if i == 0 {
                evict_cutoff = 0;
            }
        }

        let mut result: Vec<ResolvedItem> = Vec::new();
        result.extend_from_slice(&evictable[evict_cutoff..]);
        result.extend_from_slice(fresh_tail);

        let estimated_tokens = result.iter().map(|r| r.tokens).sum();

        // Build system prompt addition if summaries present
        let result_summary_count = result.iter().filter(|r| r.is_summary).count();
        let system_prompt_addition = if result_summary_count > 0 {
            let max_depth = summaries.iter().map(|s| s.depth).max().unwrap_or(0);
            let condensed_count = summaries
                .iter()
                .filter(|s| s.kind == "condensed")
                .count();
            Some(prompts::build_system_prompt_addition(
                max_depth,
                condensed_count,
            ))
        } else {
            None
        };

        Ok(AssemblyResult {
            items: result,
            estimated_tokens,
            system_prompt_addition,
            raw_message_count,
            summary_count,
        })
    }
}

/// Format a summary as XML for injection into the context.
fn format_summary_xml(
    s: &crate::store::summary::SummaryRow,
) -> String {
    let earliest = s.earliest_at.as_deref().unwrap_or("");
    let latest = s.latest_at.as_deref().unwrap_or("");

    format!(
        "<summary id=\"{}\" kind=\"{}\" depth=\"{}\" descendant_count=\"{}\" earliest_at=\"{}\" latest_at=\"{}\">\n\
         <content>\n{}\n</content>\n\
         </summary>",
        s.summary_id, s.kind, s.depth, s.descendant_count, earliest, latest, s.content,
    )
}
