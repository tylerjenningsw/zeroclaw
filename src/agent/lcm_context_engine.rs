//! LCM-backed ContextEngine implementation.
//!
//! Persists every message to SQLite via `LcmEngine::ingest()`, assembles
//! context from summaries + fresh tail, and runs compaction after turns.
//! Feature-gated behind `lossless-context`.

use crate::agent::context_engine::{ContextEngine, TurnContext};
use crate::agent::dispatcher::ToolDispatcher;
use crate::providers::{ChatMessage, ConversationMessage};
use anyhow::Result;
use async_trait::async_trait;
use lossless_context::compaction::SummarizeFn;
use lossless_context::engine::LcmEngine;
use std::sync::Arc;

/// ContextEngine implementation backed by LCM (Lossless Context Management).
///
/// SQLite is the durable source of truth.  `self.history` in Agent becomes
/// a RAM cache — `trim_history()` only saves memory, it never loses data.
pub struct LcmContextEngine {
    engine: Arc<LcmEngine>,
    /// LLM callback for summarization during compaction.
    summarize_fn: Option<SummarizeFn>,
}

impl LcmContextEngine {
    pub fn new(engine: Arc<LcmEngine>) -> Self {
        Self {
            engine,
            summarize_fn: None,
        }
    }

    /// Set the summarization callback (needed for compaction).
    /// Called after provider is available.
    pub fn set_summarize_fn(&mut self, f: SummarizeFn) {
        self.summarize_fn = Some(f);
    }

    /// Access the underlying LcmEngine (for tool registration).
    pub fn engine(&self) -> &Arc<LcmEngine> {
        &self.engine
    }
}

#[async_trait]
impl ContextEngine for LcmContextEngine {
    async fn ingest(
        &self,
        session_id: Option<&str>,
        message: &ConversationMessage,
    ) -> Result<()> {
        let (role, content) = extract_role_content(message);
        let _ = self.engine.ingest(session_id, &role, &content)?;
        Ok(())
    }

    async fn assemble(
        &self,
        session_id: Option<&str>,
        token_budget: usize,
        tool_dispatcher: &dyn ToolDispatcher,
        history: &[ConversationMessage],
    ) -> Result<Vec<ChatMessage>> {
        // Check if LCM has data for this session
        let sid = session_id.unwrap_or("default");
        let conv_id = self.engine.conversation_store().get_or_create_conversation(sid)?;
        let msg_count = self.engine.conversation_store().message_count(conv_id)?;

        if msg_count == 0 {
            // No LCM data yet — fall back to dispatcher (current behavior)
            return Ok(tool_dispatcher.to_provider_messages(history));
        }

        // Assemble from SQLite
        let assembly = self.engine.assemble(session_id, token_budget)?;

        if assembly.items.is_empty() {
            // Shouldn't happen if msg_count > 0, but be safe
            return Ok(tool_dispatcher.to_provider_messages(history));
        }

        // Build ChatMessage list:
        // 1. Keep system message from history (if present)
        // 2. Add assembled items (summaries as user messages, raw messages)
        // 3. The fresh tail from assembly already contains recent messages
        let mut messages: Vec<ChatMessage> = Vec::new();

        // System message from current history (not persisted in LCM)
        if let Some(ConversationMessage::Chat(chat)) = history.first() {
            if chat.role == "system" {
                let mut system_content = chat.content.clone();
                // Append LCM system prompt addition if present
                if let Some(ref addition) = assembly.system_prompt_addition {
                    system_content.push_str("\n\n");
                    system_content.push_str(addition);
                }
                messages.push(ChatMessage::system(system_content));
            }
        }

        // Convert assembled items to ChatMessages
        // Use tool_dispatcher for fresh tail messages to preserve XML/native semantics
        let native_format = tool_dispatcher.should_send_tool_specs();

        for item in &assembly.items {
            if item.is_summary {
                // Summaries are always user-role messages
                messages.push(ChatMessage::user(item.content.clone()));
            } else {
                // Raw messages — use the role from the item
                match item.role.as_str() {
                    "system" => {} // already handled above
                    "user" => messages.push(ChatMessage::user(item.content.clone())),
                    "assistant" => messages.push(ChatMessage::assistant(item.content.clone())),
                    "tool" => {
                        if native_format {
                            messages.push(ChatMessage::tool(item.content.clone()));
                        } else {
                            // XML dispatcher formats tool results as user messages
                            messages.push(ChatMessage::user(format!(
                                "[Tool results]\n{}",
                                item.content
                            )));
                        }
                    }
                    _ => messages.push(ChatMessage::user(item.content.clone())),
                }
            }
        }

        Ok(messages)
    }

    async fn after_turn(
        &self,
        session_id: Option<&str>,
        ctx: &TurnContext<'_>,
    ) -> Result<()> {
        if let Some(ref summarize) = self.summarize_fn {
            let results = self
                .engine
                .compact(session_id, ctx.max_context_tokens, summarize)
                .await?;
            if !results.is_empty() {
                let total_saved: usize = results
                    .iter()
                    .map(|r| r.tokens_before.saturating_sub(r.tokens_after))
                    .sum();
                tracing::info!(
                    passes = results.len(),
                    tokens_saved = total_saved,
                    "LCM compaction complete"
                );
            }
        }
        Ok(())
    }

    async fn bootstrap(
        &self,
        session_id: Option<&str>,
        existing: &[ConversationMessage],
    ) -> Result<()> {
        // Convert ConversationMessage to (role, content) pairs
        let pairs: Vec<(String, String)> = existing
            .iter()
            .filter_map(|msg| {
                let (role, content) = extract_role_content(msg);
                if role == "system" {
                    None // don't persist system prompts
                } else {
                    Some((role, content))
                }
            })
            .collect();

        self.engine.bootstrap(session_id, &pairs)
    }
}

/// Extract role and content from a ConversationMessage.
fn extract_role_content(message: &ConversationMessage) -> (String, String) {
    match message {
        ConversationMessage::Chat(chat) => (chat.role.clone(), chat.content.clone()),
        ConversationMessage::AssistantToolCalls {
            text, tool_calls, ..
        } => {
            // Store as assistant message with tool call info
            let tc_summary: Vec<String> = tool_calls
                .iter()
                .map(|tc| format!("[tool_call: {} ({})]", tc.name, tc.id))
                .collect();
            let content = format!(
                "{}\n{}",
                text.as_deref().unwrap_or(""),
                tc_summary.join("\n")
            );
            ("assistant".to_string(), content)
        }
        ConversationMessage::ToolResults(results) => {
            let content: String = results
                .iter()
                .map(|r| format!("[tool_result: {}] {}", r.tool_call_id, r.content))
                .collect::<Vec<_>>()
                .join("\n");
            ("tool".to_string(), content)
        }
    }
}
