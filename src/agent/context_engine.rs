//! Context engine trait for pluggable message persistence and context assembly.
//!
//! The default implementation preserves existing behavior exactly: `assemble()`
//! delegates to `ToolDispatcher::to_provider_messages()`, and all other methods
//! are no-ops.  The `LcmContextEngine` (behind the `lossless-context` feature)
//! persists every message to SQLite and assembles context from summaries + fresh
//! tail, enabling indefinite conversation length.

use crate::agent::dispatcher::ToolDispatcher;
use crate::providers::{ChatMessage, ConversationMessage};
use anyhow::Result;
use async_trait::async_trait;

/// Metadata passed to [`ContextEngine::after_turn`] for compaction decisions.
pub struct TurnContext<'a> {
    pub model: &'a str,
    pub max_context_tokens: usize,
}

/// Pluggable context engine that intercepts the agent message lifecycle.
///
/// # Contract
///
/// - `ingest` is called after every non-system `self.history.push()`, **before**
///   `trim_history()`.  This is the persistence point — once a message is
///   ingested, it survives in-memory trimming.
///
/// - `assemble` replaces `tool_dispatcher.to_provider_messages()`.  It must
///   return a `Vec<ChatMessage>` ready for the LLM provider, preserving
///   dispatcher semantics (XML vs native tool format) via the
///   `should_send_tool_specs()` signal.
///
/// - `after_turn` is called on successful turn completion (cache-hit or normal
///   response).  It is NOT called on error/bail exits.
///
/// - `bootstrap` is called once on session start, after `seed_history()`.
///   It runs unconditionally — even when the session backend has no messages —
///   because the engine may have its own persisted data for the session.
#[async_trait]
pub trait ContextEngine: Send + Sync {
    /// Persist a message to durable storage.
    async fn ingest(
        &self,
        _session_id: Option<&str>,
        _message: &ConversationMessage,
    ) -> Result<()> {
        Ok(())
    }

    /// Build the provider message list for the next LLM call.
    async fn assemble(
        &self,
        session_id: Option<&str>,
        token_budget: usize,
        tool_dispatcher: &dyn ToolDispatcher,
        history: &[ConversationMessage],
    ) -> Result<Vec<ChatMessage>>;

    /// Post-turn hook for compaction evaluation.
    async fn after_turn(
        &self,
        _session_id: Option<&str>,
        _ctx: &TurnContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    /// Reconcile persisted state with the in-memory history on session start.
    async fn bootstrap(
        &self,
        _session_id: Option<&str>,
        _existing: &[ConversationMessage],
    ) -> Result<()> {
        Ok(())
    }
}

/// Default implementation that preserves current ZeroClaw behavior exactly.
///
/// `assemble` delegates to `tool_dispatcher.to_provider_messages(history)`.
/// All other methods are no-ops.
pub struct DefaultContextEngine;

#[async_trait]
impl ContextEngine for DefaultContextEngine {
    async fn assemble(
        &self,
        _session_id: Option<&str>,
        _token_budget: usize,
        tool_dispatcher: &dyn ToolDispatcher,
        history: &[ConversationMessage],
    ) -> Result<Vec<ChatMessage>> {
        Ok(tool_dispatcher.to_provider_messages(history))
    }
}
