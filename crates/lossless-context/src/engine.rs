//! LCM engine — coordinates ingestion, assembly, compaction, and bootstrap.

use crate::assembler::{AssemblyResult, ContextAssembler};
use crate::compaction::{CompactionEngine, CompactionResult, SummarizeFn};
use crate::config::LcmConfig;
use crate::db::connection::LcmDatabase;
use crate::store::conversation::ConversationStore;
use crate::store::summary::SummaryStore;
use anyhow::Result;
use std::path::Path;
use std::sync::Arc;

/// Central LCM coordinator.
pub struct LcmEngine {
    config: LcmConfig,
    db: Arc<LcmDatabase>,
    conv_store: Arc<ConversationStore>,
    sum_store: Arc<SummaryStore>,
    compaction: CompactionEngine,
    assembler: ContextAssembler,
    /// Stable fallback session ID for CLI sessions where memory_session_id is None.
    fallback_session_id: String,
}

impl LcmEngine {
    /// Create a new LCM engine with a database at the given path.
    pub fn open(config: LcmConfig, db_path: &Path) -> Result<Self> {
        let db = Arc::new(LcmDatabase::open(db_path)?);
        Self::with_db(config, db)
    }

    /// Create a new LCM engine with a pre-opened database (useful for testing).
    pub fn with_db(config: LcmConfig, db: Arc<LcmDatabase>) -> Result<Self> {
        let conv_store = Arc::new(ConversationStore::new(db.clone()));
        let sum_store = Arc::new(SummaryStore::new(db.clone()));
        let compaction =
            CompactionEngine::new(config.clone(), conv_store.clone(), sum_store.clone());
        let assembler = ContextAssembler::new(
            conv_store.clone(),
            sum_store.clone(),
            config.fresh_tail_count,
        );
        let fallback_session_id = uuid::Uuid::new_v4().to_string();

        Ok(Self {
            config,
            db,
            conv_store,
            sum_store,
            compaction,
            assembler,
            fallback_session_id,
        })
    }

    /// Resolve a session_id, falling back to the stable per-instance ID for CLI.
    fn resolve_session_id<'a>(&'a self, session_id: Option<&'a str>) -> &'a str {
        session_id.unwrap_or(&self.fallback_session_id)
    }

    /// Get or create the conversation_id for a session.
    fn conversation_id(&self, session_id: &str) -> Result<i64> {
        self.conv_store.get_or_create_conversation(session_id)
    }

    /// Persist a message to SQLite.
    ///
    /// Called from the agent after every non-system `history.push()`, before
    /// `trim_history()`.
    pub fn ingest(&self, session_id: Option<&str>, role: &str, content: &str) -> Result<i64> {
        let sid = self.resolve_session_id(session_id);
        let conv_id = self.conversation_id(sid)?;
        let msg_id = self.conv_store.insert_message(conv_id, role, content)?;
        self.sum_store
            .append_message_context_item(conv_id, msg_id)?;
        Ok(msg_id)
    }

    /// Assemble the context for an LLM call from SQLite.
    ///
    /// Returns summaries (XML-wrapped) + fresh tail, fitted to `token_budget`.
    pub fn assemble(
        &self,
        session_id: Option<&str>,
        token_budget: usize,
    ) -> Result<AssemblyResult> {
        let sid = self.resolve_session_id(session_id);
        let conv_id = self.conversation_id(sid)?;
        self.assembler.assemble(conv_id, token_budget)
    }

    /// Evaluate and run compaction if needed.
    ///
    /// Called after successful turns.  Skipped if `autocompact_disabled` is set.
    pub async fn compact(
        &self,
        session_id: Option<&str>,
        token_budget: usize,
        summarize: &SummarizeFn,
    ) -> Result<Vec<CompactionResult>> {
        if self.config.autocompact_disabled {
            return Ok(vec![]);
        }

        let sid = self.resolve_session_id(session_id);
        let conv_id = self.conversation_id(sid)?;

        // Check if compaction is needed
        let assembly = self.assembler.assemble(conv_id, usize::MAX)?;
        let threshold = (token_budget as f64 * self.config.context_threshold) as usize;

        if assembly.estimated_tokens <= threshold {
            return Ok(vec![]);
        }

        tracing::info!(
            tokens = assembly.estimated_tokens,
            threshold,
            "Context exceeds threshold, running compaction"
        );

        self.compaction
            .full_sweep(conv_id, token_budget, summarize)
            .await
    }

    /// Bootstrap: reconcile existing history with LCM SQLite state.
    ///
    /// If LCM already has data for this session (from a prior connection),
    /// use it.  Otherwise, import the provided history as initial messages.
    pub fn bootstrap(
        &self,
        session_id: Option<&str>,
        messages: &[(String, String)], // (role, content) pairs
    ) -> Result<()> {
        let sid = self.resolve_session_id(session_id);
        let conv_id = self.conversation_id(sid)?;

        let existing_count = self.conv_store.message_count(conv_id)?;
        if existing_count > 0 {
            tracing::debug!(
                session_id = sid,
                existing_count,
                "LCM already has data for session, skipping import"
            );
            return Ok(());
        }

        if messages.is_empty() {
            return Ok(());
        }

        tracing::info!(
            session_id = sid,
            count = messages.len(),
            "Bootstrapping LCM from session backend history"
        );

        for (role, content) in messages {
            let msg_id = self.conv_store.insert_message(conv_id, role, content)?;
            self.sum_store
                .append_message_context_item(conv_id, msg_id)?;
        }

        Ok(())
    }

    /// Access the underlying database (for tools that need shared access).
    pub fn database(&self) -> &Arc<LcmDatabase> {
        &self.db
    }

    /// Access the conversation store.
    pub fn conversation_store(&self) -> &Arc<ConversationStore> {
        &self.conv_store
    }

    /// Access the summary store.
    pub fn summary_store(&self) -> &Arc<SummaryStore> {
        &self.sum_store
    }

    /// Check if FTS5 is available.
    pub fn fts5_available(&self) -> bool {
        self.db.fts5_available()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_engine() -> LcmEngine {
        let db = Arc::new(LcmDatabase::open_in_memory().unwrap());
        LcmEngine::with_db(LcmConfig::default(), db).unwrap()
    }

    #[test]
    fn ingest_and_assemble_round_trip() {
        let engine = test_engine();

        engine.ingest(Some("sess-1"), "user", "Hello world").unwrap();
        engine
            .ingest(Some("sess-1"), "assistant", "Hi there!")
            .unwrap();
        engine
            .ingest(Some("sess-1"), "user", "How are you?")
            .unwrap();

        let result = engine.assemble(Some("sess-1"), 100_000).unwrap();
        assert_eq!(result.raw_message_count, 3);
        assert_eq!(result.summary_count, 0);
        assert_eq!(result.items.len(), 3);
        assert_eq!(result.items[0].role, "user");
        assert_eq!(result.items[0].content, "Hello world");
        assert!(result.system_prompt_addition.is_none());
    }

    #[test]
    fn bootstrap_imports_history() {
        let engine = test_engine();

        let history = vec![
            ("user".to_string(), "Message 1".to_string()),
            ("assistant".to_string(), "Response 1".to_string()),
            ("user".to_string(), "Message 2".to_string()),
        ];

        engine.bootstrap(Some("sess-1"), &history).unwrap();

        let result = engine.assemble(Some("sess-1"), 100_000).unwrap();
        assert_eq!(result.raw_message_count, 3);
        assert_eq!(result.items[0].content, "Message 1");
    }

    #[test]
    fn bootstrap_skips_if_data_exists() {
        let engine = test_engine();

        engine.ingest(Some("sess-1"), "user", "Already here").unwrap();

        let history = vec![
            ("user".to_string(), "New message".to_string()),
        ];
        engine.bootstrap(Some("sess-1"), &history).unwrap();

        // Should still have just the original message
        let result = engine.assemble(Some("sess-1"), 100_000).unwrap();
        assert_eq!(result.raw_message_count, 1);
        assert_eq!(result.items[0].content, "Already here");
    }

    #[test]
    fn fallback_session_id_stable_across_calls() {
        let engine = test_engine();

        engine.ingest(None, "user", "CLI message 1").unwrap();
        engine.ingest(None, "user", "CLI message 2").unwrap();

        let result = engine.assemble(None, 100_000).unwrap();
        assert_eq!(result.raw_message_count, 2);
    }
}
