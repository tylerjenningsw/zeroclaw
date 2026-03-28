use crate::config::estimate_tokens;
use crate::db::connection::LcmDatabase;
use anyhow::Result;
use std::sync::Arc;

/// A persisted message row.
#[derive(Debug, Clone)]
pub struct MessageRow {
    pub message_id: i64,
    pub conversation_id: i64,
    pub seq: i64,
    pub role: String,
    pub content: String,
    pub token_count: i64,
    pub created_at: String,
}

/// A persisted message part row.
#[derive(Debug, Clone)]
pub struct MessagePartRow {
    pub part_id: String,
    pub message_id: i64,
    pub session_id: String,
    pub part_type: String,
    pub ordinal: i64,
    pub text_content: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
    pub tool_input: Option<String>,
    pub tool_output: Option<String>,
}

/// CRUD operations for conversations and messages.
pub struct ConversationStore {
    db: Arc<LcmDatabase>,
}

impl ConversationStore {
    pub fn new(db: Arc<LcmDatabase>) -> Self {
        Self { db }
    }

    /// Get or create a conversation for the given session_id.
    /// Returns the conversation_id.
    pub fn get_or_create_conversation(&self, session_id: &str) -> Result<i64> {
        self.db.with_conn(|conn| {
            // Try to find existing conversation for this session
            let existing: Option<i64> = conn
                .query_row(
                    "SELECT conversation_id FROM conversations WHERE session_id = ?1 ORDER BY created_at DESC LIMIT 1",
                    [session_id],
                    |row| row.get(0),
                )
                .ok();

            if let Some(id) = existing {
                return Ok(id);
            }

            // Create new conversation
            conn.execute(
                "INSERT INTO conversations (session_id) VALUES (?1)",
                [session_id],
            )?;
            Ok(conn.last_insert_rowid())
        })
    }

    /// Insert a message and return its message_id.
    ///
    /// Automatically assigns the next sequence number for the conversation.
    pub fn insert_message(
        &self,
        conversation_id: i64,
        role: &str,
        content: &str,
    ) -> Result<i64> {
        self.db.with_conn(|conn| {
            let next_seq: i64 = conn
                .query_row(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation_id = ?1",
                    [conversation_id],
                    |row| row.get(0),
                )
                .unwrap_or(1);

            let token_count = estimate_tokens(content) as i64;

            conn.execute(
                "INSERT INTO messages (conversation_id, seq, role, content, token_count) VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![conversation_id, next_seq, role, content, token_count],
            )?;
            Ok(conn.last_insert_rowid())
        })
    }

    /// Insert a message part linked to a message.
    pub fn insert_message_part(
        &self,
        message_id: i64,
        session_id: &str,
        part_type: &str,
        ordinal: i64,
        text_content: Option<&str>,
        tool_call_id: Option<&str>,
        tool_name: Option<&str>,
        tool_input: Option<&str>,
        tool_output: Option<&str>,
    ) -> Result<String> {
        let part_id = format!("part_{}", uuid::Uuid::new_v4().simple());

        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO message_parts (part_id, message_id, session_id, part_type, ordinal, text_content, tool_call_id, tool_name, tool_input, tool_output)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                rusqlite::params![
                    part_id,
                    message_id,
                    session_id,
                    part_type,
                    ordinal,
                    text_content,
                    tool_call_id,
                    tool_name,
                    tool_input,
                    tool_output,
                ],
            )?;
            Ok(part_id)
        })
    }

    /// Get all messages for a conversation, ordered by seq.
    pub fn get_messages(&self, conversation_id: i64) -> Result<Vec<MessageRow>> {
        self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT message_id, conversation_id, seq, role, content, token_count, created_at
                 FROM messages WHERE conversation_id = ?1 ORDER BY seq ASC",
            )?;
            let rows = stmt
                .query_map([conversation_id], |row| {
                    Ok(MessageRow {
                        message_id: row.get(0)?,
                        conversation_id: row.get(1)?,
                        seq: row.get(2)?,
                        role: row.get(3)?,
                        content: row.get(4)?,
                        token_count: row.get(5)?,
                        created_at: row.get(6)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows)
        })
    }

    /// Get message count for a conversation.
    pub fn message_count(&self, conversation_id: i64) -> Result<i64> {
        self.db.with_conn(|conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?1",
                [conversation_id],
                |row| row.get(0),
            )?;
            Ok(count)
        })
    }

    /// Get total token count for all messages in a conversation.
    pub fn total_tokens(&self, conversation_id: i64) -> Result<i64> {
        self.db.with_conn(|conn| {
            let total: i64 = conn.query_row(
                "SELECT COALESCE(SUM(token_count), 0) FROM messages WHERE conversation_id = ?1",
                [conversation_id],
                |row| row.get(0),
            )?;
            Ok(total)
        })
    }

    /// Get the parts for a message, ordered by ordinal.
    pub fn get_message_parts(&self, message_id: i64) -> Result<Vec<MessagePartRow>> {
        self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT part_id, message_id, session_id, part_type, ordinal, text_content, tool_call_id, tool_name, tool_input, tool_output
                 FROM message_parts WHERE message_id = ?1 ORDER BY ordinal ASC",
            )?;
            let rows = stmt
                .query_map([message_id], |row| {
                    Ok(MessagePartRow {
                        part_id: row.get(0)?,
                        message_id: row.get(1)?,
                        session_id: row.get(2)?,
                        part_type: row.get(3)?,
                        ordinal: row.get(4)?,
                        text_content: row.get(5)?,
                        tool_call_id: row.get(6)?,
                        tool_name: row.get(7)?,
                        tool_input: row.get(8)?,
                        tool_output: row.get(9)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows)
        })
    }

    /// Search messages by content using FTS5 or LIKE fallback.
    pub fn search_messages(
        &self,
        conversation_id: Option<i64>,
        query: &str,
        use_fts: bool,
        limit: usize,
    ) -> Result<Vec<MessageRow>> {
        self.db.with_conn(|conn| {
            let map_row = |row: &rusqlite::Row<'_>| -> rusqlite::Result<MessageRow> {
                Ok(MessageRow {
                    message_id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    seq: row.get(2)?,
                    role: row.get(3)?,
                    content: row.get(4)?,
                    token_count: row.get(5)?,
                    created_at: row.get(6)?,
                })
            };

            if use_fts {
                let sql = if let Some(conv_id) = conversation_id {
                    format!(
                        "SELECT m.message_id, m.conversation_id, m.seq, m.role, m.content, m.token_count, m.created_at
                         FROM messages_fts f
                         JOIN messages m ON m.message_id = f.rowid
                         WHERE f.content MATCH ?1 AND m.conversation_id = {conv_id}
                         ORDER BY rank LIMIT ?2"
                    )
                } else {
                    "SELECT m.message_id, m.conversation_id, m.seq, m.role, m.content, m.token_count, m.created_at
                     FROM messages_fts f
                     JOIN messages m ON m.message_id = f.rowid
                     WHERE f.content MATCH ?1
                     ORDER BY rank LIMIT ?2"
                        .to_string()
                };
                let mut stmt = conn.prepare(&sql)?;
                let rows = stmt.query_map(rusqlite::params![query, limit as i64], map_row)?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(rows)
            } else {
                let like_pattern = format!("%{query}%");
                let sql = if let Some(conv_id) = conversation_id {
                    format!(
                        "SELECT message_id, conversation_id, seq, role, content, token_count, created_at
                         FROM messages
                         WHERE content LIKE ?1 AND conversation_id = {conv_id}
                         ORDER BY seq DESC LIMIT ?2"
                    )
                } else {
                    "SELECT message_id, conversation_id, seq, role, content, token_count, created_at
                     FROM messages
                     WHERE content LIKE ?1
                     ORDER BY seq DESC LIMIT ?2"
                        .to_string()
                };
                let mut stmt = conn.prepare(&sql)?;
                let rows = stmt.query_map(rusqlite::params![like_pattern, limit as i64], map_row)?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(rows)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Arc<LcmDatabase> {
        Arc::new(LcmDatabase::open_in_memory().unwrap())
    }

    #[test]
    fn insert_and_retrieve_messages() {
        let db = test_db();
        let store = ConversationStore::new(db);

        let conv_id = store.get_or_create_conversation("test-session").unwrap();
        assert!(conv_id > 0);

        let msg1 = store.insert_message(conv_id, "user", "Hello world").unwrap();
        let msg2 = store
            .insert_message(conv_id, "assistant", "Hi there!")
            .unwrap();

        assert!(msg1 > 0);
        assert!(msg2 > msg1);

        let messages = store.get_messages(conv_id).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "Hello world");
        assert_eq!(messages[0].seq, 1);
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].seq, 2);
    }

    #[test]
    fn get_or_create_is_idempotent() {
        let db = test_db();
        let store = ConversationStore::new(db);

        let id1 = store.get_or_create_conversation("sess-1").unwrap();
        let id2 = store.get_or_create_conversation("sess-1").unwrap();
        assert_eq!(id1, id2);

        let id3 = store.get_or_create_conversation("sess-2").unwrap();
        assert_ne!(id1, id3);
    }

    #[test]
    fn message_count_and_tokens() {
        let db = test_db();
        let store = ConversationStore::new(db);

        let conv_id = store.get_or_create_conversation("test").unwrap();
        assert_eq!(store.message_count(conv_id).unwrap(), 0);
        assert_eq!(store.total_tokens(conv_id).unwrap(), 0);

        store.insert_message(conv_id, "user", "a]").unwrap();
        store
            .insert_message(conv_id, "assistant", "hello world this is a longer message")
            .unwrap();

        assert_eq!(store.message_count(conv_id).unwrap(), 2);
        assert!(store.total_tokens(conv_id).unwrap() > 0);
    }

    #[test]
    fn insert_and_retrieve_parts() {
        let db = test_db();
        let store = ConversationStore::new(db);

        let conv_id = store.get_or_create_conversation("test").unwrap();
        let msg_id = store.insert_message(conv_id, "assistant", "thinking...").unwrap();

        store
            .insert_message_part(
                msg_id,
                "test",
                "text",
                0,
                Some("thinking..."),
                None,
                None,
                None,
                None,
            )
            .unwrap();

        store
            .insert_message_part(
                msg_id,
                "test",
                "tool",
                1,
                None,
                Some("call_123"),
                Some("search"),
                Some(r#"{"q":"test"}"#),
                Some("found it"),
            )
            .unwrap();

        let parts = store.get_message_parts(msg_id).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].part_type, "text");
        assert_eq!(parts[1].part_type, "tool");
        assert_eq!(parts[1].tool_call_id.as_deref(), Some("call_123"));
        assert_eq!(parts[1].tool_name.as_deref(), Some("search"));
    }

    #[test]
    fn search_messages_like_fallback() {
        let db = test_db();
        let store = ConversationStore::new(db);

        let conv_id = store.get_or_create_conversation("test").unwrap();
        store.insert_message(conv_id, "user", "My dog Barkley is a golden retriever").unwrap();
        store.insert_message(conv_id, "assistant", "Nice dog!").unwrap();
        store.insert_message(conv_id, "user", "Tell me about cats").unwrap();

        let results = store.search_messages(Some(conv_id), "Barkley", false, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Barkley"));

        let results = store.search_messages(Some(conv_id), "dog", false, 10).unwrap();
        assert_eq!(results.len(), 2); // both "My dog Barkley" and "Nice dog!"
    }
}
