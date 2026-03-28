use crate::config::estimate_tokens;
use crate::db::connection::LcmDatabase;
use anyhow::Result;
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// A summary DAG node.
#[derive(Debug, Clone)]
pub struct SummaryRow {
    pub summary_id: String,
    pub conversation_id: i64,
    pub kind: String,
    pub depth: i64,
    pub content: String,
    pub token_count: i64,
    pub earliest_at: Option<String>,
    pub latest_at: Option<String>,
    pub descendant_count: i64,
    pub descendant_token_count: i64,
    pub source_message_token_count: i64,
    pub model: String,
    pub created_at: String,
}

/// A context item — either a message or a summary reference.
#[derive(Debug, Clone)]
pub struct ContextItemRow {
    pub conversation_id: i64,
    pub ordinal: i64,
    pub item_type: String,
    pub message_id: Option<i64>,
    pub summary_id: Option<String>,
}

/// Generate a summary ID: `sum_` + first 16 hex chars of SHA-256 of content.
fn generate_summary_id(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let hash = hasher.finalize();
    format!("sum_{}", hex::encode(&hash[..8]))
}

/// CRUD operations for summaries and the context item list.
pub struct SummaryStore {
    db: Arc<LcmDatabase>,
}

impl SummaryStore {
    pub fn new(db: Arc<LcmDatabase>) -> Self {
        Self { db }
    }

    /// Insert a leaf summary linked to source messages.
    ///
    /// Returns the generated summary_id.
    pub fn insert_leaf_summary(
        &self,
        conversation_id: i64,
        content: &str,
        model: &str,
        source_message_ids: &[i64],
        earliest_at: Option<&str>,
        latest_at: Option<&str>,
    ) -> Result<String> {
        let summary_id = generate_summary_id(content);
        let token_count = estimate_tokens(content) as i64;
        let source_token_count: i64 = source_message_ids.len() as i64 * 200; // rough estimate

        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, token_count, model, earliest_at, latest_at, source_message_token_count)
                 VALUES (?1, ?2, 'leaf', 0, ?3, ?4, ?5, ?6, ?7, ?8)",
                rusqlite::params![
                    summary_id,
                    conversation_id,
                    content,
                    token_count,
                    model,
                    earliest_at,
                    latest_at,
                    source_token_count,
                ],
            )?;

            // Link to source messages
            for (i, msg_id) in source_message_ids.iter().enumerate() {
                conn.execute(
                    "INSERT INTO summary_messages (summary_id, message_id, ordinal) VALUES (?1, ?2, ?3)",
                    rusqlite::params![summary_id, msg_id, i as i64],
                )?;
            }

            Ok(summary_id)
        })
    }

    /// Insert a condensed summary linked to child summaries.
    pub fn insert_condensed_summary(
        &self,
        conversation_id: i64,
        depth: i64,
        content: &str,
        model: &str,
        child_summary_ids: &[String],
        earliest_at: Option<&str>,
        latest_at: Option<&str>,
    ) -> Result<String> {
        let summary_id = generate_summary_id(content);
        let token_count = estimate_tokens(content) as i64;

        self.db.with_conn(|conn| {
            // Calculate descendant stats from children
            let mut descendant_count: i64 = 0;
            let mut descendant_token_count: i64 = 0;
            for child_id in child_summary_ids {
                let (dc, dtc): (i64, i64) = conn.query_row(
                    "SELECT descendant_count, descendant_token_count FROM summaries WHERE summary_id = ?1",
                    [child_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                ).unwrap_or((0, 0));
                descendant_count += dc + 1;
                descendant_token_count += dtc;
            }

            conn.execute(
                "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, token_count, model, earliest_at, latest_at, descendant_count, descendant_token_count)
                 VALUES (?1, ?2, 'condensed', ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                rusqlite::params![
                    summary_id,
                    conversation_id,
                    depth,
                    content,
                    token_count,
                    model,
                    earliest_at,
                    latest_at,
                    descendant_count,
                    descendant_token_count,
                ],
            )?;

            // Link to child summaries
            for (i, child_id) in child_summary_ids.iter().enumerate() {
                conn.execute(
                    "INSERT INTO summary_parents (summary_id, parent_summary_id, ordinal) VALUES (?1, ?2, ?3)",
                    rusqlite::params![child_id, summary_id, i as i64],
                )?;
            }

            Ok(summary_id)
        })
    }

    /// Get all summaries for a conversation, ordered by depth then created_at.
    pub fn get_summaries(&self, conversation_id: i64) -> Result<Vec<SummaryRow>> {
        self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT summary_id, conversation_id, kind, depth, content, token_count,
                        earliest_at, latest_at, descendant_count, descendant_token_count,
                        source_message_token_count, model, created_at
                 FROM summaries WHERE conversation_id = ?1
                 ORDER BY depth ASC, created_at ASC",
            )?;
            let rows = stmt
                .query_map([conversation_id], |row| {
                    Ok(SummaryRow {
                        summary_id: row.get(0)?,
                        conversation_id: row.get(1)?,
                        kind: row.get(2)?,
                        depth: row.get(3)?,
                        content: row.get(4)?,
                        token_count: row.get(5)?,
                        earliest_at: row.get(6)?,
                        latest_at: row.get(7)?,
                        descendant_count: row.get(8)?,
                        descendant_token_count: row.get(9)?,
                        source_message_token_count: row.get(10)?,
                        model: row.get(11)?,
                        created_at: row.get(12)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows)
        })
    }

    /// Get a single summary by ID.
    pub fn get_summary(&self, summary_id: &str) -> Result<Option<SummaryRow>> {
        self.db.with_conn(|conn| {
            let result = conn
                .query_row(
                    "SELECT summary_id, conversation_id, kind, depth, content, token_count,
                            earliest_at, latest_at, descendant_count, descendant_token_count,
                            source_message_token_count, model, created_at
                     FROM summaries WHERE summary_id = ?1",
                    [summary_id],
                    |row| {
                        Ok(SummaryRow {
                            summary_id: row.get(0)?,
                            conversation_id: row.get(1)?,
                            kind: row.get(2)?,
                            depth: row.get(3)?,
                            content: row.get(4)?,
                            token_count: row.get(5)?,
                            earliest_at: row.get(6)?,
                            latest_at: row.get(7)?,
                            descendant_count: row.get(8)?,
                            descendant_token_count: row.get(9)?,
                            source_message_token_count: row.get(10)?,
                            model: row.get(11)?,
                            created_at: row.get(12)?,
                        })
                    },
                )
                .ok();
            Ok(result)
        })
    }

    /// Get child summary IDs for a condensed summary (via summary_parents).
    pub fn get_child_summary_ids(&self, parent_summary_id: &str) -> Result<Vec<String>> {
        self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT summary_id FROM summary_parents WHERE parent_summary_id = ?1 ORDER BY ordinal ASC",
            )?;
            let ids = stmt
                .query_map([parent_summary_id], |row| row.get(0))?
                .collect::<rusqlite::Result<Vec<String>>>()?;
            Ok(ids)
        })
    }

    /// Get source message IDs for a leaf summary.
    pub fn get_source_message_ids(&self, summary_id: &str) -> Result<Vec<i64>> {
        self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT message_id FROM summary_messages WHERE summary_id = ?1 ORDER BY ordinal ASC",
            )?;
            let ids = stmt
                .query_map([summary_id], |row| row.get(0))?
                .collect::<rusqlite::Result<Vec<i64>>>()?;
            Ok(ids)
        })
    }

    /// Get the max depth of summaries for a conversation.
    pub fn max_depth(&self, conversation_id: i64) -> Result<i64> {
        self.db.with_conn(|conn| {
            let depth: i64 = conn
                .query_row(
                    "SELECT COALESCE(MAX(depth), -1) FROM summaries WHERE conversation_id = ?1",
                    [conversation_id],
                    |row| row.get(0),
                )
                .unwrap_or(-1);
            Ok(depth)
        })
    }

    /// Get all context items for a conversation, ordered by ordinal.
    pub fn get_context_items(&self, conversation_id: i64) -> Result<Vec<ContextItemRow>> {
        self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT conversation_id, ordinal, item_type, message_id, summary_id
                 FROM context_items WHERE conversation_id = ?1 ORDER BY ordinal ASC",
            )?;
            let rows = stmt
                .query_map([conversation_id], |row| {
                    Ok(ContextItemRow {
                        conversation_id: row.get(0)?,
                        ordinal: row.get(1)?,
                        item_type: row.get(2)?,
                        message_id: row.get(3)?,
                        summary_id: row.get(4)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows)
        })
    }

    /// Append a message context item (at the end of the list).
    pub fn append_message_context_item(
        &self,
        conversation_id: i64,
        message_id: i64,
    ) -> Result<()> {
        self.db.with_conn(|conn| {
            let next_ordinal: i64 = conn
                .query_row(
                    "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM context_items WHERE conversation_id = ?1",
                    [conversation_id],
                    |row| row.get(0),
                )
                .unwrap_or(1);

            conn.execute(
                "INSERT INTO context_items (conversation_id, ordinal, item_type, message_id) VALUES (?1, ?2, 'message', ?3)",
                rusqlite::params![conversation_id, next_ordinal, message_id],
            )?;
            Ok(())
        })
    }

    /// Replace a range of context items with a single summary item.
    ///
    /// Removes items with ordinals in `[start_ordinal, end_ordinal]` and inserts
    /// a summary item at `start_ordinal`.  Remaining items above `end_ordinal`
    /// are shifted down.
    pub fn replace_context_range_with_summary(
        &self,
        conversation_id: i64,
        start_ordinal: i64,
        end_ordinal: i64,
        summary_id: &str,
    ) -> Result<()> {
        self.db.with_conn(|conn| {
            let range_size = end_ordinal - start_ordinal + 1;
            let shift = range_size - 1; // we keep one slot for the summary

            // Delete the range
            conn.execute(
                "DELETE FROM context_items WHERE conversation_id = ?1 AND ordinal BETWEEN ?2 AND ?3",
                rusqlite::params![conversation_id, start_ordinal, end_ordinal],
            )?;

            // Insert the summary at start_ordinal
            conn.execute(
                "INSERT INTO context_items (conversation_id, ordinal, item_type, summary_id) VALUES (?1, ?2, 'summary', ?3)",
                rusqlite::params![conversation_id, start_ordinal, summary_id],
            )?;

            // Shift items above end_ordinal down by (range_size - 1)
            conn.execute(
                "UPDATE context_items SET ordinal = ordinal - ?1 WHERE conversation_id = ?2 AND ordinal > ?3",
                rusqlite::params![shift, conversation_id, end_ordinal],
            )?;

            Ok(())
        })
    }

    /// Search summaries by content using FTS5 or LIKE fallback.
    pub fn search_summaries(
        &self,
        conversation_id: Option<i64>,
        query: &str,
        use_fts: bool,
        limit: usize,
    ) -> Result<Vec<SummaryRow>> {
        self.db.with_conn(|conn| {
            if use_fts {
                let sql = if let Some(conv_id) = conversation_id {
                    format!(
                        "SELECT s.summary_id, s.conversation_id, s.kind, s.depth, s.content, s.token_count,
                                s.earliest_at, s.latest_at, s.descendant_count, s.descendant_token_count,
                                s.source_message_token_count, s.model, s.created_at
                         FROM summaries_fts f
                         JOIN summaries s ON s.rowid = f.rowid
                         WHERE f.content MATCH ?1 AND s.conversation_id = {conv_id}
                         ORDER BY rank LIMIT ?2"
                    )
                } else {
                    "SELECT s.summary_id, s.conversation_id, s.kind, s.depth, s.content, s.token_count,
                            s.earliest_at, s.latest_at, s.descendant_count, s.descendant_token_count,
                            s.source_message_token_count, s.model, s.created_at
                     FROM summaries_fts f
                     JOIN summaries s ON s.rowid = f.rowid
                     WHERE f.content MATCH ?1
                     ORDER BY rank LIMIT ?2"
                        .to_string()
                };
                let mut stmt = conn.prepare(&sql)?;
                let rows = stmt.query_map(rusqlite::params![query, limit as i64], map_summary_row)?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(rows)
            } else {
                let like_pattern = format!("%{query}%");
                let sql = if let Some(conv_id) = conversation_id {
                    format!(
                        "SELECT summary_id, conversation_id, kind, depth, content, token_count,
                                earliest_at, latest_at, descendant_count, descendant_token_count,
                                source_message_token_count, model, created_at
                         FROM summaries
                         WHERE content LIKE ?1 AND conversation_id = {conv_id}
                         ORDER BY created_at DESC LIMIT ?2"
                    )
                } else {
                    "SELECT summary_id, conversation_id, kind, depth, content, token_count,
                            earliest_at, latest_at, descendant_count, descendant_token_count,
                            source_message_token_count, model, created_at
                     FROM summaries
                     WHERE content LIKE ?1
                     ORDER BY created_at DESC LIMIT ?2"
                        .to_string()
                };
                let mut stmt = conn.prepare(&sql)?;
                let rows = stmt.query_map(rusqlite::params![like_pattern, limit as i64], map_summary_row)?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(rows)
            }
        })
    }

    /// Get summary count for a conversation.
    pub fn summary_count(&self, conversation_id: i64) -> Result<i64> {
        self.db.with_conn(|conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM summaries WHERE conversation_id = ?1",
                [conversation_id],
                |row| row.get(0),
            )?;
            Ok(count)
        })
    }
}

fn map_summary_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SummaryRow> {
    Ok(SummaryRow {
        summary_id: row.get(0)?,
        conversation_id: row.get(1)?,
        kind: row.get(2)?,
        depth: row.get(3)?,
        content: row.get(4)?,
        token_count: row.get(5)?,
        earliest_at: row.get(6)?,
        latest_at: row.get(7)?,
        descendant_count: row.get(8)?,
        descendant_token_count: row.get(9)?,
        source_message_token_count: row.get(10)?,
        model: row.get(11)?,
        created_at: row.get(12)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::conversation::ConversationStore;

    fn test_db() -> Arc<LcmDatabase> {
        Arc::new(LcmDatabase::open_in_memory().unwrap())
    }

    #[test]
    fn insert_leaf_summary() {
        let db = test_db();
        let conv_store = ConversationStore::new(db.clone());
        let sum_store = SummaryStore::new(db);

        let conv_id = conv_store.get_or_create_conversation("test").unwrap();
        let msg1 = conv_store.insert_message(conv_id, "user", "Hello").unwrap();
        let msg2 = conv_store
            .insert_message(conv_id, "assistant", "World")
            .unwrap();

        let summary_id = sum_store
            .insert_leaf_summary(
                conv_id,
                "User greeted, assistant responded.",
                "haiku",
                &[msg1, msg2],
                Some("2026-01-01T00:00:00Z"),
                Some("2026-01-01T00:01:00Z"),
            )
            .unwrap();

        assert!(summary_id.starts_with("sum_"));

        let summary = sum_store.get_summary(&summary_id).unwrap().unwrap();
        assert_eq!(summary.kind, "leaf");
        assert_eq!(summary.depth, 0);
        assert!(summary.content.contains("greeted"));

        let source_ids = sum_store.get_source_message_ids(&summary_id).unwrap();
        assert_eq!(source_ids, vec![msg1, msg2]);
    }

    #[test]
    fn insert_condensed_summary() {
        let db = test_db();
        let conv_store = ConversationStore::new(db.clone());
        let sum_store = SummaryStore::new(db);

        let conv_id = conv_store.get_or_create_conversation("test").unwrap();
        let msg1 = conv_store.insert_message(conv_id, "user", "A").unwrap();
        let msg2 = conv_store.insert_message(conv_id, "user", "B").unwrap();

        let leaf1 = sum_store
            .insert_leaf_summary(conv_id, "Summary A", "haiku", &[msg1], None, None)
            .unwrap();
        let leaf2 = sum_store
            .insert_leaf_summary(conv_id, "Summary B", "haiku", &[msg2], None, None)
            .unwrap();

        let condensed = sum_store
            .insert_condensed_summary(
                conv_id,
                1,
                "Combined summary of A and B",
                "haiku",
                &[leaf1.clone(), leaf2.clone()],
                None,
                None,
            )
            .unwrap();

        let summary = sum_store.get_summary(&condensed).unwrap().unwrap();
        assert_eq!(summary.kind, "condensed");
        assert_eq!(summary.depth, 1);
        assert!(summary.descendant_count >= 2);

        let children = sum_store.get_child_summary_ids(&condensed).unwrap();
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn context_items_lifecycle() {
        let db = test_db();
        let conv_store = ConversationStore::new(db.clone());
        let sum_store = SummaryStore::new(db);

        let conv_id = conv_store.get_or_create_conversation("test").unwrap();
        let msg1 = conv_store.insert_message(conv_id, "user", "A").unwrap();
        let msg2 = conv_store.insert_message(conv_id, "user", "B").unwrap();
        let msg3 = conv_store.insert_message(conv_id, "user", "C").unwrap();

        // Add all as context items
        sum_store.append_message_context_item(conv_id, msg1).unwrap();
        sum_store.append_message_context_item(conv_id, msg2).unwrap();
        sum_store.append_message_context_item(conv_id, msg3).unwrap();

        let items = sum_store.get_context_items(conv_id).unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].item_type, "message");

        // Summarize first two messages
        let summary_id = sum_store
            .insert_leaf_summary(conv_id, "Summary of A and B", "haiku", &[msg1, msg2], None, None)
            .unwrap();

        // Replace first two context items with the summary
        sum_store
            .replace_context_range_with_summary(conv_id, 1, 2, &summary_id)
            .unwrap();

        let items = sum_store.get_context_items(conv_id).unwrap();
        assert_eq!(items.len(), 2); // summary + msg3
        assert_eq!(items[0].item_type, "summary");
        assert_eq!(items[0].summary_id.as_deref(), Some(summary_id.as_str()));
        assert_eq!(items[1].item_type, "message");
        assert_eq!(items[1].message_id, Some(msg3));
    }

    #[test]
    fn max_depth_empty() {
        let db = test_db();
        let conv_store = ConversationStore::new(db.clone());
        let sum_store = SummaryStore::new(db);

        let conv_id = conv_store.get_or_create_conversation("test").unwrap();
        assert_eq!(sum_store.max_depth(conv_id).unwrap(), -1);
    }
}
