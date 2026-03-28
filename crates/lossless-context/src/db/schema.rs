use rusqlite::Connection;

/// Create all LCM tables and indexes.
///
/// 8 tables — deliberately drops `conversation_bootstrap_state` from the
/// upstream 9-table schema.  That table tracks OpenClaw JSONL session file
/// offsets; ZeroClaw uses `SessionBackend` instead.
pub fn create_tables(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            session_key     TEXT,
            title           TEXT,
            bootstrapped_at TEXT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS messages (
            message_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            seq             INTEGER NOT NULL,
            role            TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
            content         TEXT NOT NULL,
            token_count     INTEGER NOT NULL,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE (conversation_id, seq)
        );

        CREATE TABLE IF NOT EXISTS message_parts (
            part_id      TEXT PRIMARY KEY,
            message_id   INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
            session_id   TEXT NOT NULL,
            part_type    TEXT NOT NULL CHECK (part_type IN (
                'text', 'reasoning', 'tool', 'patch', 'file',
                'subtask', 'compaction', 'step_start', 'step_finish',
                'snapshot', 'agent', 'retry'
            )),
            ordinal        INTEGER NOT NULL,
            text_content   TEXT,
            is_ignored     INTEGER,
            is_synthetic   INTEGER,
            tool_call_id   TEXT,
            tool_name      TEXT,
            tool_status    TEXT,
            tool_input     TEXT,
            tool_output    TEXT,
            tool_error     TEXT,
            tool_title     TEXT,
            patch_hash     TEXT,
            patch_files    TEXT,
            file_mime      TEXT,
            file_name      TEXT,
            file_url       TEXT,
            subtask_prompt TEXT,
            subtask_desc   TEXT,
            subtask_agent  TEXT,
            step_reason    TEXT,
            step_cost      REAL,
            step_tokens_in  INTEGER,
            step_tokens_out INTEGER,
            snapshot_hash  TEXT,
            compaction_auto INTEGER,
            metadata       TEXT,
            UNIQUE (message_id, ordinal)
        );

        CREATE TABLE IF NOT EXISTS summaries (
            summary_id                TEXT PRIMARY KEY,
            conversation_id           INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            kind                      TEXT NOT NULL CHECK (kind IN ('leaf', 'condensed')),
            depth                     INTEGER NOT NULL DEFAULT 0,
            content                   TEXT NOT NULL,
            token_count               INTEGER NOT NULL,
            earliest_at               TEXT,
            latest_at                 TEXT,
            descendant_count          INTEGER NOT NULL DEFAULT 0,
            descendant_token_count    INTEGER NOT NULL DEFAULT 0,
            source_message_token_count INTEGER NOT NULL DEFAULT 0,
            model                     TEXT NOT NULL DEFAULT 'unknown',
            file_ids                  TEXT NOT NULL DEFAULT '[]',
            created_at                TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS summary_messages (
            summary_id TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE CASCADE,
            message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE RESTRICT,
            ordinal    INTEGER NOT NULL,
            PRIMARY KEY (summary_id, message_id)
        );

        CREATE TABLE IF NOT EXISTS summary_parents (
            summary_id        TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE CASCADE,
            parent_summary_id TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE RESTRICT,
            ordinal           INTEGER NOT NULL,
            PRIMARY KEY (summary_id, parent_summary_id)
        );

        CREATE TABLE IF NOT EXISTS context_items (
            conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            ordinal         INTEGER NOT NULL,
            item_type       TEXT NOT NULL CHECK (item_type IN ('message', 'summary')),
            message_id      INTEGER REFERENCES messages(message_id) ON DELETE RESTRICT,
            summary_id      TEXT    REFERENCES summaries(summary_id) ON DELETE RESTRICT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (conversation_id, ordinal),
            CHECK (
                (item_type = 'message' AND message_id IS NOT NULL AND summary_id IS NULL) OR
                (item_type = 'summary' AND summary_id IS NOT NULL AND message_id IS NULL)
            )
        );

        CREATE TABLE IF NOT EXISTS large_files (
            file_id         TEXT PRIMARY KEY,
            conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            file_name       TEXT,
            mime_type       TEXT,
            byte_size       INTEGER,
            storage_uri     TEXT NOT NULL,
            exploration_summary TEXT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS messages_conv_seq_idx
            ON messages (conversation_id, seq);
        CREATE INDEX IF NOT EXISTS summaries_conv_created_idx
            ON summaries (conversation_id, created_at);
        CREATE INDEX IF NOT EXISTS summaries_conv_depth_idx
            ON summaries (conversation_id, depth);
        CREATE INDEX IF NOT EXISTS message_parts_message_idx
            ON message_parts (message_id);
        CREATE INDEX IF NOT EXISTS message_parts_type_idx
            ON message_parts (part_type);
        CREATE INDEX IF NOT EXISTS context_items_conv_idx
            ON context_items (conversation_id, ordinal);
        CREATE INDEX IF NOT EXISTS large_files_conv_idx
            ON large_files (conversation_id, created_at);
        ",
    )?;
    Ok(())
}

/// Create FTS5 virtual tables for full-text search if available.
pub fn create_fts_tables(conn: &Connection) -> rusqlite::Result<()> {
    // FTS5 on messages.content
    conn.execute_batch(
        "
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            content,
            content='messages',
            content_rowid='message_id'
        );
        CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content) VALUES (new.message_id, new.content);
        END;
        CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.message_id, old.content);
        END;
        CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE OF content ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.message_id, old.content);
            INSERT INTO messages_fts(rowid, content) VALUES (new.message_id, new.content);
        END;

        CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
            content,
            content='summaries',
            content_rowid='rowid'
        );
        CREATE TRIGGER IF NOT EXISTS summaries_ai AFTER INSERT ON summaries BEGIN
            INSERT INTO summaries_fts(rowid, content) VALUES (new.rowid, new.content);
        END;
        CREATE TRIGGER IF NOT EXISTS summaries_ad AFTER DELETE ON summaries BEGIN
            INSERT INTO summaries_fts(summaries_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
        END;
        CREATE TRIGGER IF NOT EXISTS summaries_au AFTER UPDATE OF content ON summaries BEGIN
            INSERT INTO summaries_fts(summaries_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
            INSERT INTO summaries_fts(rowid, content) VALUES (new.rowid, new.content);
        END;
        ",
    )?;
    Ok(())
}
