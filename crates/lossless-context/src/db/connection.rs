use crate::db::schema;
use anyhow::{Context, Result};
use rusqlite::Connection;
use std::path::Path;
use std::sync::Mutex;

/// Thread-safe wrapper around a SQLite connection in WAL mode.
pub struct LcmDatabase {
    conn: Mutex<Connection>,
    fts5_available: bool,
}

impl LcmDatabase {
    /// Open (or create) the LCM database at the given path.
    ///
    /// Enables WAL mode, foreign keys, and creates all tables.
    /// FTS5 is attempted but gracefully degraded if unavailable.
    pub fn open(path: &Path) -> Result<Self> {
        let conn =
            Connection::open(path).with_context(|| format!("opening LCM database at {path:?}"))?;

        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;
            PRAGMA busy_timeout = 5000;
            ",
        )
        .context("setting PRAGMA options")?;

        schema::create_tables(&conn).context("creating LCM tables")?;

        let fts5_available = match schema::create_fts_tables(&conn) {
            Ok(()) => {
                tracing::debug!("LCM FTS5 tables created");
                true
            }
            Err(e) => {
                tracing::info!(error = %e, "FTS5 not available, falling back to LIKE search");
                false
            }
        };

        Ok(Self {
            conn: Mutex::new(conn),
            fts5_available,
        })
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("opening in-memory LCM database")?;

        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .context("setting PRAGMA options")?;

        schema::create_tables(&conn).context("creating LCM tables")?;

        let fts5_available = schema::create_fts_tables(&conn).is_ok();

        Ok(Self {
            conn: Mutex::new(conn),
            fts5_available,
        })
    }

    /// Whether FTS5 virtual tables are available for full-text search.
    pub fn fts5_available(&self) -> bool {
        self.fts5_available
    }

    /// Execute a closure with exclusive access to the database connection.
    pub fn with_conn<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        let conn = self
            .conn
            .lock()
            .map_err(|e| anyhow::anyhow!("LCM database lock poisoned: {e}"))?;
        f(&conn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_in_memory_creates_tables() {
        let db = LcmDatabase::open_in_memory().unwrap();
        db.with_conn(|conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%_fts%'",
                [],
                |row| row.get(0),
            )?;
            // 8 tables: conversations, messages, message_parts, summaries,
            //           summary_messages, summary_parents, context_items, large_files
            assert_eq!(count, 8, "expected 8 LCM tables");
            Ok(())
        })
        .unwrap();
    }
}
