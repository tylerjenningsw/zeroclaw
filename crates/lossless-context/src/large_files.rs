//! Large file externalization — intercepts files >threshold tokens,
//! stores them on disk, and replaces with compact references.

use crate::config::estimate_tokens;
use crate::db::connection::LcmDatabase;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Manages large file externalization to disk.
pub struct LargeFileStore {
    db: Arc<LcmDatabase>,
    /// Root directory for externalized files: `{workspace}/lcm-files/`
    storage_root: PathBuf,
    /// Token threshold for externalization.
    threshold_tokens: usize,
}

impl LargeFileStore {
    pub fn new(db: Arc<LcmDatabase>, workspace_dir: &Path, threshold_tokens: usize) -> Self {
        Self {
            db,
            storage_root: workspace_dir.join("lcm-files"),
            threshold_tokens,
        }
    }

    /// Check if content exceeds the large file threshold.
    pub fn is_large(&self, content: &str) -> bool {
        estimate_tokens(content) > self.threshold_tokens
    }

    /// Externalize a large file: write to disk, store metadata in SQLite,
    /// return the file_id and a compact reference string.
    pub fn externalize(
        &self,
        conversation_id: i64,
        content: &str,
        file_name: Option<&str>,
        mime_type: Option<&str>,
    ) -> Result<(String, String)> {
        let file_id = format!("file_{}", uuid::Uuid::new_v4().simple());

        // Create storage directory
        let conv_dir = self.storage_root.join(conversation_id.to_string());
        std::fs::create_dir_all(&conv_dir)
            .with_context(|| format!("creating LCM file storage at {conv_dir:?}"))?;

        // Determine extension
        let ext = file_name
            .and_then(|n| Path::new(n).extension())
            .and_then(|e| e.to_str())
            .unwrap_or("txt");

        let file_path = conv_dir.join(format!("{file_id}.{ext}"));
        std::fs::write(&file_path, content)
            .with_context(|| format!("writing externalized file to {file_path:?}"))?;

        let storage_uri = file_path.to_string_lossy().to_string();
        let byte_size = content.len() as i64;

        // Generate a brief exploration summary (~200 tokens)
        let exploration_summary = generate_exploration_summary(content);

        // Store metadata in SQLite
        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO large_files (file_id, conversation_id, file_name, mime_type, byte_size, storage_uri, exploration_summary)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    file_id,
                    conversation_id,
                    file_name,
                    mime_type,
                    byte_size,
                    storage_uri,
                    exploration_summary,
                ],
            )?;
            Ok(())
        })?;

        // Build compact reference
        let reference = format!(
            "[Large file stored as {file_id}; {} bytes, ~{} tokens. {}]",
            byte_size,
            estimate_tokens(content),
            exploration_summary
        );

        Ok((file_id, reference))
    }

    /// Retrieve externalized file content by ID.
    pub fn retrieve(&self, file_id: &str) -> Result<Option<String>> {
        let uri: Option<String> = self.db.with_conn(|conn| {
            Ok(conn
                .query_row(
                    "SELECT storage_uri FROM large_files WHERE file_id = ?1",
                    [file_id],
                    |row| row.get(0),
                )
                .ok())
        })?;

        match uri {
            Some(path) => {
                let content = std::fs::read_to_string(&path)
                    .with_context(|| format!("reading externalized file at {path}"))?;
                Ok(Some(content))
            }
            None => Ok(None),
        }
    }
}

/// Generate a brief structural summary of file content (~200 tokens).
fn generate_exploration_summary(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let total_chars = content.len();

    let mut summary = format!("{total_lines} lines, {total_chars} chars. ");

    // Show first few lines as preview
    let preview_lines: Vec<&str> = lines.iter().take(5).copied().collect();
    if !preview_lines.is_empty() {
        summary.push_str("Starts with: ");
        let preview = preview_lines.join(" | ");
        if preview.len() > 300 {
            summary.push_str(&preview[..300]);
            summary.push_str("...");
        } else {
            summary.push_str(&preview);
        }
    }

    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::conversation::ConversationStore;

    #[test]
    fn externalize_and_retrieve() {
        let db = Arc::new(LcmDatabase::open_in_memory().unwrap());
        let conv_store = ConversationStore::new(db.clone());
        let conv_id = conv_store.get_or_create_conversation("test").unwrap();

        let tmp = tempfile::tempdir().unwrap();
        let store = LargeFileStore::new(db, tmp.path(), 100);

        let content = "x".repeat(1000);
        let (file_id, reference) = store
            .externalize(conv_id, &content, Some("data.txt"), Some("text/plain"))
            .unwrap();

        assert!(file_id.starts_with("file_"));
        assert!(reference.contains(&file_id));
        assert!(reference.contains("1000 bytes"));

        let retrieved = store.retrieve(&file_id).unwrap().unwrap();
        assert_eq!(retrieved, content);
    }

    #[test]
    fn is_large_threshold() {
        let db = Arc::new(LcmDatabase::open_in_memory().unwrap());
        let tmp = tempfile::tempdir().unwrap();
        let store = LargeFileStore::new(db, tmp.path(), 100);

        assert!(!store.is_large("short text"));
        assert!(store.is_large(&"x".repeat(500))); // 500 chars ≈ 125 tokens > 100
    }
}
