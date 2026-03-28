//! `lcm_grep` tool — search messages and summaries by regex or full-text.

use crate::engine::LcmEngine;
use serde_json::json;
use std::sync::Arc;

pub struct LcmGrepTool {
    engine: Arc<LcmEngine>,
}

impl LcmGrepTool {
    pub fn new(engine: Arc<LcmEngine>) -> Self {
        Self { engine }
    }

    pub fn name() -> &'static str {
        "lcm_grep"
    }

    pub fn description() -> &'static str {
        "Search conversation history (messages and summaries) by text query. \
         Returns matching excerpts with IDs for use with lcm_describe."
    }

    pub fn parameters_schema() -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to search for in messages and summaries"
                },
                "scope": {
                    "type": "string",
                    "enum": ["messages", "summaries", "both"],
                    "description": "Where to search. Default: both",
                    "default": "both"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return. Default: 20",
                    "default": 20
                }
            },
            "required": ["query"]
        })
    }

    pub fn execute(
        &self,
        args: serde_json::Value,
        session_id: Option<&str>,
    ) -> anyhow::Result<String> {
        let query = args["query"].as_str().unwrap_or("");
        let scope = args["scope"].as_str().unwrap_or("both");
        let limit = args["limit"].as_u64().unwrap_or(20) as usize;

        if query.is_empty() {
            return Ok("Error: query is required".to_string());
        }

        let sid = session_id.unwrap_or("default");
        let conv_id = self.engine.conversation_store().get_or_create_conversation(sid)?;
        let use_fts = self.engine.fts5_available();

        let mut output = String::new();

        if scope == "messages" || scope == "both" {
            let messages = self.engine.conversation_store().search_messages(
                Some(conv_id),
                query,
                use_fts,
                limit,
            )?;
            if !messages.is_empty() {
                output.push_str(&format!("## Messages ({} matches)\n\n", messages.len()));
                for msg in &messages {
                    let snippet = truncate_snippet(&msg.content, 200);
                    output.push_str(&format!(
                        "- [msg:{}] ({}, seq:{}) {}\n",
                        msg.message_id, msg.role, msg.seq, snippet
                    ));
                }
                output.push('\n');
            }
        }

        if scope == "summaries" || scope == "both" {
            let summaries = self.engine.summary_store().search_summaries(
                Some(conv_id),
                query,
                use_fts,
                limit,
            )?;
            if !summaries.is_empty() {
                output.push_str(&format!("## Summaries ({} matches)\n\n", summaries.len()));
                for s in &summaries {
                    let snippet = truncate_snippet(&s.content, 200);
                    output.push_str(&format!(
                        "- [{}] (depth:{}, {}) {}\n",
                        s.summary_id, s.depth, s.kind, snippet
                    ));
                }
                output.push('\n');
            }
        }

        if output.is_empty() {
            output = format!("No results found for \"{query}\"");
        }

        Ok(output)
    }
}

fn truncate_snippet(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.replace('\n', " ")
    } else {
        format!("{}...", &text[..max_len].replace('\n', " "))
    }
}
