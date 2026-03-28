//! `lcm_describe` tool — inspect a summary by ID with lineage tree.

use crate::engine::LcmEngine;
use serde_json::json;
use std::sync::Arc;

pub struct LcmDescribeTool {
    engine: Arc<LcmEngine>,
}

impl LcmDescribeTool {
    pub fn new(engine: Arc<LcmEngine>) -> Self {
        Self { engine }
    }

    pub fn name() -> &'static str {
        "lcm_describe"
    }

    pub fn description() -> &'static str {
        "Inspect a specific summary by ID. Returns full content, metadata, and lineage \
         (parent/child summary IDs, source message IDs). Cheap — no sub-agent needed."
    }

    pub fn parameters_schema() -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Summary ID (e.g., sum_abc123) to inspect"
                }
            },
            "required": ["id"]
        })
    }

    pub fn execute(&self, args: serde_json::Value) -> anyhow::Result<String> {
        let id = args["id"].as_str().unwrap_or("");

        if id.is_empty() {
            return Ok("Error: id is required".to_string());
        }

        let summary = self.engine.summary_store().get_summary(id)?;
        let Some(s) = summary else {
            return Ok(format!("No summary found with id \"{id}\""));
        };

        let mut output = format!(
            "## Summary: {}\n\n\
             - Kind: {}\n\
             - Depth: {}\n\
             - Tokens: {}\n\
             - Descendants: {} (tokens: {})\n\
             - Time range: {} — {}\n\
             - Model: {}\n\
             - Created: {}\n\n",
            s.summary_id,
            s.kind,
            s.depth,
            s.token_count,
            s.descendant_count,
            s.descendant_token_count,
            s.earliest_at.as_deref().unwrap_or("?"),
            s.latest_at.as_deref().unwrap_or("?"),
            s.model,
            s.created_at,
        );

        // Lineage
        if s.kind == "leaf" {
            let msg_ids = self.engine.summary_store().get_source_message_ids(id)?;
            if !msg_ids.is_empty() {
                output.push_str(&format!(
                    "### Source messages ({})\n",
                    msg_ids.len()
                ));
                for mid in &msg_ids {
                    output.push_str(&format!("- msg:{mid}\n"));
                }
                output.push('\n');
            }
        } else {
            let child_ids = self.engine.summary_store().get_child_summary_ids(id)?;
            if !child_ids.is_empty() {
                output.push_str(&format!(
                    "### Child summaries ({})\n",
                    child_ids.len()
                ));
                for cid in &child_ids {
                    output.push_str(&format!("- {cid}\n"));
                }
                output.push('\n');
            }
        }

        // Full content
        output.push_str("### Content\n\n");
        output.push_str(&s.content);
        output.push('\n');

        Ok(output)
    }
}
