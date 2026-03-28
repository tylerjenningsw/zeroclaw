//! `lcm_expand_query` tool — deep recall via direct Provider::chat() call.
//!
//! Gathers DAG source text for specified summaries, constructs a focused
//! prompt with the user's question, and makes a single LLM call.
//! No sub-agent RPC needed — follows the LlmTaskTool pattern.

use crate::engine::LcmEngine;
use crate::compaction::SummarizeFn;
use serde_json::json;
use std::sync::Arc;

pub struct LcmExpandQueryTool {
    engine: Arc<LcmEngine>,
    /// Callback for LLM calls (shared with compaction engine).
    llm_call: SummarizeFn,
}

impl LcmExpandQueryTool {
    pub fn new(engine: Arc<LcmEngine>, llm_call: SummarizeFn) -> Self {
        Self { engine, llm_call }
    }

    pub fn name() -> &'static str {
        "lcm_expand_query"
    }

    pub fn description() -> &'static str {
        "Deep recall: expands compressed summaries and answers a question about \
         conversation history. Use when you need specific details from compacted context. \
         Provide either summary IDs or a search query, plus a focused prompt."
    }

    pub fn parameters_schema() -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The question to answer about conversation history"
                },
                "summary_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Summary IDs to expand (e.g., [\"sum_abc123\"])"
                },
                "query": {
                    "type": "string",
                    "description": "Text query to find relevant summaries (alternative to summary_ids)"
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Max tokens for the answer. Default: 2000",
                    "default": 2000
                }
            },
            "required": ["prompt"]
        })
    }

    pub async fn execute(
        &self,
        args: serde_json::Value,
        session_id: Option<&str>,
    ) -> anyhow::Result<String> {
        let prompt = args["prompt"].as_str().unwrap_or("");
        let max_tokens = args["max_tokens"].as_u64().unwrap_or(2000) as usize;

        if prompt.is_empty() {
            return Ok("Error: prompt is required".to_string());
        }

        let sid = session_id.unwrap_or("default");
        let conv_id = self.engine.conversation_store().get_or_create_conversation(sid)?;

        // Gather source material
        let mut source_text = String::new();
        let mut cited_ids: Vec<String> = Vec::new();

        // If summary_ids provided, expand those directly
        if let Some(ids) = args["summary_ids"].as_array() {
            for id_val in ids {
                if let Some(id) = id_val.as_str() {
                    if let Some(expanded) = self.expand_summary(id)? {
                        source_text.push_str(&expanded);
                        source_text.push_str("\n\n---\n\n");
                        cited_ids.push(id.to_string());
                    }
                }
            }
        }

        // If query provided (or no summary_ids), search for relevant summaries
        if source_text.is_empty() {
            let query = args["query"].as_str().unwrap_or(prompt);
            let use_fts = self.engine.fts5_available();
            let summaries = self.engine.summary_store().search_summaries(
                Some(conv_id),
                query,
                use_fts,
                5,
            )?;
            for s in &summaries {
                if let Some(expanded) = self.expand_summary(&s.summary_id)? {
                    source_text.push_str(&expanded);
                    source_text.push_str("\n\n---\n\n");
                    cited_ids.push(s.summary_id.clone());
                }
            }

            // Also search raw messages
            let messages = self.engine.conversation_store().search_messages(
                Some(conv_id),
                query,
                use_fts,
                10,
            )?;
            for m in &messages {
                source_text.push_str(&format!(
                    "[msg:{}, {}, seq:{}] {}\n\n",
                    m.message_id, m.role, m.seq, m.content
                ));
            }
        }

        if source_text.is_empty() {
            return Ok("No relevant conversation history found for this query.".to_string());
        }

        // Make a single LLM call with the expanded context
        let system = "You answer questions about conversation history using the provided \
                      source material. Be precise and cite summary IDs when referencing \
                      compressed content. If the source material doesn't contain the answer, \
                      say so clearly."
            .to_string();

        let user_prompt = format!(
            "Source material from conversation history:\n\n{source_text}\n\n\
             ---\n\nQuestion: {prompt}\n\n\
             Answer concisely. Cite summary IDs (e.g., sum_xxx) when drawing from summaries."
        );

        let answer = (self.llm_call)(system, user_prompt, 0.3, max_tokens).await?;

        let mut output = answer;
        if !cited_ids.is_empty() {
            output.push_str(&format!("\n\n[Cited: {}]", cited_ids.join(", ")));
        }

        Ok(output)
    }

    /// Expand a summary: return its content plus source material (child summaries or messages).
    fn expand_summary(&self, summary_id: &str) -> anyhow::Result<Option<String>> {
        let Some(summary) = self.engine.summary_store().get_summary(summary_id)? else {
            return Ok(None);
        };

        let mut text = format!(
            "<expanded_summary id=\"{}\" kind=\"{}\" depth=\"{}\">\n{}\n",
            summary.summary_id, summary.kind, summary.depth, summary.content
        );

        // Walk one level down for more detail
        if summary.kind == "leaf" {
            let msg_ids = self.engine.summary_store().get_source_message_ids(summary_id)?;
            let conv_id = summary.conversation_id;
            let messages = self.engine.conversation_store().get_messages(conv_id)?;
            for mid in msg_ids.iter().take(20) {
                if let Some(msg) = messages.iter().find(|m| m.message_id == *mid) {
                    text.push_str(&format!(
                        "\n[source msg:{}, {}] {}\n",
                        msg.message_id, msg.role, msg.content
                    ));
                }
            }
        } else {
            let child_ids = self.engine.summary_store().get_child_summary_ids(summary_id)?;
            for cid in child_ids.iter().take(10) {
                if let Some(child) = self.engine.summary_store().get_summary(cid)? {
                    text.push_str(&format!(
                        "\n[child {} depth:{}] {}\n",
                        child.summary_id, child.depth, child.content
                    ));
                }
            }
        }

        text.push_str("</expanded_summary>");
        Ok(Some(text))
    }
}
