//! ZeroClaw tool wrappers for LCM tools.
//!
//! These implement the `Tool` trait and delegate to `lossless-context` crate tools.
//! Feature-gated behind `lossless-context`.

use crate::tools::traits::{Tool, ToolResult};
use async_trait::async_trait;
use lossless_context::compaction::SummarizeFn;
use lossless_context::engine::LcmEngine;
use lossless_context::tools::{lcm_describe, lcm_expand_query, lcm_grep};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Build all LCM tools.  Called during agent setup when the `lossless-context`
/// feature is enabled and `config.agent.lcm.enabled == true`.
pub fn build_lcm_tools(
    engine: Arc<LcmEngine>,
    llm_call: SummarizeFn,
    session_id: Arc<Mutex<Option<String>>>,
) -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(LcmGrepToolWrapper::new(engine.clone(), session_id.clone())),
        Box::new(LcmDescribeToolWrapper::new(engine.clone())),
        Box::new(LcmExpandQueryToolWrapper::new(engine, llm_call, session_id)),
    ]
}

// ── lcm_grep ────────────────────────────────────────────────────────────

pub struct LcmGrepToolWrapper {
    inner: lcm_grep::LcmGrepTool,
    session_id: Arc<Mutex<Option<String>>>,
}

impl LcmGrepToolWrapper {
    pub fn new(engine: Arc<LcmEngine>, session_id: Arc<Mutex<Option<String>>>) -> Self {
        Self {
            inner: lcm_grep::LcmGrepTool::new(engine),
            session_id,
        }
    }
}

#[async_trait]
impl Tool for LcmGrepToolWrapper {
    fn name(&self) -> &str {
        lcm_grep::LcmGrepTool::name()
    }

    fn description(&self) -> &str {
        lcm_grep::LcmGrepTool::description()
    }

    fn parameters_schema(&self) -> serde_json::Value {
        lcm_grep::LcmGrepTool::parameters_schema()
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let sid = self.session_id.lock().await;
        match self.inner.execute(args, sid.as_deref()) {
            Ok(output) => Ok(ToolResult {
                success: true,
                output,
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(e.to_string()),
            }),
        }
    }
}

// ── lcm_describe ────────────────────────────────────────────────────────

pub struct LcmDescribeToolWrapper {
    inner: lcm_describe::LcmDescribeTool,
}

impl LcmDescribeToolWrapper {
    pub fn new(engine: Arc<LcmEngine>) -> Self {
        Self {
            inner: lcm_describe::LcmDescribeTool::new(engine),
        }
    }
}

#[async_trait]
impl Tool for LcmDescribeToolWrapper {
    fn name(&self) -> &str {
        lcm_describe::LcmDescribeTool::name()
    }

    fn description(&self) -> &str {
        lcm_describe::LcmDescribeTool::description()
    }

    fn parameters_schema(&self) -> serde_json::Value {
        lcm_describe::LcmDescribeTool::parameters_schema()
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        match self.inner.execute(args) {
            Ok(output) => Ok(ToolResult {
                success: true,
                output,
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(e.to_string()),
            }),
        }
    }
}

// ── lcm_expand_query ────────────────────────────────────────────────────

pub struct LcmExpandQueryToolWrapper {
    inner: lcm_expand_query::LcmExpandQueryTool,
    session_id: Arc<Mutex<Option<String>>>,
}

impl LcmExpandQueryToolWrapper {
    pub fn new(
        engine: Arc<LcmEngine>,
        llm_call: SummarizeFn,
        session_id: Arc<Mutex<Option<String>>>,
    ) -> Self {
        Self {
            inner: lcm_expand_query::LcmExpandQueryTool::new(engine, llm_call),
            session_id,
        }
    }
}

#[async_trait]
impl Tool for LcmExpandQueryToolWrapper {
    fn name(&self) -> &str {
        lcm_expand_query::LcmExpandQueryTool::name()
    }

    fn description(&self) -> &str {
        lcm_expand_query::LcmExpandQueryTool::description()
    }

    fn parameters_schema(&self) -> serde_json::Value {
        lcm_expand_query::LcmExpandQueryTool::parameters_schema()
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let sid = self.session_id.lock().await;
        match self.inner.execute(args, sid.as_deref()).await {
            Ok(output) => Ok(ToolResult {
                success: true,
                output,
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(e.to_string()),
            }),
        }
    }
}
