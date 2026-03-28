/// Depth-aware summarization prompt templates.
///
/// Ported from upstream lossless-claw `summarize.ts`.

/// System prompt for the summarizer model.
pub const SUMMARIZER_SYSTEM_PROMPT: &str =
    "You are a context-compaction summarization engine. Follow user instructions exactly and return plain text summary content only.";

/// Build the leaf (depth 0) summarization prompt.
pub fn build_leaf_prompt(
    text: &str,
    target_tokens: usize,
    aggressive: bool,
    previous_context: Option<&str>,
) -> String {
    let policy = if aggressive {
        "Aggressive summary policy:\n\
         - Keep only durable facts and current task state.\n\
         - Remove examples, repetition, and low-value narrative details.\n\
         - Preserve explicit TODOs, blockers, decisions, and constraints."
    } else {
        "Normal summary policy:\n\
         - Preserve key decisions, rationale, constraints, and active tasks.\n\
         - Keep essential technical details needed to continue work safely.\n\
         - Remove obvious repetition and conversational filler."
    };

    let prev = previous_context.unwrap_or("(none)");

    format!(
        "You summarize a SEGMENT of a conversation for future model turns.\n\
         Treat this as incremental memory compaction input, not a full-conversation summary.\n\
         {policy}\n\n\
         Output requirements:\n\
         - Plain text only.\n\
         - No preamble, headings, or markdown formatting.\n\
         - Keep it concise while preserving required details.\n\
         - Track file operations (created, modified, deleted, renamed) with file paths and current status.\n\
         - If no file operations appear, include exactly: \"Files: none\".\n\
         - End with exactly: \"Expand for details about: <comma-separated list of what was dropped or compressed>\".\n\
         - Target length: about {target_tokens} tokens or less.\n\n\
         <previous_context>\n{prev}\n</previous_context>\n\n\
         <conversation_segment>\n{text}\n</conversation_segment>"
    )
}

/// Build the depth-1 condensation prompt.
pub fn build_d1_prompt(
    text: &str,
    target_tokens: usize,
    previous_context: Option<&str>,
) -> String {
    let prev_block = if let Some(prev) = previous_context {
        format!(
            "It already has this preceding summary as context. Do not repeat information\n\
             that appears there unchanged. Focus on what is new, changed, or resolved:\n\n\
             <previous_context>\n{prev}\n</previous_context>\n"
        )
    } else {
        "Focus on what matters for continuation:\n".to_string()
    };

    format!(
        "You are compacting leaf-level conversation summaries into a single condensed memory node.\n\
         You are preparing context for a fresh model instance that will continue this conversation.\n\n\
         {prev_block}\n\
         Preserve:\n\
         - Decisions made and their rationale when rationale matters going forward.\n\
         - Earlier decisions that were superseded, and what replaced them.\n\
         - Completed tasks/topics with outcomes.\n\
         - In-progress items with current state and what remains.\n\
         - Blockers, open questions, and unresolved tensions.\n\
         - Specific references (names, paths, URLs, identifiers) needed for continuation.\n\n\
         Drop low-value detail:\n\
         - Context that has not changed from previous_context.\n\
         - Intermediate dead ends where the conclusion is already known.\n\
         - Transient states that are already resolved.\n\
         - Tool-internal mechanics and process scaffolding.\n\n\
         Use plain text. No mandatory structure.\n\
         Include a timeline with timestamps (hour or half-hour) for significant events.\n\
         Present information chronologically and mark superseded decisions.\n\
         End with exactly: \"Expand for details about: <comma-separated list of what was dropped or compressed>\".\n\
         Target length: about {target_tokens} tokens.\n\n\
         <conversation_to_condense>\n{text}\n</conversation_to_condense>"
    )
}

/// Build the depth-2 condensation prompt.
pub fn build_d2_prompt(text: &str, target_tokens: usize) -> String {
    format!(
        "You are condensing multiple session-level summaries into a higher-level memory node.\n\
         A future model should understand trajectory, not per-session minutiae.\n\n\
         Preserve:\n\
         - Decisions still in effect and their rationale.\n\
         - Decisions that evolved: what changed and why.\n\
         - Completed work with outcomes.\n\
         - Active constraints, limitations, and known issues.\n\
         - Current state of in-progress work.\n\n\
         Drop:\n\
         - Session-local operational detail and process mechanics.\n\
         - Identifiers that are no longer relevant.\n\
         - Intermediate states superseded by later outcomes.\n\n\
         Use plain text. Brief headers are fine if useful.\n\
         Include a timeline with dates and approximate time of day for key milestones.\n\
         End with exactly: \"Expand for details about: <comma-separated list of what was dropped or compressed>\".\n\
         Target length: about {target_tokens} tokens.\n\n\
         <conversation_to_condense>\n{text}\n</conversation_to_condense>"
    )
}

/// Build the depth 3+ condensation prompt.
pub fn build_d3_plus_prompt(text: &str, target_tokens: usize) -> String {
    format!(
        "You are creating a high-level memory node from multiple phase-level summaries.\n\
         This may persist for the rest of the conversation. Keep only durable context.\n\n\
         Preserve:\n\
         - Key decisions and rationale.\n\
         - What was accomplished and current state.\n\
         - Active constraints and hard limitations.\n\
         - Important relationships between people, systems, or concepts.\n\
         - Durable lessons learned.\n\n\
         Drop:\n\
         - Operational and process detail.\n\
         - Method details unless the method itself was the decision.\n\
         - Specific references unless essential for continuation.\n\n\
         Use plain text. Be concise.\n\
         Include a brief timeline with dates (or date ranges) for major milestones.\n\
         End with exactly: \"Expand for details about: <comma-separated list of what was dropped or compressed>\".\n\
         Target length: about {target_tokens} tokens.\n\n\
         <conversation_to_condense>\n{text}\n</conversation_to_condense>"
    )
}

/// Build the appropriate prompt for a given depth.
pub fn build_prompt_for_depth(
    depth: i64,
    text: &str,
    target_tokens: usize,
    aggressive: bool,
    previous_context: Option<&str>,
) -> String {
    match depth {
        0 => build_leaf_prompt(text, target_tokens, aggressive, previous_context),
        1 => build_d1_prompt(text, target_tokens, previous_context),
        2 => build_d2_prompt(text, target_tokens),
        _ => build_d3_plus_prompt(text, target_tokens),
    }
}

/// Compute target tokens for a summarization call.
pub fn resolve_target_tokens(
    input_tokens: usize,
    aggressive: bool,
    is_condensed: bool,
    condensed_target_tokens: usize,
) -> usize {
    if is_condensed {
        condensed_target_tokens.max(512)
    } else if aggressive {
        96.max(640.min(input_tokens * 20 / 100))
    } else {
        192.max(1200.min(input_tokens * 35 / 100))
    }
}

/// Build a deterministic fallback summary by truncating.
pub fn deterministic_fallback(source_text: &str, target_tokens: usize) -> String {
    let max_chars = 256_usize.max(target_tokens * 4).min(source_text.len());
    let truncated = &source_text[..max_chars];
    if max_chars < source_text.len() {
        format!("{truncated}\n[LCM fallback summary; truncated for context management]")
    } else {
        truncated.to_string()
    }
}

/// Build the LCM Recall system prompt addition for when summaries are present.
pub fn build_system_prompt_addition(max_depth: i64, condensed_count: usize) -> String {
    let mut s = String::from(
        "## LCM Recall\n\n\
         Summaries above are compressed context — maps to details, not the details themselves.\n\n\
         **Recall priority:** Use LCM tools first for compacted conversation history.\n\n\
         **Tool escalation:**\n\
         1. `lcm_grep` — search by regex or full-text across messages and summaries\n\
         2. `lcm_describe` — inspect a specific summary (cheap, no sub-agent)\n\
         3. `lcm_expand_query` — deep recall with expanded DAG context\n\n\
         **Summaries include \"Expand for details about:\" footers** listing compressed specifics.\n"
    );

    if max_depth >= 2 || condensed_count >= 2 {
        s.push_str(
            "\n**Warning: Deeply compacted context — expand before asserting specifics.**\n\n\
             Do not guess exact commands, SHAs, file paths, timestamps, config values, or \
             causal claims from condensed summaries. Expand first or state uncertainty.\n"
        );
    } else {
        s.push_str(
            "\n**For precision/evidence questions** (exact commands, SHAs, paths, timestamps, \
             config values, root-cause chains): expand before answering.\n"
        );
    }

    s
}
