// Pedantic-gate dispositions as src/lib.rs (large_futures/too_many_lines/casts).
#![allow(
    clippy::large_futures,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

//! Session identity wiring: every model request the session engine issues
//! carries the conversation's session id — the identity `OpenCode` Go routes
//! and caches by (`x-opencode-session`; TS `AgentOptions.sessionId`).
use std::sync::{Arc, Mutex};

use pa_agent::scripted::ScriptedProvider;
use pa_core::session_engine::engine::{create_session, SessionEngineConfig};
use pa_core::session_engine::{PromptOptions, PromptOutcome};

#[tokio::test]
async fn stream_requests_carry_the_session_id() {
    let model: pa_agent::types::Model = serde_json::from_value(serde_json::json!({
        "id": "m", "name": "m", "api": "openai-completions", "provider": "test",
        "baseUrl": "http://localhost", "reasoning": false, "input": ["text"],
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
        "contextWindow": 1000, "maxTokens": 100
    }))
    .unwrap();
    let scripted = Arc::new(ScriptedProvider::new(model.clone()));
    scripted.push_text_turn("hello from the model");
    let seen = Arc::new(Mutex::new(Vec::new()));
    let inner = scripted.stream_fn();
    let stream_fn: pa_agent::stream::StreamFn = {
        let seen = Arc::clone(&seen);
        Arc::new(move |model, context, options| {
            seen.lock().unwrap().push(options.session_id.clone());
            inner(model, context, options)
        })
    };
    let dir = tempfile::TempDir::new().expect("temp dir");
    let agent_dir = dir.path().join("agent");
    std::fs::create_dir_all(&agent_dir).expect("agent dir");
    let engine = create_session(SessionEngineConfig {
        cwd: dir.path().to_path_buf(),
        agent_dir,
        model: Some(model),
        stream_fn: Some(stream_fn),
        ..Default::default()
    })
    .await
    .expect("create the session");
    let outcome = engine
        .prompt("hi there", PromptOptions::default())
        .await
        .expect("prompt");
    assert_eq!(outcome, PromptOutcome::Prompt);
    engine.session.agent().wait_for_idle().await;

    // The request carried the session's own id (the wire identity every
    // conversation request rides).
    let session_id = engine.session.session_id().await;
    assert_eq!(*seen.lock().unwrap(), vec![Some(session_id)]);
}
