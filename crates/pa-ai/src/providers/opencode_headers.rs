//! `OpenCode` client-identity headers (TS `opencode-headers.ts`).
//!
//! The `OpenCode` Go endpoint rejects every request that lacks a stable
//! per-conversation id (`x-opencode-session`) and a client-identifying
//! `User-Agent` ([the Go docs](https://opencode.ai/docs/go/#where-can-i-use-it)).
//! For providers `opencode` and `opencode-go` this merge puts those two
//! headers under the request's own headers, which stay authoritative.

/// Merge the `OpenCode` identity headers over `headers`, mirroring TS
/// `withOpenCodeHeaders`: providers other than `opencode`/`opencode-go`
/// return `headers` unchanged; otherwise `User-Agent: prime-agent` and
/// `x-opencode-session` (the conversation's `session_id`) are defaults the
/// caller's own headers override. Names normalize to lowercase except
/// `User-Agent`, whose casing survives so an SDK that replaces its default
/// user agent sees it. A caller header of the same name (any casing)
/// replaces the default's value in place.
pub(crate) fn with_opencode_headers(
    provider: &str,
    session_id: Option<&str>,
    headers: Vec<(String, String)>,
) -> Vec<(String, String)> {
    if provider != "opencode" && provider != "opencode-go" {
        return headers;
    }
    let mut merged = vec![("User-Agent".to_string(), "prime-agent".to_string())];
    if let Some(session_id) = session_id {
        merged.push(("x-opencode-session".to_string(), session_id.to_string()));
    }
    for (name, value) in headers {
        let name = if name.eq_ignore_ascii_case("user-agent") {
            "User-Agent".to_string()
        } else {
            name.to_lowercase()
        };
        match merged
            .iter_mut()
            .find(|(existing, _)| existing.eq_ignore_ascii_case(&name))
        {
            Some(slot) => slot.1 = value,
            None => merged.push((name, value)),
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
            .collect()
    }

    /// TS `withOpenCodeHeaders` pins: the identity headers ride every opencode
    /// request under the caller's own headers, and no other provider's request
    /// gains them.
    #[test]
    fn opencode_requests_carry_the_identity_headers() {
        let headers = with_opencode_headers(
            "opencode-go",
            Some("session-1"),
            pairs(&[("Accept", "application/json"), ("X-Trace", "abc")]),
        );
        assert_eq!(
            headers,
            pairs(&[
                ("User-Agent", "prime-agent"),
                ("x-opencode-session", "session-1"),
                ("accept", "application/json"),
                ("x-trace", "abc"),
            ])
        );
    }

    /// A caller-supplied header of the same name (any casing) wins over the
    /// identity default, in the default's position — the caller knows its own
    /// user agent and session better than the merge.
    #[test]
    fn caller_headers_override_the_identity_defaults() {
        let headers = with_opencode_headers(
            "opencode",
            Some("session-1"),
            pairs(&[
                ("user-agent", "custom/1.0"),
                ("X-Opencode-Session", "session-2"),
            ]),
        );
        assert_eq!(
            headers,
            pairs(&[
                ("User-Agent", "custom/1.0"),
                ("x-opencode-session", "session-2")
            ])
        );
    }

    /// Without a conversation id the default user agent still ships; the
    /// session header is absent rather than empty.
    #[test]
    fn a_missing_session_id_ships_no_session_header() {
        let headers = with_opencode_headers("opencode-go", None, Vec::new());
        assert_eq!(headers, pairs(&[("User-Agent", "prime-agent")]));
    }

    /// Every other provider's headers pass through untouched.
    #[test]
    fn other_providers_are_unchanged() {
        let headers = pairs(&[("Accept", "application/json")]);
        assert_eq!(
            with_opencode_headers("anthropic", Some("session-1"), headers.clone()),
            headers
        );
    }
}
