use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use tr::tr;

use crate::config::{Colors, Config};
use crate::util::ask;

const SYSTEM_PROMPT: &str = "\
You are a security auditor reviewing Arch Linux PKGBUILD files. \
Analyze each PKGBUILD for security issues such as: \
arbitrary command execution risks, suspicious source URLs or checksums, \
use of eval or dynamic code execution, download of unverified binaries, \
overly broad permissions, and any other red flags. \
Respond with JSON: {\"packages\": [{\"name\": \"<name from the === name === header>\", \
\"verdict\": \"SAFE\" | \"CAUTION\" | \"DANGER\", \"findings\": [\"<short finding>\"]}]}. \
Use an empty findings list if nothing is suspicious.";

// --- Cache ---

/// "model:pkg" -> (content sent to the LLM, verdict). Comparing the full
/// content instead of a hash means a crafted PKGBUILD can't collide with a
/// cached SAFE one, and only the latest version of each package is kept.
struct LlmCache {
    path: PathBuf,
    data: HashMap<String, (String, String)>,
}

impl LlmCache {
    fn load(cache_dir: &Path) -> Self {
        let path = cache_dir.join("llm-review.json");
        let data = fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Self { path, data }
    }

    fn get(&self, model: &str, pkg: &str, content: &str) -> Option<&str> {
        self.data
            .get(&format!("{}:{}", model, pkg))
            .filter(|(cached, _)| cached == content)
            .map(|(_, verdict)| verdict.as_str())
    }

    fn insert(&mut self, model: &str, pkg: &str, content: &str, verdict: String) {
        self.data
            .insert(format!("{}:{}", model, pkg), (content.to_string(), verdict));
    }

    fn save(&self) {
        if let Ok(json) = serde_json::to_string(&self.data) {
            let _ = fs::write(&self.path, json);
        }
    }
}

#[derive(Deserialize)]
struct Report {
    packages: Vec<PkgReport>,
}

#[derive(Deserialize)]
struct PkgReport {
    name: String,
    verdict: String,
    #[serde(default)]
    findings: Vec<String>,
}

/// Returns "VERDICT\n- finding\n- finding" per package name.
fn parse_response(response: &str) -> Result<HashMap<String, String>> {
    let report: Report =
        serde_json::from_str(response.trim()).context("LLM returned malformed JSON")?;
    Ok(report
        .packages
        .into_iter()
        .map(|p| {
            let mut text = p.verdict;
            for f in p.findings {
                text.push_str("\n- ");
                text.push_str(&f);
            }
            (p.name, text)
        })
        .collect())
}

// --- Shared ---

fn build_prompt(pkgs: &[(&str, &str)]) -> String {
    let mut prompt = String::new();
    for &(name, content) in pkgs {
        prompt.push_str(&format!("=== {} ===\n```\n{}\n```\n\n", name, content));
    }
    prompt
}

// --- OpenAI-compatible chat completions ---

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    response_format: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

fn check_pkgbuilds(config: &Config, model: &str, pkgs: &[(&str, &str)]) -> Result<String> {
    let request = ChatRequest {
        model: model.to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: build_prompt(pkgs),
            },
        ],
        response_format: serde_json::json!({ "type": "json_object" }),
    };

    let url = format!(
        "{}/chat/completions",
        config.llm_api_url.trim_end_matches('/')
    );
    // the blocking default of 30s is too short for reasoning or local models
    let client = Client::builder().timeout(None).build()?;
    let mut req = client.post(url).json(&request);
    if let Some(key) = &config.llm_api_key {
        req = req.bearer_auth(&key.0);
    }
    let response = req.send().context("failed to contact LLM API")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        anyhow::bail!("LLM API error {}: {}", status, body);
    }

    let parsed: ChatResponse = response
        .json()
        .context("failed to parse LLM API response")?;
    parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .context("empty response from LLM API")
}

pub fn report(config: &Config, fetch: &aur_fetch::Fetch, pkgs: &[&str]) -> Vec<u8> {
    let Some(model) = &config.llm_model else {
        return Vec::new();
    };
    if !ask(config, &tr!("Verify PKGBUILDs with LLM?"), false) {
        return Vec::new();
    }

    let c = config.color;

    // read once: what is cached is exactly what the LLM sees
    let pkg_contents: Vec<(&str, String)> = pkgs
        .iter()
        .filter_map(|&pkg| {
            let bytes = fs::read(fetch.clone_dir.join(pkg).join("PKGBUILD")).ok()?;
            Some((pkg, String::from_utf8_lossy(&bytes).into_owned()))
        })
        .collect();

    if pkg_contents.is_empty() {
        return Vec::new();
    }

    let mut cache = LlmCache::load(&config.cache_dir);
    let mut out: Vec<u8> = Vec::new();

    let _ = writeln!(
        out,
        "{} {}",
        c.action.paint("::"),
        c.bold.paint(format!("LLM security check ({}):", model))
    );
    llm_run_check(config, model, &pkg_contents, &mut cache, c, &mut out);

    cache.save();
    out
}

/// Verdicts are cached per model, so switching models re-reviews.
fn llm_run_check(
    config: &Config,
    model: &str,
    pkg_contents: &[(&str, String)],
    cache: &mut LlmCache,
    c: Colors,
    out: &mut impl Write,
) {
    let mut results: HashMap<String, (String, bool)> = HashMap::new();
    let mut uncached: Vec<(&str, &str)> = Vec::new();

    for (pkg, content) in pkg_contents {
        match cache.get(model, pkg, content) {
            Some(hit) => {
                results.insert(pkg.to_string(), (hit.to_string(), true));
            }
            None => uncached.push((pkg, content)),
        }
    }

    if !uncached.is_empty() {
        match tokio::task::block_in_place(|| check_pkgbuilds(config, model, &uncached))
            .and_then(|r| parse_response(&r))
        {
            Ok(parsed) => {
                for (pkg, content) in pkg_contents {
                    // cached ones were not sent: a PKGBUILD could inject verdicts for them
                    if results.contains_key(*pkg) {
                        continue;
                    }
                    if let Some(result) = parsed.get(*pkg) {
                        cache.insert(model, pkg, content, result.clone());
                        results.insert(pkg.to_string(), (result.clone(), false));
                    }
                }
            }
            // written into the report: stderr is hidden behind the pager
            Err(e) => {
                let _ = writeln!(
                    out,
                    "  {} check failed: {:#}",
                    c.warning.paint("warning:"),
                    e
                );
            }
        }
    }

    for (pkg, _) in pkg_contents {
        let Some((result, cached)) = results.get(*pkg) else {
            let _ = writeln!(
                out,
                "  {} {}: no verdict",
                c.action.paint("->"),
                c.bold.paint(*pkg)
            );
            continue;
        };
        let suffix = if *cached { " (cached)" } else { "" };
        let _ = writeln!(
            out,
            "  {} {}{}:",
            c.action.paint("->"),
            c.bold.paint(*pkg),
            suffix
        );
        for line in result.lines() {
            let _ = writeln!(out, "    {}", colorize_verdict(c, line));
        }
    }
    let _ = writeln!(out);
}

fn colorize_verdict(c: Colors, line: &str) -> String {
    for (word, style) in &[
        ("DANGER", c.error),
        ("CAUTION", c.warning),
        ("SAFE", c.upgrade),
    ] {
        if let Some(rest) = line.strip_prefix(word) {
            return format!("{}{}", style.paint(*word), rest);
        }
    }
    line.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn tmp() -> TempDir {
        tempfile::tempdir().expect("failed to create tempdir")
    }

    // --- parse_response ---

    #[test]
    fn parse_response_formats_verdict_then_findings() {
        let input = r#"{"packages": [
            {"name": "clean-pkg", "verdict": "SAFE", "findings": []},
            {"name": "bad-pkg", "verdict": "DANGER", "findings": ["Runs curl | bash", "Uses eval"]}
        ]}"#;
        let result = parse_response(input).unwrap();
        assert_eq!(result.get("clean-pkg").unwrap(), "SAFE");
        assert_eq!(
            result.get("bad-pkg").unwrap(),
            "DANGER\n- Runs curl | bash\n- Uses eval"
        );
    }

    // Truncated or non-JSON output must be an error, not an empty report.
    #[test]
    fn parse_response_rejects_malformed_json() {
        assert!(parse_response(r#"{"packages": [{"name": "pkg", "verd"#).is_err());
        assert!(parse_response("=== pkg ===\nSAFE").is_err());
    }

    #[test]
    fn cache_round_trip() {
        let dir = tmp();
        let mut cache = LlmCache::load(dir.path());
        assert!(cache.get("model-a", "pkg", "v1").is_none());
        cache.insert("model-a", "pkg", "v1", "SAFE".to_string());
        cache.insert("model-a", "pkg", "v2", "DANGER\n- curl | bash".to_string());
        cache.save();

        let cache = LlmCache::load(dir.path());
        assert_eq!(
            cache.get("model-a", "pkg", "v2"),
            Some("DANGER\n- curl | bash")
        );
        // a changed PKGBUILD or another model must not reuse the verdict
        assert!(cache.get("model-a", "pkg", "v1").is_none());
        assert!(cache.get("model-b", "pkg", "v2").is_none());

        // a corrupt file falls back to an empty cache
        fs::write(dir.path().join("llm-review.json"), "not json").unwrap();
        assert!(LlmCache::load(dir.path())
            .get("model-a", "pkg", "v2")
            .is_none());
    }

    #[test]
    fn build_prompt_wraps_each_pkgbuild_in_order() {
        let prompt = build_prompt(&[("alpha", "pkgname=alpha"), ("beta", "pkgname=beta")]);
        assert_eq!(
            prompt,
            "=== alpha ===\n```\npkgname=alpha\n```\n\n=== beta ===\n```\npkgname=beta\n```\n\n"
        );
    }

    // Colors::default() paints nothing, so this only checks no line is mangled.
    #[test]
    fn colorize_verdict_keeps_text() {
        for line in [
            "DANGER: downloads curl | bash",
            "CAUTION: source uses HTTP",
            "SAFE",
            "The verdict is DANGER here",
            "danger: something bad",
            "- Uses eval on network response",
            "",
        ] {
            assert_eq!(colorize_verdict(Colors::default(), line), line);
        }
    }
}
