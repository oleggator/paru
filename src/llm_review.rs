use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

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

struct LlmCache {
    path: PathBuf,
    data: HashMap<String, String>,
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

    fn get(&self, model: &str, pkg: &str, hash: &str) -> Option<&str> {
        self.data
            .get(&format!("{}:{}:{}", model, pkg, hash))
            .map(|s| s.as_str())
    }

    fn insert(&mut self, model: &str, pkg: &str, hash: &str, result: String) {
        self.data
            .insert(format!("{}:{}:{}", model, pkg, hash), result);
    }

    fn save(&self) {
        if let Ok(json) = serde_json::to_string(&self.data) {
            let _ = fs::write(&self.path, json);
        }
    }
}

/// Cache key: the PKGBUILD text is exactly what the LLM sees.
// ponytail: DefaultHasher may change between Rust releases, costing one cache miss
fn content_hash(path: &Path) -> Option<String> {
    let mut hasher = DefaultHasher::new();
    fs::read(path).ok()?.hash(&mut hasher);
    Some(format!("{:016x}", hasher.finish()))
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

fn build_prompt(pkgs: &[(&str, &Path)]) -> Result<String> {
    let mut prompt = String::new();
    for &(name, path) in pkgs {
        let content = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        prompt.push_str(&format!("=== {} ===\n```\n{}\n```\n\n", name, content));
    }
    Ok(prompt)
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

fn check_pkgbuilds(config: &Config, model: &str, pkgs: &[(&str, &Path)]) -> Result<String> {
    let request = ChatRequest {
        model: model.to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: build_prompt(pkgs)?,
            },
        ],
        response_format: serde_json::json!({ "type": "json_object" }),
    };

    let url = format!(
        "{}/chat/completions",
        config.llm_api_url.trim_end_matches('/')
    );
    let mut req = Client::new().post(url).json(&request);
    if let Some(key) = &config.llm_api_key {
        req = req.bearer_auth(key);
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

    let pkg_paths: Vec<(&str, PathBuf, String)> = pkgs
        .iter()
        .filter_map(|&pkg| {
            let path = fetch.clone_dir.join(pkg).join("PKGBUILD");
            let hash = content_hash(&path)?;
            Some((pkg, path, hash))
        })
        .collect();

    if pkg_paths.is_empty() {
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
    llm_run_check(config, model, &pkg_paths, &mut cache, c, &mut out);

    cache.save();
    out
}

/// Verdicts are cached per model, so switching models re-reviews.
fn llm_run_check(
    config: &Config,
    model: &str,
    pkg_paths: &[(&str, PathBuf, String)],
    cache: &mut LlmCache,
    c: Colors,
    out: &mut impl Write,
) {
    let mut results: HashMap<String, (String, bool)> = HashMap::new();
    let mut uncached: Vec<(&str, &Path)> = Vec::new();

    for (pkg, path, hash) in pkg_paths {
        match cache.get(model, pkg, hash) {
            Some(hit) => {
                results.insert(pkg.to_string(), (hit.to_string(), true));
            }
            None => uncached.push((pkg, path)),
        }
    }

    if !uncached.is_empty() {
        match tokio::task::block_in_place(|| check_pkgbuilds(config, model, &uncached))
            .and_then(|r| parse_response(&r))
        {
            Ok(parsed) => {
                for (pkg, _, hash) in pkg_paths {
                    // cached ones were not sent: a PKGBUILD could inject verdicts for them
                    if results.contains_key(*pkg) {
                        continue;
                    }
                    if let Some(result) = parsed.get(*pkg) {
                        cache.insert(model, pkg, hash, result.clone());
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

    for (pkg, _, _) in pkg_paths {
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
        assert!(cache.get("model-a", "pkg", "h1").is_none());
        cache.insert("model-a", "pkg", "h1", "SAFE".to_string());
        cache.insert("model-a", "pkg", "h1", "DANGER\n- curl | bash".to_string());
        cache.save();

        let cache = LlmCache::load(dir.path());
        assert_eq!(
            cache.get("model-a", "pkg", "h1"),
            Some("DANGER\n- curl | bash")
        );
        // a changed PKGBUILD or another model must not reuse the verdict
        assert!(cache.get("model-a", "pkg", "h2").is_none());
        assert!(cache.get("model-b", "pkg", "h1").is_none());

        // a corrupt file falls back to an empty cache
        fs::write(dir.path().join("llm-review.json"), "not json").unwrap();
        assert!(LlmCache::load(dir.path())
            .get("model-a", "pkg", "h1")
            .is_none());
    }

    #[test]
    fn build_prompt_wraps_each_pkgbuild_in_order() {
        let dir = tmp();
        let (a, b) = (dir.path().join("a"), dir.path().join("b"));
        fs::write(&a, "pkgname=alpha").unwrap();
        fs::write(&b, "pkgname=beta").unwrap();

        let prompt = build_prompt(&[("alpha", a.as_path()), ("beta", b.as_path())]).unwrap();
        assert_eq!(
            prompt,
            "=== alpha ===\n```\npkgname=alpha\n```\n\n=== beta ===\n```\npkgname=beta\n```\n\n"
        );
        assert!(build_prompt(&[("ghost", Path::new("/nonexistent/PKGBUILD"))]).is_err());
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
