use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::path::PathBuf;

use anyhow::{Context, Result};
use reqwest::blocking::{Client, RequestBuilder};
use serde::de::DeserializeOwned;
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

    pub fn get(&self, provider: &str, pkg: &str, hash: &str) -> Option<&str> {
        self.data
            .get(&format!("{}:{}:{}", provider, pkg, hash))
            .map(|s| s.as_str())
    }

    pub fn insert(&mut self, provider: &str, pkg: &str, hash: &str, result: String) {
        self.data
            .insert(format!("{}:{}:{}", provider, pkg, hash), result);
    }

    pub fn save(&self) {
        if let Ok(json) = serde_json::to_string(&self.data) {
            let _ = fs::write(&self.path, json);
        }
    }
}

/// Cache key: the PKGBUILD text is exactly what the LLM sees.
// ponytail: DefaultHasher may change between Rust releases, costing one cache miss
pub fn content_hash(path: &Path) -> Option<String> {
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

fn post_json<T: DeserializeOwned>(
    req: RequestBuilder,
    body: &impl Serialize,
    api: &str,
) -> Result<T> {
    let response = req
        .json(body)
        .send()
        .with_context(|| format!("failed to contact {} API", api))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        anyhow::bail!("{} API error {}: {}", api, status, body);
    }

    response
        .json()
        .with_context(|| format!("failed to parse {} API response", api))
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

    let req = Client::new().post(GROQ_API_URL).bearer_auth(api_key);
    let parsed: ChatResponse = post_json(req, &request, "Groq")?;

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

    let req = Client::new()
        .post(GEMINI_API_URL)
        .query(&[("key", api_key)]);
    let parsed: GeminiResponse = post_json(req, &request, "Gemini")?;

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

    #[test]
    fn cache_round_trip() {
        let dir = tmp();
        let mut cache = LlmCache::load(dir.path());
        assert!(cache.get("gemini", "pkg", "h1").is_none());
        cache.insert("gemini", "pkg", "h1", "SAFE".to_string());
        cache.insert("gemini", "pkg", "h1", "DANGER\n- curl | bash".to_string());
        cache.save();

        let cache = LlmCache::load(dir.path());
        assert_eq!(
            cache.get("gemini", "pkg", "h1"),
            Some("DANGER\n- curl | bash")
        );
        // a changed PKGBUILD or another provider must not reuse the verdict
        assert!(cache.get("gemini", "pkg", "h2").is_none());
        assert!(cache.get("groq", "pkg", "h1").is_none());

        // a corrupt file falls back to an empty cache
        fs::write(dir.path().join("llm-review.json"), "not json").unwrap();
        assert!(LlmCache::load(dir.path())
            .get("gemini", "pkg", "h1")
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
}
