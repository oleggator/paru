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
For each package output its name as a header (=== name ===), then a one-line verdict \
(SAFE / CAUTION / DANGER), then list specific findings if any. \
If nothing is suspicious, say so briefly.";

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

pub fn parse_response(response: &str) -> HashMap<String, String> {
    let mut results: HashMap<String, String> = HashMap::new();
    let mut current_pkg: Option<String> = None;
    let mut current_lines: Vec<&str> = Vec::new();

    for line in response.lines() {
        if let Some(name) = line
            .strip_prefix("=== ")
            .and_then(|s| s.strip_suffix(" ==="))
        {
            if let Some(pkg) = current_pkg.take() {
                results.insert(pkg, current_lines.join("\n").trim().to_string());
            }
            current_pkg = Some(name.to_string());
            current_lines.clear();
        } else if current_pkg.is_some() {
            current_lines.push(line);
        }
    }
    if let Some(pkg) = current_pkg {
        results.insert(pkg, current_lines.join("\n").trim().to_string());
    }

    results
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
        max_tokens: 1024,
        temperature: 0.1,
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
            max_output_tokens: 1024,
            temperature: 0.1,
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
