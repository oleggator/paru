use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use tr::tr;

use crate::config::{Colors, Config};
use crate::exec;
use crate::util::ask;

const SYSTEM_PROMPT: &str = "\
You are a security auditor reviewing Arch Linux AUR packages. \
Each package lists every file in its git repository: the PKGBUILD, \
install scripts (run as root), patches and other local sources. \
Analyze each package for security issues such as: \
arbitrary command execution risks, suspicious source URLs or checksums, \
use of eval or dynamic code execution, download of unverified binaries, \
overly broad permissions, and any other red flags. \
Respond with JSON: {\"packages\": [{\"name\": \"<name from the === name === header>\", \
\"verdict\": \"SAFE\" | \"CAUTION\" | \"DANGER\", \"findings\": [\"<short finding>\"]}]}. \
Use an empty findings list if nothing is suspicious.";

// --- Cache ---

/// Verdicts from an older SYSTEM_PROMPT are stale, so the file is dropped
/// when it changes.
#[derive(Serialize, Deserialize)]
struct CacheFile {
    prompt: String,
    /// "provider:pkg" -> (content sent to the LLM, verdict). Comparing the
    /// full content instead of a hash means a crafted PKGBUILD can't collide
    /// with a cached SAFE one, and only the latest version of each package is
    /// kept.
    entries: HashMap<String, (String, String)>,
}

struct LlmCache {
    path: PathBuf,
    data: CacheFile,
    /// entries added this run, merged into the file on save
    added: HashMap<String, (String, String)>,
}

impl LlmCache {
    fn read(path: &Path) -> CacheFile {
        fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str::<CacheFile>(&s).ok())
            .filter(|d| d.prompt == SYSTEM_PROMPT)
            .unwrap_or_else(|| CacheFile {
                prompt: SYSTEM_PROMPT.to_string(),
                entries: HashMap::new(),
            })
    }

    fn load(cache_dir: &Path) -> Self {
        let path = cache_dir.join("llm-review.json");
        Self {
            data: Self::read(&path),
            path,
            added: HashMap::new(),
        }
    }

    fn get(&self, provider: &str, pkg: &str, content: &str) -> Option<&str> {
        self.data
            .entries
            .get(&format!("{}:{}", provider, pkg))
            .filter(|(cached, _)| cached == content)
            .map(|(_, verdict)| verdict.as_str())
    }

    fn insert(&mut self, provider: &str, pkg: &str, content: &str, verdict: String) {
        let key = format!("{}:{}", provider, pkg);
        let entry = (content.to_string(), verdict);
        self.data.entries.insert(key.clone(), entry.clone());
        self.added.insert(key, entry);
    }

    /// Merges into what is on disk now, since another paru may have saved
    /// since load, then renames a temp file over it so it is never torn.
    // ponytail: no lock, two saves inside the read-rename window can still drop entries
    fn save(self) {
        if self.added.is_empty() {
            return;
        }
        let mut data = Self::read(&self.path);
        data.entries.extend(self.added);
        let Ok(json) = serde_json::to_string(&data) else {
            return;
        };
        let tmp = self
            .path
            .with_extension(format!("json.{}", std::process::id()));
        if fs::write(&tmp, json).is_err() || fs::rename(&tmp, &self.path).is_err() {
            let _ = fs::remove_file(&tmp);
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
        serde_json::from_str(response.trim()).context(tr!("LLM returned malformed JSON"))?;
    report
        .packages
        .into_iter()
        .map(|p| {
            // servers that ignore the schema can return anything
            anyhow::ensure!(
                matches!(p.verdict.as_str(), "SAFE" | "CAUTION" | "DANGER"),
                tr!(
                    "LLM returned unknown verdict {verdict} for {pkg}",
                    verdict = format!("{:?}", p.verdict),
                    pkg = p.name
                )
            );
            let mut text = p.verdict;
            for f in p.findings {
                text.push_str("\n- ");
                text.push_str(&f);
            }
            Ok((p.name, text))
        })
        .collect()
}

// --- Shared ---

/// Every file git tracks upstream, not just the PKGBUILD: `.install` scripts
/// run as root and patches change the built code. Read from `HEAD@{u}`, the
/// ref the review diff shows: PKGBUILD repos are reviewed before they are
/// merged, so their working tree is still the old version.
fn read_pkg(fetch: &aur_fetch::Fetch, pkg: &str) -> Result<String> {
    let dir = fetch.clone_dir.join(pkg);
    let git = |args: &[&str]| -> Result<Vec<u8>> {
        let mut cmd = Command::new(&fetch.git);
        cmd.args(&fetch.git_flags).current_dir(&dir).args(args);
        Ok(exec::command_output(&mut cmd)?.stdout)
    };

    let mut content = String::new();
    let tree = git(&["ls-tree", "-r", "-z", "HEAD@{u}"])?;
    for entry in tree.split(|&b| b == 0).filter(|e| !e.is_empty()) {
        // "<mode> <type> <object>\t<path>"
        let entry = String::from_utf8_lossy(entry);
        let Some((meta, file)) = entry.split_once('\t') else {
            continue;
        };
        let meta: Vec<&str> = meta.split(' ').collect();
        let [mode, _, object] = meta[..] else {
            continue;
        };
        // generated from the PKGBUILD and never run
        if file == ".SRCINFO" {
            continue;
        }
        let text = match mode {
            "160000" => format!("<submodule at {}>", object),
            // a link's blob is its target: nothing outside the repo is read
            "120000" => format!(
                "<symlink to {}>",
                String::from_utf8_lossy(&git(&["cat-file", "blob", object])?)
            ),
            _ => {
                let bytes = git(&["cat-file", "blob", object])?;
                if bytes.contains(&0) {
                    format!("<binary file, {} bytes>", bytes.len())
                } else {
                    String::from_utf8_lossy(&bytes).into_owned()
                }
            }
        };
        content.push_str(&format!("--- {} ---\n```\n{}\n```\n", file, text));
    }
    Ok(content)
}

fn build_prompt(pkgs: &[(&str, &str)]) -> String {
    let mut prompt = String::new();
    for &(name, content) in pkgs {
        prompt.push_str(&format!("=== {} ===\n{}\n", name, content));
    }
    prompt
}

// --- OpenAI-compatible chat completions ---

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    // unset by default: OpenAI reasoning models reject anything but 1
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
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
        temperature: config.llm_temperature,
        response_format: serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "report",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": { "packages": { "type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "verdict": { "type": "string", "enum": ["SAFE", "CAUTION", "DANGER"] },
                            "findings": { "type": "array", "items": { "type": "string" } },
                        },
                        "required": ["name", "verdict", "findings"],
                        "additionalProperties": false,
                    }}},
                    "required": ["packages"],
                    "additionalProperties": false,
                },
            },
        }),
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
    let response = req.send().context(tr!("failed to contact LLM API"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        anyhow::bail!(tr!(
            "LLM API error {status}: {body}",
            status = status,
            body = body
        ));
    }

    let parsed: ChatResponse = response
        .json()
        .context(tr!("failed to parse LLM API response"))?;
    parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .context(tr!("empty response from LLM API"))
}

pub fn report(config: &Config, fetch: &aur_fetch::Fetch, pkgs: &[&str]) -> Vec<u8> {
    let Some(model) = &config.llm_model else {
        return Vec::new();
    };
    if !ask(config, &tr!("Verify PKGBUILDs with LLM?"), false) {
        return Vec::new();
    }

    let c = config.color;
    let mut out: Vec<u8> = Vec::new();

    let _ = writeln!(
        out,
        "{} {}",
        c.action.paint("::"),
        c.bold
            .paint(tr!("LLM security check ({model}):", model = model))
    );

    // read once: what is cached is exactly what the LLM sees
    let mut pkg_contents: Vec<(&str, String)> = Vec::new();
    for &pkg in pkgs {
        match read_pkg(fetch, pkg) {
            Ok(content) => pkg_contents.push((pkg, content)),
            Err(e) => {
                let _ = writeln!(
                    out,
                    "  {} {}",
                    c.warning.paint(tr!("warning:")),
                    tr!(
                        "{pkg}: not checked: {err}",
                        pkg = pkg,
                        err = printable(&format!("{:#}", e))
                    )
                );
            }
        }
    }

    let mut cache = LlmCache::load(&config.cache_dir);
    llm_run_check(config, model, &pkg_contents, &mut cache, c, &mut out);

    cache.save();
    out
}

/// Verdicts are cached per API URL and model, so switching either re-reviews.
fn llm_run_check(
    config: &Config,
    model: &str,
    pkg_contents: &[(&str, String)],
    cache: &mut LlmCache,
    c: Colors,
    out: &mut impl Write,
) {
    let provider = format!("{} {}", config.llm_api_url, model);
    let mut results: HashMap<String, (String, bool)> = HashMap::new();
    let mut uncached: Vec<(&str, &str)> = Vec::new();

    for (pkg, content) in pkg_contents {
        match cache.get(&provider, pkg, content) {
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
                        cache.insert(&provider, pkg, content, result.clone());
                        results.insert(pkg.to_string(), (result.clone(), false));
                    }
                }
            }
            // written into the report: stderr is hidden behind the pager
            Err(e) => {
                let _ = writeln!(
                    out,
                    "  {} {}",
                    c.warning.paint(tr!("warning:")),
                    tr!("check failed: {err}", err = printable(&format!("{:#}", e)))
                );
            }
        }
    }

    for (pkg, _) in pkg_contents {
        let Some((result, cached)) = results.get(*pkg) else {
            let _ = writeln!(
                out,
                "  {} {}: {}",
                c.action.paint("->"),
                c.bold.paint(*pkg),
                tr!("no verdict")
            );
            continue;
        };
        let suffix = if *cached {
            format!(" {}", tr!("(cached)"))
        } else {
            String::new()
        };
        let _ = writeln!(
            out,
            "  {} {}{}:",
            c.action.paint("->"),
            c.bold.paint(*pkg),
            suffix
        );
        for line in result.lines() {
            let _ = writeln!(out, "    {}", colorize_verdict(c, &printable(line)));
        }
    }
    let _ = writeln!(out);
}

/// LLM output, API errors and file names are untrusted: drop escape codes and
/// other control characters so they can't redraw the terminal.
fn printable(s: &str) -> String {
    s.chars()
        .filter(|&c| c == '\n' || !c.is_control())
        .collect()
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
        assert!(
            parse_response(r#"{"packages": [{"name": "pkg", "verdict": "Low risk"}]}"#).is_err()
        );
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

        // verdicts from another prompt are dropped
        let file = dir.path().join("llm-review.json");
        let old = fs::read_to_string(&file).unwrap();
        fs::write(&file, old.replace("security auditor", "auditor")).unwrap();
        assert!(LlmCache::load(dir.path())
            .get("model-a", "pkg", "v2")
            .is_none());

        // a corrupt file falls back to an empty cache
        fs::write(&file, "not json").unwrap();
        assert!(LlmCache::load(dir.path())
            .get("model-a", "pkg", "v2")
            .is_none());
    }

    #[test]
    fn cache_save_merges_concurrent_runs() {
        let dir = tmp();
        // nothing added: no file written
        LlmCache::load(dir.path()).save();
        assert!(!dir.path().join("llm-review.json").exists());

        let mut a = LlmCache::load(dir.path());
        let mut b = LlmCache::load(dir.path());
        a.insert("m", "pkg-a", "a", "SAFE".to_string());
        b.insert("m", "pkg-b", "b", "DANGER".to_string());
        a.save();
        b.save();

        let cache = LlmCache::load(dir.path());
        assert_eq!(cache.get("m", "pkg-a", "a"), Some("SAFE"));
        assert_eq!(cache.get("m", "pkg-b", "b"), Some("DANGER"));
    }

    #[test]
    fn build_prompt_wraps_each_package_in_order() {
        let prompt = build_prompt(&[("alpha", "a"), ("beta", "b")]);
        assert_eq!(prompt, "=== alpha ===\na\n=== beta ===\nb\n");
    }

    // needs git on PATH, like the rest of paru
    #[test]
    fn read_pkg_reads_upstream_tracked_files() {
        let dir = tmp();
        let git = |cwd: &Path, args: &[&str]| {
            let out = Command::new("git")
                .current_dir(cwd)
                .args([
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t",
                    "-c",
                    "commit.gpgsign=false",
                ])
                .args(args)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
        };
        let up = dir.path().join("upstream");
        fs::create_dir(&up).unwrap();
        fs::write(up.join("PKGBUILD"), "pkgname=pkg").unwrap();
        fs::write(up.join("pkg.install"), "post_install() { :; }").unwrap();
        fs::write(up.join(".SRCINFO"), "pkgbase = pkg").unwrap();
        fs::write(up.join("icon.png"), b"\x89PNG\0\0").unwrap();
        std::os::unix::fs::symlink("/etc/passwd", up.join("link")).unwrap();
        git(&up, &["init", "-q"]);
        git(&up, &["add", "."]);
        git(&up, &["commit", "-qm", "init"]);

        let clone = dir.path().join("clone");
        fs::create_dir(&clone).unwrap();
        git(&clone, &["clone", "-q", up.to_str().unwrap(), "pkg"]);
        // the working tree is not what gets reviewed
        fs::write(clone.join("pkg/PKGBUILD"), "local edit").unwrap();
        fs::write(clone.join("pkg/untracked"), "x").unwrap();

        let fetch = aur_fetch::Fetch::with_combined_cache_dir(&clone);
        assert_eq!(
            read_pkg(&fetch, "pkg").unwrap(),
            "--- PKGBUILD ---\n```\npkgname=pkg\n```\n\
             --- icon.png ---\n```\n<binary file, 6 bytes>\n```\n\
             --- link ---\n```\n<symlink to /etc/passwd>\n```\n\
             --- pkg.install ---\n```\npost_install() { :; }\n```\n"
        );
        assert!(read_pkg(&fetch, "ghost").is_err());
    }

    #[test]
    fn printable_drops_control_chars() {
        assert_eq!(printable("a\x1b[1A\x1b[2Kb\r\nc\x07"), "a[1A[2Kb\nc");
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
