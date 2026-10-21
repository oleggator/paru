use std::cmp::Reverse;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use futures::stream::{self, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tr::tr;

use ansiterm::Style;

use crate::config::{Colors, Config};
use crate::exec;
use crate::util::ask;

const SYSTEM_PROMPT: &str = "\
You are a security auditor reviewing an Arch Linux AUR package. \
You get every file in its git repository: the PKGBUILD, \
install scripts (run as root), patches and other local sources. \
Analyze it for security issues such as: \
arbitrary command execution risks, suspicious source URLs or checksums, \
use of eval or dynamic code execution, download of unverified binaries, \
overly broad permissions, and any other red flags. \
Respond with JSON: {\"verdict\": \"SAFE\" | \"CAUTION\" | \"DANGER\", \
\"findings\": [\"<short finding>\"]}. \
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
    entries: HashMap<String, (String, Verdict)>,
}

struct LlmCache {
    path: PathBuf,
    data: CacheFile,
    /// entries added this run, merged into the file on save
    added: HashMap<String, (String, Verdict)>,
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

    fn get(&self, provider: &str, pkg: &str, content: &str) -> Option<&Verdict> {
        self.data
            .entries
            .get(&format!("{}:{}", provider, pkg))
            .filter(|(cached, _)| cached == content)
            .map(|(_, verdict)| verdict)
    }

    fn insert(&mut self, provider: &str, pkg: &str, content: &str, verdict: Verdict) {
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

/// Ordered by severity.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "UPPERCASE")]
enum Level {
    Safe,
    Caution,
    Danger,
}

impl Level {
    fn word(self) -> &'static str {
        match self {
            Level::Safe => "SAFE",
            Level::Caution => "CAUTION",
            Level::Danger => "DANGER",
        }
    }

    fn style(self, c: Colors) -> Style {
        match self {
            Level::Safe => c.upgrade,
            Level::Caution => c.warning,
            Level::Danger => c.error,
        }
    }
}

/// Also the LLM's response, so an unknown verdict fails to parse: servers
/// that ignore the schema can return anything.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
struct Verdict {
    #[serde(alias = "verdict")]
    level: Level,
    #[serde(default)]
    findings: Vec<String>,
    /// set on cache hits, never stored
    #[serde(skip)]
    cached: bool,
}

fn parse_response(response: &str) -> Result<Verdict> {
    serde_json::from_str(response.trim()).context(tr!("LLM returned malformed JSON"))
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

/// One request per package: a PKGBUILD can't steer another package's verdict,
/// one failure loses one verdict, and large updates stay within context.
async fn check_pkg(
    config: &Config,
    client: &Client,
    model: &str,
    pkg: &str,
    content: &str,
) -> Result<Verdict> {
    let request = ChatRequest {
        model: model.to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: format!("Package {}:\n{}", pkg, content),
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
                    "properties": {
                        "verdict": { "type": "string", "enum": ["SAFE", "CAUTION", "DANGER"] },
                        "findings": { "type": "array", "items": { "type": "string" } },
                    },
                    "required": ["verdict", "findings"],
                    "additionalProperties": false,
                },
            },
        }),
    };

    let url = format!(
        "{}/chat/completions",
        config.llm_api_url.trim_end_matches('/')
    );
    let mut req = client.post(url).json(&request);
    if let Some(key) = &config.llm_api_key {
        req = req.bearer_auth(&key.0);
    }
    let response = req.send().await.context(tr!("failed to contact LLM API"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!(tr!(
            "LLM API error {status}: {body}",
            status = status,
            body = api_error_message(&body)
        ));
    }

    let parsed: ChatResponse = response
        .json()
        .await
        .context(tr!("failed to parse LLM API response"))?;
    let answer = parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .context(tr!("empty response from LLM API"))?;
    parse_response(&answer)
}

/// Verdicts for one review: a summary goes at the top of the review and a
/// note under each package's header, next to its diff.
pub struct LlmReport {
    model: String,
    pkgs: Vec<String>,
    verdicts: HashMap<String, Verdict>,
    /// already printable
    warnings: Vec<String>,
}

/// None when no model is configured or the user declines. Verdicts are cached
/// per API URL and model, so switching either re-reviews.
/// The `error.message` of an OpenAI-style error body (Gemini wraps it in an
/// array), else the body on one line, cut short in case it is an HTML page.
fn api_error_message(body: &str) -> String {
    let json: Option<serde_json::Value> = serde_json::from_str(body).ok();
    let message = json.as_ref().and_then(|v| {
        let v = v.as_array().and_then(|a| a.first()).unwrap_or(v);
        v["error"]["message"].as_str()
    });
    match message {
        Some(m) => m.to_string(),
        None => body
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(300)
            .collect(),
    }
}

/// One warning per distinct error, so an outage isn't repeated per package.
fn failure_warnings(failures: Vec<(&str, String)>, total: usize) -> Vec<String> {
    let mut groups: Vec<(String, Vec<&str>)> = Vec::new();
    for (pkg, err) in failures {
        match groups.iter_mut().find(|(e, _)| *e == err) {
            Some((_, pkgs)) => pkgs.push(pkg),
            None => groups.push((err, vec![pkg])),
        }
    }
    groups
        .into_iter()
        .map(|(err, pkgs)| {
            if pkgs.len() == total {
                tr!("check failed: {err}", err = err)
            } else {
                tr!(
                    "{pkgs}: check failed: {err}",
                    pkgs = pkgs.join(", "),
                    err = err
                )
            }
        })
        .collect()
}

pub fn check(config: &Config, fetch: &aur_fetch::Fetch, pkgs: &[&str]) -> Option<LlmReport> {
    let model = config.llm_model.as_ref()?;
    if !ask(config, &tr!("Verify PKGBUILDs with LLM?"), false) {
        return None;
    }

    let mut report = LlmReport {
        model: model.clone(),
        pkgs: pkgs.iter().map(|p| p.to_string()).collect(),
        verdicts: HashMap::new(),
        warnings: Vec::new(),
    };

    // read once: what is cached is exactly what the LLM sees
    let mut pkg_contents: Vec<(&str, String)> = Vec::new();
    for &pkg in pkgs {
        match read_pkg(fetch, pkg) {
            Ok(content) => pkg_contents.push((pkg, content)),
            Err(e) => report.warnings.push(tr!(
                "{pkg}: not checked: {err}",
                pkg = pkg,
                err = printable(&format!("{:#}", e))
            )),
        }
    }

    let provider = format!("{} {}", config.llm_api_url, model);
    let mut cache = LlmCache::load(&config.cache_dir);
    let mut uncached: Vec<(&str, &str)> = Vec::new();
    for (pkg, content) in &pkg_contents {
        match cache.get(&provider, pkg, content) {
            Some(hit) => {
                let hit = Verdict {
                    cached: true,
                    ..hit.clone()
                };
                report.verdicts.insert(pkg.to_string(), hit);
            }
            None => uncached.push((pkg, content)),
        }
    }

    if !uncached.is_empty() {
        let c = config.color;
        let n = uncached.len();
        println!(
            "{} {}",
            c.action.paint("::"),
            c.bold.paint(tr!(
                "Checking {n} package with {model}..."
                    | "Checking {n} packages with {model}..." % n,
                model = model
            ))
        );

        // paru's shared async client: no total timeout, which slow models need
        let client = config.raur.client().clone();
        // ponytail: 4 at a time eases rate limits but can't promise to stay under a
        // per-minute cap, and a 429 is not retried; add backoff if that bites
        let results: Vec<Result<Verdict>> = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(
                stream::iter(&uncached)
                    .map(|&(pkg, content)| check_pkg(config, &client, model, pkg, content))
                    .buffered(4)
                    .collect(),
            )
        });

        let mut failures = Vec::new();
        for (&(pkg, content), result) in uncached.iter().zip(results) {
            match result {
                Ok(verdict) => {
                    cache.insert(&provider, pkg, content, verdict.clone());
                    report.verdicts.insert(pkg.to_string(), verdict);
                }
                Err(e) => failures.push((pkg, printable(&format!("{:#}", e)))),
            }
        }
        report.warnings.extend(failure_warnings(failures, n));
    }

    cache.save();
    Some(report)
}

impl LlmReport {
    /// Nothing got a verdict and a warning says why: rows or notes of
    /// "no verdict" would only repeat it.
    fn failed(&self) -> bool {
        self.verdicts.is_empty() && !self.warnings.is_empty()
    }

    /// One line per package, worst first, then any warnings.
    pub fn write_summary(&self, out: &mut impl Write, c: Colors) {
        let _ = writeln!(
            out,
            "{} {}",
            c.action.paint("::"),
            c.bold
                .paint(tr!("LLM security check ({model}):", model = self.model))
        );

        if !self.failed() {
            let mut pkgs: Vec<&String> = self.pkgs.iter().collect();
            pkgs.sort_by_key(|p| Reverse(self.verdicts.get(*p).map(|v| v.level)));
            let width = pkgs.iter().map(|p| p.len()).max().unwrap_or(0);

            for pkg in pkgs {
                let name = c.bold.paint(format!("{:<width$}", pkg, width = width));
                let Some(v) = self.verdicts.get(pkg) else {
                    let unknown = c.warning.paint(format!("{:<7}", "?"));
                    let _ = writeln!(out, "    {}  {}  {}", unknown, name, tr!("no verdict"));
                    continue;
                };
                let mut detail = Vec::new();
                if !v.findings.is_empty() {
                    let n = v.findings.len();
                    detail.push(tr!("{n} finding" | "{n} findings" % n));
                }
                if v.cached {
                    detail.push(tr!("(cached)"));
                }
                let level = v.level.style(c).paint(format!("{:<7}", v.level.word()));
                let line = format!("    {}  {}  {}", level, name, detail.join(" "));
                let _ = writeln!(out, "{}", line.trim_end());
            }
        }

        for warning in &self.warnings {
            let _ = writeln!(out, "  {} {}", c.warning.paint(tr!("warning:")), warning);
        }
        let _ = writeln!(out);
    }

    /// The verdict and findings, written under the package's `:: pkg` header.
    pub fn write_pkg(&self, out: &mut impl Write, c: Colors, pkg: &str) {
        let Some(v) = self.verdicts.get(pkg) else {
            if !self.failed() {
                let _ = writeln!(out, "  LLM: {}", c.warning.paint(tr!("no verdict")));
            }
            return;
        };
        let cached = if v.cached {
            format!(" {}", tr!("(cached)"))
        } else {
            String::new()
        };
        let level = v.level.style(c).paint(v.level.word());
        let _ = writeln!(out, "  LLM: {}{}", level, cached);
        for finding in &v.findings {
            let _ = writeln!(out, "  - {}", printable(finding));
        }
    }
}

/// LLM output, API errors and file names are untrusted: drop escape codes and
/// other control characters so they can't redraw the terminal.
fn printable(s: &str) -> String {
    s.chars()
        .filter(|&c| c == '\n' || !c.is_control())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn tmp() -> TempDir {
        tempfile::tempdir().expect("failed to create tempdir")
    }

    fn verdict(level: Level, findings: &[&str]) -> Verdict {
        Verdict {
            level,
            findings: findings.iter().map(|f| f.to_string()).collect(),
            cached: false,
        }
    }

    // --- parse_response ---

    #[test]
    fn parse_response_reads_verdict() {
        assert_eq!(
            parse_response(r#"{"verdict": "SAFE", "findings": []}"#).unwrap(),
            verdict(Level::Safe, &[])
        );
        assert_eq!(
            parse_response(r#" {"verdict": "DANGER", "findings": ["curl | sh", "eval"]} "#)
                .unwrap(),
            verdict(Level::Danger, &["curl | sh", "eval"])
        );
    }

    // Truncated, non-JSON or off-schema output must be an error, not a verdict.
    #[test]
    fn parse_response_rejects_malformed_json() {
        assert!(parse_response(r#"{"verdict": "SA"#).is_err());
        assert!(parse_response("SAFE").is_err());
        assert!(parse_response(r#"{"verdict": "Low risk"}"#).is_err());
    }

    #[test]
    fn failure_warnings_group_identical_errors() {
        let outage = vec![("a", "503".to_string()), ("b", "503".to_string())];
        assert_eq!(failure_warnings(outage, 2), ["check failed: 503"]);

        let mixed = vec![
            ("a", "503".to_string()),
            ("b", "400".to_string()),
            ("c", "503".to_string()),
        ];
        assert_eq!(
            failure_warnings(mixed, 4),
            ["a, c: check failed: 503", "b: check failed: 400"]
        );
    }

    #[test]
    fn cache_round_trip() {
        let dir = tmp();
        let mut cache = LlmCache::load(dir.path());
        assert!(cache.get("model-a", "pkg", "v1").is_none());
        cache.insert("model-a", "pkg", "v1", verdict(Level::Safe, &[]));
        let danger = verdict(Level::Danger, &["curl | bash"]);
        cache.insert("model-a", "pkg", "v2", danger.clone());
        cache.save();

        let cache = LlmCache::load(dir.path());
        assert_eq!(cache.get("model-a", "pkg", "v2"), Some(&danger));
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
        a.insert("m", "pkg-a", "a", verdict(Level::Safe, &[]));
        b.insert("m", "pkg-b", "b", verdict(Level::Danger, &[]));
        a.save();
        b.save();

        let cache = LlmCache::load(dir.path());
        assert_eq!(cache.get("m", "pkg-a", "a").unwrap().level, Level::Safe);
        assert_eq!(cache.get("m", "pkg-b", "b").unwrap().level, Level::Danger);
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

    // Colors::default() paints nothing, so this checks layout and order.
    #[test]
    fn report_renders_summary_worst_first_and_notes() {
        let mut cached = verdict(Level::Safe, &[]);
        cached.cached = true;
        let report = LlmReport {
            model: "m".to_string(),
            pkgs: ["ok", "bad-pkg", "gone", "meh"].map(String::from).to_vec(),
            verdicts: HashMap::from([
                ("ok".to_string(), cached),
                (
                    "bad-pkg".to_string(),
                    verdict(Level::Danger, &["curl | sh", "http source"]),
                ),
                (
                    "meh".to_string(),
                    verdict(Level::Caution, &["\x1b[2Kuses eval"]),
                ),
            ]),
            warnings: vec!["check failed: timeout".to_string()],
        };
        let c = Colors::default();

        let mut out = Vec::new();
        report.write_summary(&mut out, c);
        assert_eq!(
            String::from_utf8(out).unwrap(),
            ":: LLM security check (m):
    DANGER   bad-pkg  2 findings
    CAUTION  meh      1 finding
    SAFE     ok       (cached)
    ?        gone     no verdict
  warning: check failed: timeout

"
        );

        let mut out = Vec::new();
        report.write_pkg(&mut out, c, "meh");
        report.write_pkg(&mut out, c, "ok");
        report.write_pkg(&mut out, c, "gone");
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "  LLM: CAUTION\n  - [2Kuses eval\n  LLM: SAFE (cached)\n  LLM: no verdict\n"
        );

        // a failed check shows its warning once, with no rows or notes
        let failed = LlmReport {
            verdicts: HashMap::new(),
            ..report
        };
        let mut out = Vec::new();
        failed.write_summary(&mut out, c);
        failed.write_pkg(&mut out, c, "gone");
        assert_eq!(
            String::from_utf8(out).unwrap(),
            ":: LLM security check (m):\n  warning: check failed: timeout\n\n"
        );
    }

    #[test]
    fn api_error_message_extracts_message() {
        let gemini =
            r#"[{"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}]"#;
        let openai = r#"{"error": {"message": "bad key", "type": "invalid_request_error"}}"#;
        assert_eq!(api_error_message(gemini), "high demand");
        assert_eq!(api_error_message(openai), "bad key");
        assert_eq!(
            api_error_message("<html>\n  <b>502</b>\n</html>"),
            "<html> <b>502</b> </html>"
        );
        assert_eq!(api_error_message(&"x".repeat(1000)).len(), 300);
    }

    // A canned reply on a local port: the request goes out as the API expects
    // and the reply comes back as a verdict, without a real server.
    #[tokio::test(flavor = "multi_thread")]
    async fn check_pkg_round_trip() {
        use std::io::{BufRead, BufReader, Read};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let (mut head, mut len) = (String::new(), 0);
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap();
                }
                head.push_str(&line);
            }
            let mut body = vec![0; len];
            reader.read_exact(&mut body).unwrap();

            let content = r#"{\"verdict\": \"CAUTION\", \"findings\": [\"http source\"]}"#;
            let answer = format!(
                r#"{{"choices": [{{"message": {{"role": "assistant", "content": "{}"}}}}]}}"#,
                content
            );
            write!(
                &stream,
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                answer.len(),
                answer
            )
            .unwrap();
            (head, body)
        });

        let mut config = Config::default();
        config.llm_api_url = format!("http://{}/v1/", addr);
        config.llm_api_key = Some(crate::config::Secret("k".to_string()));
        // the sandbox or user may have a proxy set
        let client = Client::builder().no_proxy().build().unwrap();
        let got = check_pkg(&config, &client, "m", "pkg", "files")
            .await
            .unwrap();
        assert_eq!(got, verdict(Level::Caution, &["http source"]));

        let (head, body) = server.join().unwrap();
        assert!(head.starts_with("POST /v1/chat/completions "), "{}", head);
        assert!(head
            .to_ascii_lowercase()
            .contains("authorization: bearer k"));
        let req: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(req["model"], "m");
        assert_eq!(req["messages"][1]["content"], "Package pkg:\nfiles");
        assert!(req.get("temperature").is_none());
    }
}
