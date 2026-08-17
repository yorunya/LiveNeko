use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};

/// A resident Python model server. Spawned once at pipeline start; the Python process loads its models a single time and then waits for JSON requests on stdin, writing JSON responses (and `{"progress":N}` lines) on stdout.
pub struct ModelServer {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
    /// Reusable read buffer so high-frequency progress lines don't allocate.
    line: String,
    cancel: Arc<AtomicBool>,
}

fn hide_console(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }
}

/// Write a log line to the shared log file (if one is open) and emit it to the UI. Every pipeline log line flows through this so the running log can be persisted next to the results.
pub fn log_line(
    app: &AppHandle,
    log_file: &Arc<Mutex<Option<std::fs::File>>>,
    item_id: &str,
    line: &str,
) {
    if let Ok(mut guard) = log_file.lock() {
        if let Some(f) = guard.as_mut() {
            let _ = writeln!(f, "[{item_id}] {line}");
        }
    }
    let _ = app.emit(
        "pipeline://log",
        serde_json::json!({
            "itemId": item_id, "line": line
        }),
    );
}

impl ModelServer {
    /// Spawn `python -u <script> <args>` and wait for the ready handshake. The script is expected to emit `{"cmd":"ready","ok":true,...}` once its models are loaded.
    pub fn spawn(
        app: &AppHandle,
        python: &str,
        script: &Path,
        args: &[String],
        cancel: Arc<AtomicBool>,
        pids: Arc<Mutex<Vec<u32>>>,
        log_file: Arc<Mutex<Option<std::fs::File>>>,
    ) -> Result<Self, String> {
        let mut cmd = Command::new(python);
        cmd.arg("-u").arg(script);
        cmd.args(args);
        hide_console(&mut cmd);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", script.display()))?;
        pids.lock().unwrap().push(child.id());
        let stdin = child.stdin.take().ok_or("no stdin on child")?;
        let stdout = child.stdout.take().ok_or("no stdout on child")?;

        // forward python stderr to the UI (and log file) as log lines
        let app2 = app.clone();
        let log_file2 = log_file.clone();
        let stderr = child.stderr.take().expect("stderr piped");
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(l) = line {
                    log_line(&app2, &log_file2, "model-server", &l);
                }
            }
        });

        let mut server = Self {
            child,
            stdin,
            reader: BufReader::new(stdout),
            line: String::new(),
            cancel,
        };

        // wait for the ready handshake (bounded: 2 minutes for model loading)
        let mut ready: Option<Value> = None;
        for _ in 0..2400 {
            match server.next_event() {
                Ok(ev) => {
                    if ev.get("cmd").and_then(|c| c.as_str()) == Some("ready") {
                        ready = Some(ev);
                        break;
                    }
                }
                Err(e) => return Err(format!("model server startup failed: {e}")),
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        match ready {
            Some(ev) if ev.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) => Ok(server),
            Some(ev) => {
                let msg = ev
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                Err(format!("model server failed to load: {msg}"))
            }
            None => Err("model server did not report ready in time".to_string()),
        }
    }

    fn next_event(&mut self) -> Result<Value, String> {
        loop {
            self.line.clear();
            let n = self
                .reader
                .read_line(&mut self.line)
                .map_err(|e| format!("read from model server: {e}"))?;
            if n == 0 {
                // EOF: the python process exited
                return Err("model server process exited".to_string());
            }
            let t = self.line.trim();
            if t.is_empty() {
                continue;
            }
            return serde_json::from_str(t).map_err(|e| format!("bad json from server: {e}: {t}"));
        }
    }

    /// Send a request and wait for its matching response, forwarding progress events to the UI as `pipeline://stage` events (including part info). Returns the response object (which has "ok").
    pub fn request(
        &mut self,
        app: &AppHandle,
        item_id: &str,
        stage: u8,
        part: u32,
        total_parts: u32,
        req: Value,
    ) -> Result<Value, String> {
        let mut line = serde_json::to_string(&req).map_err(|e| e.to_string())?;
        line.push('\n');
        self.stdin
            .write_all(line.as_bytes())
            .map_err(|e| format!("write to model server: {e}"))?;
        self.stdin.flush().map_err(|e| format!("flush: {e}"))?;

        let mut last_progress: Option<u64> = None;
        loop {
            if self.cancel.load(Ordering::SeqCst) {
                return Err("cancelled".to_string());
            }
            let ev = self.next_event()?;
            if let Some(p) = ev.get("progress").and_then(|v| v.as_u64()) {
                let p = p.min(100);
                // Skip unchanged values: servers can emit duplicate progress lines (e.g. a filter that repeats 100%), and each one costs a full JSON event round-trip to the UI.
                if last_progress != Some(p) {
                    last_progress = Some(p);
                    let _ = app.emit(
                        "pipeline://stage",
                        serde_json::json!({
                            "itemId": item_id,
                            "stage": stage,
                            "progress": p,
                            "part": part,
                            "totalParts": total_parts,
                        }),
                    );
                }
                continue;
            }
            return Ok(ev);
        }
    }

    pub fn shutdown(&mut self) {
        let _ = self.stdin.write_all(b"{\"cmd\":\"shutdown\"}\n");
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for ModelServer {
    fn drop(&mut self) {
        // best-effort cleanup so a cancelled/abandoned pipeline doesn't leak
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
