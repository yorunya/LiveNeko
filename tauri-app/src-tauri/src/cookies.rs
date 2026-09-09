//! Browser cookie import, equivalent to `yt-dlp --cookies-from-browser <browser>`.
//!
//! Mirrors `yt_dlp/cookies.py` (2026.08.19) for firefox / chrome / edge:
//! - firefox: `moz_cookies` table of the newest `cookies.sqlite` under the
//!   profile roots (DB is copied to a temp file first because the browser locks it).
//! - chrome/edge: the newest `Cookies` sqlite file under the browser's
//!   `User Data` dir. Values are decrypted like yt-dlp's Windows decryptor:
//!   `v10`-prefixed blobs are AES-256-GCM (nonce = first 12 bytes, tag = last
//!   16) with the DPAPI-unwrapped `os_crypt.encrypted_key` from `Local State`;
//!   anything else is DPAPI-encrypted as a whole. `meta.version >= 24` values
//!   carry a 32-byte host-hash prefix that must be trimmed. Like yt-dlp, no
//!   special handling exists for Chromium's newer app-bound (`v20`) cookies,
//!   so those fail to decrypt exactly as they do in yt-dlp.
//!
//! On the download side only the helper `cookie_header` / `find_cookie` are
//! used: cookies are matched per request host (yt-dlp does the same domain
//! matching through http.cookiejar).

use std::path::{Path, PathBuf};

use rusqlite::Connection;

/// How many profile dirs deep to look for cookie databases.
const MAX_WALK_DEPTH: usize = 6;
/// firefox cookies DB schema >= 16 stores expiry in milliseconds (FF142+).
const MAX_SUPPORTED_FF_SCHEMA: i64 = 17;

#[derive(Clone, Debug)]
pub struct Cookie {
    pub name: String,
    pub value: String,
    /// As stored, e.g. ".youtube.com" or "www.bilibili.com".
    pub domain: String,
    /// Kept for completeness; requests are matched by host only.
    #[allow(dead_code)]
    pub path: String,
    /// Unix seconds; None = session cookie.
    pub expires: Option<i64>,
    pub secure: bool,
}

/// Extract cookies from the selected browser (same sources as yt-dlp).
pub fn load_browser_cookies(browser: &str, on_log: &dyn Fn(&str)) -> Result<Vec<Cookie>, String> {
    let browser = browser.trim().to_lowercase();
    match browser.as_str() {
        "firefox" => extract_firefox(on_log),
        "chrome" | "edge" => extract_chromium(&browser, on_log),
        other => Err(format!(
            "unsupported browser for cookie import: \"{other}\" (use firefox, chrome or edge)"
        )),
    }
}

/// http.cookiejar.domain_match: host equals the domain, or the domain (with a
/// leading dot) is a dot-suffix of the host.
pub fn domain_matches(host: &str, domain: &str) -> bool {
    let host = host.to_ascii_lowercase();
    let d = domain.trim_start_matches('.').to_ascii_lowercase();
    host == d || (host.len() > d.len() && host.ends_with(&format!(".{d}")))
}

impl Cookie {
    pub fn alive(&self, now: i64, https: bool) -> bool {
        (!self.secure || https) && self.expires.map(|e| e > now).unwrap_or(true)
    }
}

/// Build the `Cookie` header value for a request to `host` (None if nothing matches).
pub fn cookie_header(cookies: &[Cookie], host: &str) -> Option<String> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let pairs: Vec<String> = cookies
        .iter()
        .filter(|c| domain_matches(host, &c.domain) && c.alive(now, true))
        .map(|c| format!("{}={}", c.name, c.value))
        .collect();
    if pairs.is_empty() {
        None
    } else {
        Some(pairs.join("; "))
    }
}

/// Look up a single cookie value by host + name.
pub fn find_cookie<'a>(cookies: &'a [Cookie], host: &str, name: &str) -> Option<&'a str> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    cookies
        .iter()
        .find(|c| c.name == name && domain_matches(host, &c.domain) && c.alive(now, true))
        .map(|c| c.value.as_str())
}

// ---------------------------------------------------------------------------
// firefox
// ---------------------------------------------------------------------------

fn firefox_roots() -> Vec<PathBuf> {
    // Same roots as yt-dlp `_firefox_browser_dirs` on Windows.
    let mut roots = Vec::new();
    if let Some(appdata) = std::env::var_os("APPDATA") {
        roots.push(PathBuf::from(appdata).join(r"Mozilla\Firefox\Profiles"));
    }
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        roots.push(PathBuf::from(local)
            .join(r"Packages\Mozilla.Firefox_n80bbvh6b1yt2\LocalCache\Roaming\Mozilla\Firefox\Profiles"));
    }
    roots
}

fn extract_firefox(on_log: &dyn Fn(&str)) -> Result<Vec<Cookie>, String> {
    on_log("Extracting cookies from firefox");
    // Same glob patterns as `_firefox_cookie_dbs`: "", "*/", "Profiles/*/".
    let mut dbs: Vec<PathBuf> = Vec::new();
    for root in firefox_roots() {
        if let Ok(rd) = std::fs::read_dir(&root) {
            for entry in rd.flatten() {
                let p = entry.path().join("cookies.sqlite");
                if p.is_file() {
                    dbs.push(p);
                }
            }
        }
        let profiles = root.join("Profiles");
        if let Ok(rd) = std::fs::read_dir(&profiles) {
            for entry in rd.flatten() {
                let p = entry.path().join("cookies.sqlite");
                if p.is_file() {
                    dbs.push(p);
                }
            }
        }
    }
    let db = newest(&dbs).ok_or(
        "could not find firefox cookies database (is Firefox installed? open its profile once, then retry)",
    )?;
    on_log(&format!("Extracting cookies from: {}", db.display()));

    let (conn, _guard) = open_database_copy(&db)?;
    let schema: i64 = conn
        .query_row("PRAGMA user_version;", [], |r| r.get(0))
        .unwrap_or(0);
    if schema > MAX_SUPPORTED_FF_SCHEMA {
        on_log(&format!(
            "Possibly unsupported firefox cookies database version: {schema}"
        ));
    }
    let ms_expiry = schema >= 16;
    let mut stmt = conn
        .prepare("SELECT host, name, value, path, expiry, isSecure FROM moz_cookies")
        .map_err(|e| format!("firefox cookies query: {e}"))?;
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Option<i64>>(4)?,
                r.get::<_, i64>(5)?,
            ))
        })
        .map_err(|e| format!("firefox cookies query: {e}"))?;
    let mut out = Vec::new();
    for row in rows.flatten() {
        let (host, name, value, path, expiry, secure) = row;
        // FF142+ stores milliseconds; yt-dlp divides by 1000.
        let expires = expiry.map(|e| if ms_expiry { e / 1000 } else { e });
        out.push(Cookie {
            name,
            value,
            domain: host,
            path,
            expires,
            secure: secure != 0,
        });
    }
    on_log(&format!("Extracted {} cookies from firefox", out.len()));
    Ok(out)
}

// ---------------------------------------------------------------------------
// chrome / edge
// ---------------------------------------------------------------------------

fn chromium_dir(browser: &str) -> Option<PathBuf> {
    let local = std::env::var_os("LOCALAPPDATA")?;
    let dir = match browser {
        "chrome" => r"Google\Chrome\User Data",
        "edge" => r"Microsoft\Edge\User Data",
        _ => return None,
    };
    Some(PathBuf::from(local).join(dir))
}

fn extract_chromium(browser: &str, on_log: &dyn Fn(&str)) -> Result<Vec<Cookie>, String> {
    on_log(&format!("Extracting cookies from {browser}"));
    let root = chromium_dir(browser)
        .ok_or("cannot resolve the browser's user data directory (%LOCALAPPDATA% missing)")?;
    // yt-dlp `_find_files`: walk the whole dir, newest file named "Cookies" wins.
    let mut files: Vec<PathBuf> = Vec::new();
    walk_files(&root, "Cookies", 0, &mut files);
    let db = newest(&files)
        .ok_or_else(|| format!("could not find {browser} cookies database in {}", root.display()))?;
    on_log(&format!("Extracting cookies from: {}", db.display()));

    let (conn, _guard) = open_database_copy(&db)?;
    let meta_version: u32 = conn
        .query_row("SELECT value FROM meta WHERE key = 'version'", [], |r| {
            r.get::<_, String>(0)
        })
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    // column is `is_secure` on newer schemas, `secure` on older ones
    let secure_col = if conn
        .prepare("PRAGMA table_info(cookies)")
        .and_then(|mut s| {
            s.query_map([], |r| r.get::<_, String>(1))
                .map(|rows| rows.flatten().any(|name| name == "is_secure"))
        })
        .unwrap_or(false)
    {
        "is_secure"
    } else {
        "secure"
    };
    let sql = format!(
        "SELECT host_key, name, value, encrypted_value, path, expires_utc, {secure_col} FROM cookies"
    );
    let mut stmt = conn.prepare(&sql).map_err(|e| format!("{browser} cookies query: {e}"))?;
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, Vec<u8>>(0)?,
                r.get::<_, Vec<u8>>(1)?,
                r.get::<_, Vec<u8>>(2)?,
                r.get::<_, Vec<u8>>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, i64>(5)?,
                r.get::<_, i64>(6)?,
            ))
        })
        .map_err(|e| format!("{browser} cookies query: {e}"))?;

    // DPAPI-unwrapped AES key from `Local State` (yt-dlp `_get_windows_v10_key`).
    let v10_key = match windows_v10_key(&root) {
        Ok(k) => k,
        Err(e) => {
            on_log(&e);
            None
        }
    };
    let mut warned_v10_no_key = false;
    let mut failed = 0usize;
    let mut out = Vec::new();
    for row in rows.flatten() {
        let (host_key, name, value, encrypted_value, path, expires_utc, secure) = row;
        let text = if !value.is_empty() {
            Some(String::from_utf8_lossy(&value).to_string())
        } else if encrypted_value.is_empty() {
            None
        } else {
            match decrypt_chromium_value(&encrypted_value, v10_key.as_deref(), meta_version, &mut warned_v10_no_key, on_log) {
                Some(v) => Some(v),
                None => {
                    failed += 1;
                    None
                }
            }
        };
        let Some(value) = text else { continue };
        // Chrome epoch: 100 ns since 1601-01-01. 0 = session cookie.
        let expires = if expires_utc == 0 {
            None
        } else {
            Some(expires_utc / 10_000_000 - 11_644_473_600)
        };
        out.push(Cookie {
            name: String::from_utf8_lossy(&name).to_string(),
            value,
            domain: String::from_utf8_lossy(&host_key).to_string(),
            path,
            expires,
            secure: secure != 0,
        });
    }
    if failed > 0 {
        on_log(&format!(
            "{failed} cookie(s) could not be decrypted (Chromium app-bound cookies are not decryptable, same as yt-dlp)"
        ));
    }
    on_log(&format!("Extracted {} cookies from {browser}", out.len()));
    Ok(out)
}

/// Decrypt one `encrypted_value` blob exactly like `WindowsChromeCookieDecryptor.decrypt`.
fn decrypt_chromium_value(
    enc: &[u8],
    v10_key: Option<&[u8]>,
    meta_version: u32,
    warned_no_key: &mut bool,
    on_log: &dyn Fn(&str),
) -> Option<String> {
    if enc.starts_with(b"v10") {
        let Some(key) = v10_key else {
            if !*warned_no_key {
                *warned_no_key = true;
                on_log("cannot decrypt v10 cookies: no key found in Local State");
            }
            return None;
        };
        // kNonceLength = 12, AEAD tag = 16 (os_crypt_win.cc)
        if enc.len() < 3 + 12 + 16 {
            return None;
        }
        let nonce = &enc[3..3 + 12];
        let tag_start = enc.len() - 16;
        let ciphertext = &enc[15..tag_start];
        let tag = &enc[tag_start..];
        let plain = aes_gcm_decrypt(key, nonce, ciphertext.to_vec(), tag).ok()?;
        // meta_version >= 24 prefixes the plaintext with a 32-byte host hash.
        let plain = if meta_version >= 24 && plain.len() > 32 {
            plain[32..].to_vec()
        } else {
            plain
        };
        String::from_utf8(plain).ok()
    } else {
        // Any other prefix: the whole blob is DPAPI encrypted.
        let plain = dpapi_unprotect(enc).ok()?;
        String::from_utf8(plain).ok()
    }
}

/// Read + DPAPI-decrypt `os_crypt.encrypted_key` from the newest `Local State`
/// under the browser root (yt-dlp `_get_windows_v10_key`).
fn windows_v10_key(browser_root: &Path) -> Result<Option<Vec<u8>>, String> {
    let mut files: Vec<PathBuf> = Vec::new();
    walk_files(browser_root, "Local State", 0, &mut files);
    let Some(path) = newest(&files) else {
        return Err("could not find Local State file (browser never run?)".into());
    };
    let text = std::fs::read_to_string(&path).map_err(|e| format!("read Local State: {e}"))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse Local State: {e}"))?;
    let Some(b64) = json
        .pointer("/os_crypt/encrypted_key")
        .and_then(serde_json::Value::as_str)
    else {
        return Err("no encrypted key in Local State".into());
    };
    use base64::Engine;
    let blob = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| format!("decode encrypted_key: {e}"))?;
    let Some(rest) = blob.strip_prefix(b"DPAPI") else {
        return Err("invalid encrypted key prefix".into());
    };
    Ok(dpapi_unprotect(rest).ok())
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

/// Newest file by modification time (yt-dlp `_newest`).
fn newest(files: &[PathBuf]) -> Option<PathBuf> {
    files
        .iter()
        .max_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
        .cloned()
}

fn walk_files(dir: &Path, name: &str, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > MAX_WALK_DEPTH {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut subdirs = Vec::new();
    for entry in rd.flatten() {
        let p = entry.path();
        match entry.file_type() {
            Ok(ft) if ft.is_dir() => subdirs.push(p),
            Ok(ft) if ft.is_file() => {
                if entry.file_name().to_string_lossy() == name {
                    out.push(p);
                }
            }
            _ => {}
        }
    }
    for d in subdirs {
        walk_files(&d, name, depth + 1, out);
    }
}

/// yt-dlp `_open_database_copy`: the live DB is locked by the browser, so copy
/// it to a temp file and open the copy. The copy is removed when the returned
/// guard is dropped.
fn open_database_copy(db: &Path) -> Result<(Connection, TempDbGuard), String> {
    let tmp = std::env::temp_dir().join(format!(
        "liveneko_cookie_{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::copy(db, &tmp).map_err(|e| {
        format!(
            "could not copy cookie database {} (close the browser and retry): {e}",
            db.display()
        )
    })?;
    let conn = Connection::open(&tmp).map_err(|e| format!("open cookie copy: {e}"))?;
    Ok((conn, TempDbGuard(tmp)))
}

struct TempDbGuard(PathBuf);

impl Drop for TempDbGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

// ---------------------------------------------------------------------------
// crypto
// ---------------------------------------------------------------------------

#[cfg(windows)]
fn dpapi_unprotect(data: &[u8]) -> Result<Vec<u8>, String> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{CryptUnprotectData, CRYPT_INTEGER_BLOB};
    unsafe {
        let input = CRYPT_INTEGER_BLOB {
            cbData: data.len() as u32,
            pbData: data.as_ptr() as *mut u8,
        };
        let mut out = CRYPT_INTEGER_BLOB {
            cbData: 0,
            pbData: std::ptr::null_mut(),
        };
        if CryptUnprotectData(
            &input,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            &mut out,
        ) == 0
        {
            return Err(
                "Failed to decrypt with DPAPI (cookies encrypted for a different user account?)"
                    .into(),
            );
        }
        let slice = std::slice::from_raw_parts(out.pbData, out.cbData as usize);
        let v = slice.to_vec();
        LocalFree(out.pbData as _);
        Ok(v)
    }
}

#[cfg(not(windows))]
fn dpapi_unprotect(_data: &[u8]) -> Result<Vec<u8>, String> {
    Err("browser cookie decryption is only supported on Windows".into())
}

fn aes_gcm_decrypt(key: &[u8], nonce: &[u8], mut data: Vec<u8>, tag: &[u8]) -> Result<Vec<u8>, String> {
    use aes_gcm::aead::generic_array::GenericArray;
    use aes_gcm::aead::AeadInPlace;
    use aes_gcm::{Aes256Gcm, KeyInit};
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| format!("aes key: {e}"))?;
    let nonce = GenericArray::from_slice(nonce);
    let tag = GenericArray::from_slice(tag);
    cipher
        .decrypt_in_place_detached(&nonce, b"", &mut data, &tag)
        .map_err(|_| "failed to decrypt cookie (AES-GCM MAC check failed; possibly the key is wrong)".to_string())?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ck(name: &str, domain: &str) -> Cookie {
        Cookie {
            name: name.into(),
            value: "v".into(),
            domain: domain.into(),
            path: "/".into(),
            expires: None,
            secure: false,
        }
    }

    #[test]
    fn test_domain_matches() {
        assert!(domain_matches("www.youtube.com", ".youtube.com"));
        assert!(domain_matches("youtube.com", ".youtube.com"));
        assert!(domain_matches("youtube.com", "youtube.com"));
        assert!(!domain_matches("notyoutube.com", ".youtube.com"));
        assert!(!domain_matches("youtube.com.evil.io", ".youtube.com"));
        assert!(domain_matches("api.bilibili.com", ".bilibili.com"));
    }

    #[test]
    fn test_cookie_header_matches_and_expiry() {
        let cookies = vec![
            ck("SESSDATA", ".bilibili.com"),
            ck("OTHER", ".example.org"),
            Cookie {
                name: "OLD".into(),
                value: "v".into(),
                domain: ".bilibili.com".into(),
                path: "/".into(),
                expires: Some(1), // long expired
                secure: false,
            },
        ];
        let hdr = cookie_header(&cookies, "api.bilibili.com").unwrap();
        assert_eq!(hdr, "SESSDATA=v");
        assert!(cookie_header(&cookies, "www.youtube.com").is_none());
        assert_eq!(find_cookie(&cookies, "www.bilibili.com", "SESSDATA"), Some("v"));
        assert_eq!(find_cookie(&cookies, "www.bilibili.com", "OLD"), None);
    }
}
