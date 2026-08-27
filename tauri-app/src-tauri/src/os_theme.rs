//! OS appearance detection: light/dark theme plus the accent color where the OS
//! exposes it. Everything here runs exactly once at application startup; there
//! is deliberately no runtime theme-change listener anywhere in the app.

use tauri::{App, Manager};

/// Current OS light/dark theme via the official Tauri window API, which itself
/// uses the official platform methods (Windows apps-use-light-theme setting,
/// macOS NSAppearance, Linux GTK/freedesktop settings). The window has no
/// explicit `theme` set in tauri.conf.json, so this reports the system theme.
pub fn detect_theme(app: &App) -> &'static str {
    match app
        .get_webview_window("main")
        .and_then(|w| w.theme().ok())
    {
        Some(tauri::Theme::Dark) => "dark",
        _ => "light",
    }
}

/// OS accent color as "#rrggbb" when the platform exposes it, read via the
/// official OS settings interfaces. `None` means "no accent exposed" and the UI
/// keeps its built-in default accent.
pub fn detect_accent_color() -> Option<String> {
    platform_accent_color()
}

#[cfg(target_os = "windows")]
fn platform_accent_color() -> Option<String> {
    use winreg::enums::HKEY_CURRENT_USER;
    use winreg::RegKey;
    // Official per-user accent color: HKCU\Software\Microsoft\Windows\DWM\AccentColor
    // (DWORD in ABGR byte order).
    let dwm = RegKey::predef(HKEY_CURRENT_USER)
        .open_subkey(r"Software\Microsoft\Windows\DWM")
        .ok()?;
    let abgr: u32 = dwm.get_value("AccentColor").ok()?;
    Some(format!(
        "#{:02x}{:02x}{:02x}",
        abgr & 0xff,
        (abgr >> 8) & 0xff,
        (abgr >> 16) & 0xff
    ))
}

#[cfg(target_os = "linux")]
fn platform_accent_color() -> Option<String> {
    // GNOME exposes the accent color by name via its official settings tool.
    let out = std::process::Command::new("gsettings")
        .args(["get", "org.gnome.desktop.interface", "accent-color"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let name = String::from_utf8_lossy(&out.stdout)
        .trim()
        .trim_matches('\'')
        .to_string();
    let hex = match name.as_str() {
        "blue" => "#3584e4",
        "teal" => "#2190a4",
        "green" => "#3a944a",
        "yellow" => "#c88800",
        "orange" => "#ed5b00",
        "red" => "#e62d42",
        "pink" => "#d56199",
        "purple" => "#9141ac",
        "slate" => "#6f8396",
        _ => return None,
    };
    Some(hex.to_string())
}

#[cfg(target_os = "macos")]
fn platform_accent_color() -> Option<String> {
    // macOS global accent preference (AppleAccentColor). A missing key means the
    // default blue/multicolor, in which case the built-in accent is kept.
    let out = std::process::Command::new("defaults")
        .args(["read", "-g", "AppleAccentColor"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let id: i32 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    let hex = match id {
        0 => "#ff3b30",  // red
        1 => "#ff9500",  // orange
        2 => "#ffcc00",  // yellow
        3 => "#28cd41",  // green
        4 => "#007aff",  // blue
        5 => "#bf5af2",  // purple
        6 => "#ff375f",  // pink
        -1 => "#98989d", // graphite
        _ => return None,
    };
    Some(hex.to_string())
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn platform_accent_color() -> Option<String> {
    None
}
