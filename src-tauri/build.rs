fn main() {
    // 环境检测：桌面 GUI 需要系统级 GTK/WebKit 开发库。缺库时 atk-sys/pango-sys/gdk-sys
    // 的 build script 会失败（典型报错 "Package atk was not found"），这里提前
    // 用 pkg-config 检测并给出安装命令。
    //
    // 注意：若 atk-sys 等**依赖**先于本 crate 编译失败，此 warning 可能来不及
    // 打印，此时仍可参考依赖自身的报错。
    let required = ["gtk+-3.0", "webkit2gtk-4.1"];
    for pkg in required {
        let ok = std::process::Command::new("pkg-config")
            .args(["--exists", pkg])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            println!(
                "cargo:warning=⚠️  banqi-tauri 编译依赖缺失：pkg-config 找不到 `{}`。\n\
                 cargo:warning=  桌面 GUI 需要系统级 GTK/WebKit 开发库，请安装：\n\
                 cargo:warning=    sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev \\\n\
                 cargo:warning=      libayatana-appindicator3-dev librsvg2-dev patchelf\n\
                 cargo:warning=  （macOS 无需安装；Windows 需 WebView2。）",
                pkg
            );
            return;
        }
    }

    tauri_build::build()
}
