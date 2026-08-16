fn main() {
    // 环境检测：在编译早期就给出清晰报错/警告，替代深层依赖的晦涩链接错误。
    // （详见 Cargo.toml 中 banqi-py-collector / banqi-tauri 的注释）
    check_py_collector_env();
    check_tauri_env();

    // Tauri build（仅在启用 tauri feature 时编译，避免训练/纯库场景拉取 GTK 依赖）
    #[cfg(feature = "tauri")]
    tauri_build::build();

    // libtorch linking configuration - 仅在启用 torch 特性时链接
    #[cfg(feature = "torch")]
    {
        let os = std::env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
        match os.as_str() {
            "linux" => {
                // 设置 NVIDIA 环境变量以解决显式同步问题
                println!("cargo:rustc-env=__NV_DISABLE_EXPLICIT_SYNC=1");
                if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
                    println!(
                        "cargo:rustc-link-arg=-Wl,-rpath={}",
                        lib_path.to_string_lossy()
                    );
                }
                println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
                println!("cargo:rustc-link-arg=-ltorch");
            }
            "windows" => {
                if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
                    println!(
                        "cargo:rustc-link-arg=-Wl,-rpath={}",
                        lib_path.to_string_lossy()
                    );
                }
                println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
                println!("cargo:rustc-link-arg=-ltorch");
            }
            _ => {}
        }
    }

    // gRPC code generation (已禁用 - 现在使用本地模型推理)
    // 如果需要重新启用 gRPC，请取消以下注释并创建 proto/banqi.proto 文件
    // tonic_build::configure()
    //     .build_server(false)
    //     .build_client(true)
    //     .compile_protos(&["proto/banqi.proto"], &["proto"])
    //     .unwrap_or_else(|e| panic!("Failed to compile protos: {}", e));
    // println!("cargo:rerun-if-changed=proto/banqi.proto");
}

// ── 环境检测：banqi-py-collector ──────────────────────────────────────────
// 嵌入 Python 的独立二进制必须链接**共享版** libpython（*.so / *.dylib）。
// pyenv 默认只编译静态 libpython3.x.a，链接时会报晦涩的
// "relocation R_X86_64_PC32 cannot be used against symbol 'stderr'" 错误。
//
// 触发条件：`pyo3` 已启用但 `pyo3-extension` 未启用。
//   - `--features pyo3`（构建 banqi-py-collector bin / 本地检查）→ 触发
//   - `--features pyo3-extension`（maturin 构建 wheel）→ 不触发（扩展模块由
//     宿主 Python 提供符号，无需链接 libpython）
//
// 注意：crate 级 build script 无法通过 CARGO_BIN_NAME 区分 target
// （该变量在 build script 中为空），故用 feature 组合作为判断依据。
fn check_py_collector_env() {
    let pyo3_enabled = std::env::var("CARGO_FEATURE_PYO3").is_ok();
    let extension_enabled = std::env::var("CARGO_FEATURE_PYO3_EXTENSION").is_ok();
    if !pyo3_enabled || extension_enabled {
        return;
    }

    // 优先用 pkg-config 判断是否能找到 libpython 的共享库
    if pkg_config_has_python() {
        return;
    }

    // 回退：通过 python 的 sysconfig 定位 libdir，检查是否存在共享库
    if let Some(libdir) = python_libdir() {
        let has_shared = std::fs::read_dir(&libdir)
            .map(|it| {
                it.filter_map(Result::ok).any(|e| {
                    let n = e.file_name().to_string_lossy().to_string();
                    (n.starts_with("libpython") && n.ends_with(".so"))
                        || (n.starts_with("libpython") && n.ends_with(".dylib"))
                })
            })
            .unwrap_or(false);
        if !has_shared {
            // 用 cargo:warning 前缀输出，cargo 才会在终端显示 build script 的消息
            println!(
                "cargo:warning=⚠️  banqi-py-collector 链接失败预警：当前 Python 环境({:?})只有静态 libpython，没有共享库(.so)。\n\
                 cargo:warning=  独立二进制必须链接 libpython.so，否则链接期会报：\n\
                 cargo:warning=  \"relocation R_X86_64_PC32 cannot be used against symbol 'stderr'\"。\n\
                 cargo:warning=  解决方案：\n\
                 cargo:warning=    · pyenv：PYTHON_CONFIGURE_OPTS=\"--enable-shared\" pyenv install 3.11.x\n\
                 cargo:warning=    · 系统包：apt install libpython3.11-dev（提供 libpython3.11.so）\n\
                 cargo:warning=    · 或改用 maturin 构建 wheel（cargo build --features pyo3-extension）",
                libdir
            );
        }
    }
}

// 用 pkg-config 检查是否具备 libpython 共享库（返回 true = 可用）
fn pkg_config_has_python() -> bool {
    let py_major_minor = python_version().unwrap_or_else(|| "3.11".to_string());
    let name = format!("python-{}", py_major_minor);
    std::process::Command::new("pkg-config")
        .args(["--exists", &name])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

// 读取当前 Python 的主次版本（如 "3.11"）
fn python_version() -> Option<String> {
    let out = std::process::Command::new("python3")
        .args(["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"])
        .output()
        .ok()?;
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

// 通过 sysconfig 获取 Python 库目录
fn python_libdir() -> Option<String> {
    let out = std::process::Command::new("python3")
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')",
        ])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

// ── 环境检测：tauri ───────────────────────────────────────────────────────
// 桌面 GUI 需要系统级 GTK/WebKit 开发库。缺库时 atk-sys/pango-sys/gdk-sys
// 的 build script 会失败（典型报错 "Package atk was not found"），这里提前
// 用 pkg-config 检测并给出安装命令。
//
// 注意：若 atk-sys 等**依赖**先于本 crate 编译失败，此 warning 可能来不及
// 打印，此时仍可参考 Cargo.toml 中 banqi-tauri 的注释与依赖自身的报错。
fn check_tauri_env() {
    if std::env::var("CARGO_FEATURE_TAURI").is_err() {
        return;
    }

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
                 cargo:warning=  （macOS 无需安装；Windows 需 WebView2。）\n\
                 cargo:warning=  若仅做训练/数据收集，请勿启用 tauri feature。",
                pkg
            );
            return;
        }
    }
}
