fn main() {
    // Tauri build
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
