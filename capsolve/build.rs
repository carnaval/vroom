use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

fn compile_slang(path: &Path, output_path: &Path) {
    let sdk_version = "1.4.350.0";
    let slangc = format!("C:\\VulkanSDK\\{}\\Bin\\slangc.exe", sdk_version);
    //let slangc = "C:\\Users\\carna\\Downloads\\slang-2026.14-windows-x86_64\\bin\\slangc.exe";

    let mut proc = std::process::Command::new(slangc)
        //.arg("-fvk-use-c-layout")
        .arg("-matrix-layout-column-major")
        .arg("-fvk-use-entrypoint-name")
        .args(&["-target", "spirv"])
        .args(&["-profile", "spirv_1_3"])
        .args(&["-g1"])
        //.arg("-report-perf-benchmark")
        .args(&["-O2"])
        .arg(path)
        .arg("-o")
        .arg(&output_path)
        .spawn()
        .expect("failed to spawn slangc");
    let result = proc.wait().expect("failed to wait for slangc");

    if !result.success() {
        panic!("compilation failed");
    }
}

fn main2() {
    /*
    let out_dir = env::var_os("OUT_DIR").unwrap();

    let shader_path = Path::new("src/shader.slang");
    let output_path = Path::new(&out_dir).join(
        shader_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned()
            + ".spv",
    );
    compile_slang(&shader_path, &output_path);
    // aaa
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed={shader_path:?}");
    */
}

fn main() {
    let slang_dir = PathBuf::from(env::var_os("SLANG_DIR").expect("SLANG_DIR missing"));

    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    for name in ["slang.dll", "slang-compiler.dll"] {
        let src = slang_dir.join("bin").join(name);

        fs::copy(&src, out.join(name)).unwrap_or_else(|e| panic!("{}: {e}", src.display()));

        println!("cargo::rerun-if-changed={}", src.display());
    }

    println!("cargo::rustc-link-search=native={}", out.display());
}
