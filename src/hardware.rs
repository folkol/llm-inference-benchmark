use serde::{Deserialize, Serialize};
use sysinfo::{CpuExt, System, SystemExt};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub os: String,
    pub os_version: String,
    pub cpu_brand: String,
    pub cpu_cores_physical: usize,
    pub cpu_cores_logical: usize,
    pub ram_total_gb: f64,
    pub gpu_info: Vec<GpuInfo>,
    pub llama_cpp_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: String,
    pub vram_mb: Option<u64>,
}

impl HardwareInfo {
    /// Stable directory token: OS & arch, CPU model, GPU(s).
    ///
    /// Collapses whitespace and punctuation for use in filenames. A bench `run_id`
    /// (timestamp) should be appended by the caller, e.g. `{label}__{run_id}`.
    pub fn report_path_label(&self) -> String {
        let cpu = sanitize_path_token(&strip_tm_noise(&self.cpu_brand));
        let gpu = gpu_path_token(&self.gpu_info);

        let os = self.os.as_str();

        let mut s = format!("{os}-{arch}__CPU-{cpu}__GPU-{gpu}", arch = std::env::consts::ARCH);

        if s.len() > 200 {
            s.truncate(200);
            while s.ends_with('_') {
                s.pop();
            }
        }
        if s.is_empty() {
            "unknown-host".into()
        } else {
            s
        }
    }

    pub fn summary(&self) -> String {
        let mut lines = vec![
            format!("OS          : {} {}", self.os, self.os_version),
            format!(
                "CPU         : {} ({} physical / {} logical cores)",
                self.cpu_brand, self.cpu_cores_physical, self.cpu_cores_logical
            ),
            format!("RAM         : {:.1} GB", self.ram_total_gb),
        ];
        if self.gpu_info.is_empty() {
            lines.push("GPU         : none detected".to_string());
        }
        for g in &self.gpu_info {
            let vram = g
                .vram_mb
                .map(|v| format!(" ({} MB VRAM)", v))
                .unwrap_or_default();
            lines.push(format!("GPU         : {} [{}]{}", g.name, g.vendor, vram));
        }
        if let Some(v) = &self.llama_cpp_version {
            lines.push(format!("llama.cpp   : {}", v));
        }
        lines.join("\n")
    }
}

fn strip_tm_noise(s: &str) -> String {
    s.replace("(R)", "")
        .replace("(r)", "")
        .replace("(TM)", "")
        .replace("(tm)", "")
        .replace('™', "")
        .replace('®', "")
}

fn sanitize_path_token(raw: &str) -> String {
    let stripped = strip_tm_noise(raw.trim());
    let mut out = String::with_capacity(stripped.len());
    let mut pending_sep = false;

    for ch in stripped.chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_sep && !out.is_empty() {
                out.push('_');
            }
            pending_sep = false;
            out.push(ch);
            continue;
        }

        // Treat separators / punctuation clusters as single underscore between words.
        if ch.is_ascii_whitespace() || matches!(ch, '-' | '_' | '.' | '+' | '/') {
            if !out.is_empty() {
                pending_sep = true;
            }
            continue;
        }
    }

    let out = if out.is_empty() {
        return "unknown".into();
    } else if out.len() > 96 {
        out[..96].trim_end_matches('_').to_string()
    } else {
        out
    };

    out
}

fn gpu_path_token(gpus: &[GpuInfo]) -> String {
    if gpus.is_empty() {
        return "none".into();
    }
    if gpus.len() == 1 {
        return sanitize_path_token(&gpus[0].name);
    }
    format!(
        "{}_plus{}",
        sanitize_path_token(&gpus[0].name),
        gpus.len() - 1
    )
}

pub fn detect() -> HardwareInfo {
    let mut sys = System::new_all();
    sys.refresh_all();

    let os = std::env::consts::OS.to_string();
    let os_version = sys.os_version().unwrap_or_else(|| "unknown".to_string());
    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let cpu_cores_logical = sys.cpus().len();
    // Physical core count via sysinfo physical_core_count
    let cpu_cores_physical = sys.physical_core_count().unwrap_or(cpu_cores_logical);

    let ram_total_gb = sys.total_memory() as f64 / 1_073_741_824.0;

    let gpu_info = detect_gpus();

    HardwareInfo {
        os,
        os_version,
        cpu_brand,
        cpu_cores_physical,
        cpu_cores_logical,
        ram_total_gb,
        gpu_info,
        llama_cpp_version: None,
    }
}

fn detect_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    // Try nvidia-smi
    if let Ok(out) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            for line in text.lines() {
                let parts: Vec<&str> = line.splitn(2, ',').collect();
                if parts.len() == 2 {
                    let name = parts[0].trim().to_string();
                    let vram_mb = parts[1].trim().parse::<u64>().ok();
                    gpus.push(GpuInfo {
                        name,
                        vendor: "NVIDIA".to_string(),
                        vram_mb,
                    });
                }
            }
        }
    }

    // Try rocm-smi (AMD)
    if gpus.is_empty() {
        if let Ok(out) = std::process::Command::new("rocm-smi")
            .args(["--showproductname", "--json"])
            .output()
        {
            if out.status.success() {
                gpus.push(GpuInfo {
                    name: "AMD GPU (rocm-smi)".to_string(),
                    vendor: "AMD".to_string(),
                    vram_mb: None,
                });
            }
        }
    }

    // On macOS, try system_profiler for Apple Silicon
    #[cfg(target_os = "macos")]
    if gpus.is_empty() {
        if let Ok(out) = std::process::Command::new("system_profiler")
            .args(["SPDisplaysDataType", "-json"])
            .output()
        {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                // Simple heuristic: look for "Apple" GPU
                if text.contains("Apple") {
                    gpus.push(GpuInfo {
                        name: "Apple Silicon GPU".to_string(),
                        vendor: "Apple".to_string(),
                        vram_mb: None,
                    });
                }
            }
        }
    }

    gpus
}

#[cfg(test)]
mod tests {
    use super::{GpuInfo, HardwareInfo};

    #[test]
    fn report_path_label_matches_known_snapshot_host() {
        let hw = HardwareInfo {
            os: "windows".into(),
            os_version: "11".into(),
            cpu_brand: "Intel(R) Core(TM) Ultra 9 285K".into(),
            cpu_cores_physical: 24,
            cpu_cores_logical: 24,
            ram_total_gb: 31.4,
            gpu_info: vec![GpuInfo {
                name: "NVIDIA GeForce RTX 4080 SUPER".into(),
                vendor: "NVIDIA".into(),
                vram_mb: Some(16376),
            }],
            llama_cpp_version: None,
        };

        assert_eq!(
            hw.report_path_label(),
            "windows-x86_64__CPU-Intel_Core_Ultra_9_285K__GPU-NVIDIA_GeForce_RTX_4080_SUPER"
        );
    }

    #[test]
    fn report_path_label_without_gpu() {
        let hw = HardwareInfo {
            os: "linux".into(),
            os_version: "6.12".into(),
            cpu_brand: "AMD Ryzen 7 9700X".into(),
            cpu_cores_physical: 8,
            cpu_cores_logical: 16,
            ram_total_gb: 64.0,
            gpu_info: vec![],
            llama_cpp_version: None,
        };

        assert_eq!(
            hw.report_path_label(),
            format!(
                "{}-{}__CPU-AMD_Ryzen_7_9700X__GPU-none",
                "linux",
                std::env::consts::ARCH,
            ),
        );
    }
}
