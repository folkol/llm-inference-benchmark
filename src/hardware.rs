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
