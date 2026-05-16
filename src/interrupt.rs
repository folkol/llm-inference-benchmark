/// Graceful interrupt handling for the benchmark loop.
///
/// On Windows, llama-cli (or its CUDA runtime) calls GenerateConsoleCtrlEvent
/// which broadcasts CTRL_C_EVENT to ALL processes sharing the console — including
/// llmb.exe.  Without a handler, Windows terminates llmb.exe with
/// STATUS_CONTROL_C_EXIT (0xc000013a).
///
/// We install a handler that:
///   • CTRL_C_EVENT   → sets STOP_REQUESTED flag, prints a notice, returns TRUE
///                       (we handled it — don't kill the process).
///   • CTRL_BREAK_EVENT → same: lets the current scenario finish, then exits.
///   • Other events   → returns FALSE (let the default handler deal with it).
///
/// The benchmark loop calls `requested()` after every scenario and exits early
/// if the flag is set, so the user can still abort with Ctrl+C or Ctrl+Break.

use std::sync::atomic::{AtomicBool, Ordering};

pub static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

pub fn install() {
    #[cfg(windows)]
    install_windows();
}

pub fn requested() -> bool {
    STOP_REQUESTED.load(Ordering::SeqCst)
}

#[cfg(windows)]
fn install_windows() {
    // We need SetConsoleCtrlHandler from kernel32.dll.
    // Use the same extern-link pattern used in doctor.rs.
    #[link(name = "kernel32")]
    extern "system" {
        fn SetConsoleCtrlHandler(
            handler_routine: unsafe extern "system" fn(u32) -> i32,
            add: i32,
        ) -> i32;
    }

    unsafe extern "system" fn handler(ctrl_type: u32) -> i32 {
        match ctrl_type {
            0 | 1 => {
                // CTRL_C_EVENT (0) or CTRL_BREAK_EVENT (1)
                if !STOP_REQUESTED.swap(true, Ordering::SeqCst) {
                    // First signal — ask to stop after the current scenario.
                    eprintln!("\n[interrupt] Finishing current scenario, then stopping...");
                    eprintln!("[interrupt] Press Ctrl+C again to abort immediately.");
                } else {
                    // Second signal — exit now.
                    std::process::exit(130);
                }
                1 // TRUE: we handled it, don't kill the process
            }
            _ => 0, // FALSE: pass to next handler (e.g. CTRL_CLOSE_EVENT)
        }
    }

    unsafe {
        SetConsoleCtrlHandler(handler, 1);
    }
}
