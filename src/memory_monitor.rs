//! Memory monitoring and OOM recovery for production resilience
//!
//! This module provides real-time memory monitoring and automatic recovery
//! strategies to handle out-of-memory conditions gracefully.
//!
//! ## Features
//!
//! - **Memory pressure detection**: Monitor GPU and system memory usage
//! - **Automatic recovery**: Execute recovery strategies when memory is low
//! - **Configurable thresholds**: Customize warning and critical levels
//! - **Background monitoring**: Non-blocking memory tracking
//!
//! ## Example
//!
//! ```rust,no_run
//! use mullama::memory_monitor::{MemoryMonitor, MemoryConfig, MemoryPressure};
//!
//! let config = MemoryConfig::default();
//! let monitor = MemoryMonitor::new(config);
//!
//! // Check current memory pressure
//! match monitor.pressure() {
//!     MemoryPressure::Normal => println!("Memory usage normal"),
//!     MemoryPressure::Warning => println!("Memory usage elevated"),
//!     MemoryPressure::Critical => println!("Memory critically low!"),
//!     MemoryPressure::Emergency => println!("Emergency! Immediate action needed"),
//! }
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Memory pressure levels indicating system memory state
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum MemoryPressure {
    /// Memory usage is normal (< warning threshold)
    Normal = 0,
    /// Memory usage is elevated (warning to critical threshold)
    Warning = 1,
    /// Memory usage is high (critical to emergency threshold)
    Critical = 2,
    /// Memory usage is emergency level (> emergency threshold)
    Emergency = 3,
}

impl MemoryPressure {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Normal,
            1 => Self::Warning,
            2 => Self::Critical,
            _ => Self::Emergency,
        }
    }
}

impl std::fmt::Display for MemoryPressure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal => write!(f, "normal"),
            Self::Warning => write!(f, "warning"),
            Self::Critical => write!(f, "critical"),
            Self::Emergency => write!(f, "emergency"),
        }
    }
}

/// Configuration for memory monitoring
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Threshold for warning level (0.0-1.0, default: 0.70)
    pub warning_threshold: f32,
    /// Threshold for critical level (0.0-1.0, default: 0.85)
    pub critical_threshold: f32,
    /// Threshold for emergency level (0.0-1.0, default: 0.95)
    pub emergency_threshold: f32,
    /// Interval between memory checks in milliseconds (default: 100)
    pub check_interval_ms: u64,
    /// Enable automatic recovery when memory pressure is detected
    pub enable_auto_recovery: bool,
    /// Monitor GPU memory (if available)
    pub monitor_gpu: bool,
    /// Monitor system memory
    pub monitor_system: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            warning_threshold: 0.70,
            critical_threshold: 0.85,
            emergency_threshold: 0.95,
            check_interval_ms: 100,
            enable_auto_recovery: true,
            monitor_gpu: true,
            monitor_system: true,
        }
    }
}

impl MemoryConfig {
    /// Create a config optimized for memory-constrained systems
    pub fn memory_constrained() -> Self {
        Self {
            warning_threshold: 0.60,
            critical_threshold: 0.75,
            emergency_threshold: 0.90,
            check_interval_ms: 50,
            enable_auto_recovery: true,
            monitor_gpu: true,
            monitor_system: true,
        }
    }

    /// Create a config optimized for high-memory systems
    pub fn high_memory() -> Self {
        Self {
            warning_threshold: 0.80,
            critical_threshold: 0.90,
            emergency_threshold: 0.97,
            check_interval_ms: 200,
            enable_auto_recovery: true,
            monitor_gpu: true,
            monitor_system: true,
        }
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<(), String> {
        if self.warning_threshold < 0.0 || self.warning_threshold > 1.0 {
            return Err("warning_threshold must be between 0.0 and 1.0".to_string());
        }
        if self.critical_threshold < 0.0 || self.critical_threshold > 1.0 {
            return Err("critical_threshold must be between 0.0 and 1.0".to_string());
        }
        if self.emergency_threshold < 0.0 || self.emergency_threshold > 1.0 {
            return Err("emergency_threshold must be between 0.0 and 1.0".to_string());
        }
        if self.warning_threshold >= self.critical_threshold {
            return Err("warning_threshold must be less than critical_threshold".to_string());
        }
        if self.critical_threshold >= self.emergency_threshold {
            return Err("critical_threshold must be less than emergency_threshold".to_string());
        }
        if self.check_interval_ms == 0 {
            return Err("check_interval_ms must be greater than 0".to_string());
        }
        Ok(())
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryStats {
    /// GPU memory used in bytes
    pub gpu_used: u64,
    /// GPU memory total in bytes
    pub gpu_total: u64,
    /// System memory used in bytes
    pub system_used: u64,
    /// System memory total in bytes
    pub system_total: u64,
}

impl MemoryStats {
    /// Get GPU memory usage ratio (0.0-1.0)
    pub fn gpu_usage(&self) -> f32 {
        if self.gpu_total == 0 {
            0.0
        } else {
            self.gpu_used as f32 / self.gpu_total as f32
        }
    }

    /// Get system memory usage ratio (0.0-1.0)
    pub fn system_usage(&self) -> f32 {
        if self.system_total == 0 {
            0.0
        } else {
            self.system_used as f32 / self.system_total as f32
        }
    }

    /// Get the maximum usage between GPU and system memory
    pub fn max_usage(&self) -> f32 {
        self.gpu_usage().max(self.system_usage())
    }

    /// Get GPU memory available in bytes
    pub fn gpu_available(&self) -> u64 {
        self.gpu_total.saturating_sub(self.gpu_used)
    }

    /// Get system memory available in bytes
    pub fn system_available(&self) -> u64 {
        self.system_total.saturating_sub(self.system_used)
    }
}

/// Real-time memory monitor
pub struct MemoryMonitor {
    config: MemoryConfig,
    current_pressure: AtomicU8,
    gpu_memory_used: AtomicU64,
    gpu_memory_total: AtomicU64,
    system_memory_used: AtomicU64,
    system_memory_total: AtomicU64,
    running: AtomicBool,
}

impl MemoryMonitor {
    /// Create a new memory monitor with the given configuration
    pub fn new(config: MemoryConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            current_pressure: AtomicU8::new(MemoryPressure::Normal as u8),
            gpu_memory_used: AtomicU64::new(0),
            gpu_memory_total: AtomicU64::new(0),
            system_memory_used: AtomicU64::new(0),
            system_memory_total: AtomicU64::new(0),
            running: AtomicBool::new(false),
        })
    }

    /// Create a monitor with default configuration
    pub fn with_defaults() -> Arc<Self> {
        Self::new(MemoryConfig::default())
    }

    /// Start background monitoring thread
    pub fn start(self: &Arc<Self>) -> JoinHandle<()> {
        let monitor = Arc::clone(self);
        self.running.store(true, Ordering::SeqCst);

        thread::spawn(move || {
            while monitor.running.load(Ordering::SeqCst) {
                monitor.update_stats();
                thread::sleep(Duration::from_millis(monitor.config.check_interval_ms));
            }
        })
    }

    /// Stop the background monitoring thread
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if the monitor is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get current memory pressure level
    pub fn pressure(&self) -> MemoryPressure {
        MemoryPressure::from_u8(self.current_pressure.load(Ordering::Relaxed))
    }

    /// Get current memory statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            gpu_used: self.gpu_memory_used.load(Ordering::Relaxed),
            gpu_total: self.gpu_memory_total.load(Ordering::Relaxed),
            system_used: self.system_memory_used.load(Ordering::Relaxed),
            system_total: self.system_memory_total.load(Ordering::Relaxed),
        }
    }

    /// Get GPU memory usage (used, total) in bytes
    pub fn gpu_memory(&self) -> (u64, u64) {
        (
            self.gpu_memory_used.load(Ordering::Relaxed),
            self.gpu_memory_total.load(Ordering::Relaxed),
        )
    }

    /// Get system memory usage (used, total) in bytes
    pub fn system_memory(&self) -> (u64, u64) {
        (
            self.system_memory_used.load(Ordering::Relaxed),
            self.system_memory_total.load(Ordering::Relaxed),
        )
    }

    /// Manually update memory statistics (also called by background thread)
    pub fn update_stats(&self) {
        // Update system memory
        if self.config.monitor_system {
            if let Some((used, total)) = get_system_memory() {
                self.system_memory_used.store(used, Ordering::Relaxed);
                self.system_memory_total.store(total, Ordering::Relaxed);
            }
        }

        // Update GPU memory
        if self.config.monitor_gpu {
            if let Some((used, total)) = get_gpu_memory() {
                self.gpu_memory_used.store(used, Ordering::Relaxed);
                self.gpu_memory_total.store(total, Ordering::Relaxed);
            }
        }

        // Calculate pressure level
        let stats = self.stats();
        let usage = stats.max_usage();
        let pressure = self.calculate_pressure(usage);
        self.current_pressure.store(pressure as u8, Ordering::Relaxed);
    }

    /// Calculate pressure level from usage ratio
    fn calculate_pressure(&self, usage: f32) -> MemoryPressure {
        if usage >= self.config.emergency_threshold {
            MemoryPressure::Emergency
        } else if usage >= self.config.critical_threshold {
            MemoryPressure::Critical
        } else if usage >= self.config.warning_threshold {
            MemoryPressure::Warning
        } else {
            MemoryPressure::Normal
        }
    }

    /// Check if recovery should be triggered
    pub fn should_recover(&self) -> bool {
        self.config.enable_auto_recovery && self.pressure() >= MemoryPressure::Critical
    }

    /// Get configuration
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }
}

impl Drop for MemoryMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Recovery strategy when memory pressure is detected
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Clear KV cache and retry
    ClearCache,
    /// Reduce context size by specified factor and retry
    ReduceContext {
        /// Factor to reduce context by (e.g., 0.5 = halve the context)
        factor: f32,
        /// Minimum context size to maintain
        min_size: u32,
    },
    /// Evict least-recently-used sequences
    EvictLRU {
        /// Number of sequences to keep
        keep_count: usize,
    },
    /// Shift context (remove oldest tokens)
    ShiftContext {
        /// Ratio of tokens to keep (e.g., 0.5 = keep newest half)
        keep_ratio: f32,
    },
    /// Abort the operation with an error
    Abort,
}

impl Default for RecoveryStrategy {
    fn default() -> Self {
        Self::ShiftContext { keep_ratio: 0.5 }
    }
}

/// Recovery action result
#[derive(Debug)]
pub enum RecoveryResult {
    /// Recovery successful, can continue
    Success,
    /// Recovery successful but context was reduced
    ContextReduced { new_size: u32 },
    /// Recovery successful but some sequences were evicted
    SequencesEvicted { evicted: usize },
    /// Recovery successful but context was shifted
    ContextShifted { tokens_removed: u32 },
    /// Recovery failed, must abort
    Failed { reason: String },
}

/// Manager for executing recovery strategies
pub struct RecoveryManager {
    /// Ordered list of strategies to try
    strategies: Vec<RecoveryStrategy>,
    /// Reference to memory monitor
    monitor: Option<Arc<MemoryMonitor>>,
    /// Maximum recovery attempts before giving up
    max_attempts: u32,
}

impl RecoveryManager {
    /// Create a new recovery manager with default strategies
    pub fn new() -> Self {
        Self {
            strategies: vec![
                RecoveryStrategy::ShiftContext { keep_ratio: 0.75 },
                RecoveryStrategy::ShiftContext { keep_ratio: 0.5 },
                RecoveryStrategy::ClearCache,
                RecoveryStrategy::Abort,
            ],
            monitor: None,
            max_attempts: 3,
        }
    }

    /// Create a recovery manager with custom strategies
    pub fn with_strategies(strategies: Vec<RecoveryStrategy>) -> Self {
        Self {
            strategies,
            monitor: None,
            max_attempts: 3,
        }
    }

    /// Attach a memory monitor
    pub fn with_monitor(mut self, monitor: Arc<MemoryMonitor>) -> Self {
        self.monitor = Some(monitor);
        self
    }

    /// Set maximum recovery attempts
    pub fn with_max_attempts(mut self, max: u32) -> Self {
        self.max_attempts = max;
        self
    }

    /// Get the configured strategies
    pub fn strategies(&self) -> &[RecoveryStrategy] {
        &self.strategies
    }

    /// Check if recovery is needed based on attached monitor
    pub fn needs_recovery(&self) -> bool {
        self.monitor
            .as_ref()
            .map(|m| m.should_recover())
            .unwrap_or(false)
    }

    /// Get current memory pressure from attached monitor
    pub fn pressure(&self) -> Option<MemoryPressure> {
        self.monitor.as_ref().map(|m| m.pressure())
    }

    /// Get the next strategy to try based on attempt number
    pub fn get_strategy(&self, attempt: usize) -> Option<&RecoveryStrategy> {
        self.strategies.get(attempt)
    }
}

impl Default for RecoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

// Platform-specific memory querying functions

/// Get system memory usage (used, total) in bytes
#[cfg(target_os = "linux")]
fn get_system_memory() -> Option<(u64, u64)> {
    use std::fs;

    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let mut total: u64 = 0;
    let mut available: u64 = 0;

    for line in meminfo.lines() {
        if line.starts_with("MemTotal:") {
            total = parse_meminfo_value(line)?;
        } else if line.starts_with("MemAvailable:") {
            available = parse_meminfo_value(line)?;
        }
    }

    if total > 0 {
        Some((total.saturating_sub(available), total))
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn parse_meminfo_value(line: &str) -> Option<u64> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 2 {
        // Value is in kB, convert to bytes
        parts[1].parse::<u64>().ok().map(|v| v * 1024)
    } else {
        None
    }
}

#[cfg(target_os = "macos")]
fn get_system_memory() -> Option<(u64, u64)> {
    use std::process::Command;

    // Get total memory using sysctl
    let total_output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let total_str = String::from_utf8_lossy(&total_output.stdout);
    let total: u64 = total_str.trim().parse().ok()?;

    // Get page size and free pages using vm_stat
    let vm_output = Command::new("vm_stat").output().ok()?;
    let vm_str = String::from_utf8_lossy(&vm_output.stdout);

    let mut free_pages: u64 = 0;
    let mut inactive_pages: u64 = 0;
    let page_size: u64 = 4096; // Standard macOS page size

    for line in vm_str.lines() {
        if line.starts_with("Pages free:") {
            free_pages = parse_vm_stat_value(line)?;
        } else if line.starts_with("Pages inactive:") {
            inactive_pages = parse_vm_stat_value(line)?;
        }
    }

    let available = (free_pages + inactive_pages) * page_size;
    Some((total.saturating_sub(available), total))
}

#[cfg(target_os = "macos")]
fn parse_vm_stat_value(line: &str) -> Option<u64> {
    let parts: Vec<&str> = line.split(':').collect();
    if parts.len() >= 2 {
        parts[1].trim().trim_end_matches('.').parse().ok()
    } else {
        None
    }
}

#[cfg(target_os = "windows")]
fn get_system_memory() -> Option<(u64, u64)> {
    use std::mem;

    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    let mut status = MemoryStatusEx {
        dw_length: mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };

    unsafe {
        if GlobalMemoryStatusEx(&mut status) != 0 {
            let used = status.ull_total_phys.saturating_sub(status.ull_avail_phys);
            Some((used, status.ull_total_phys))
        } else {
            None
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn get_system_memory() -> Option<(u64, u64)> {
    // Fallback: return None for unsupported platforms
    None
}

/// Get GPU memory usage (used, total) in bytes
/// Returns None if no GPU is available or query fails
fn get_gpu_memory() -> Option<(u64, u64)> {
    // Try CUDA first
    #[cfg(feature = "cuda")]
    {
        if let Some(result) = get_cuda_memory() {
            return Some(result);
        }
    }

    // Try Metal on macOS
    #[cfg(target_os = "macos")]
    {
        if let Some(result) = get_metal_memory() {
            return Some(result);
        }
    }

    // No GPU memory available
    None
}

#[cfg(feature = "cuda")]
fn get_cuda_memory() -> Option<(u64, u64)> {
    // CUDA memory query would go here
    // For now, return None as this requires CUDA runtime
    None
}

#[cfg(target_os = "macos")]
fn get_metal_memory() -> Option<(u64, u64)> {
    // Metal unified memory - return system memory as approximation
    // Real Metal memory tracking would require Metal framework bindings
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_config_validation() {
        let config = MemoryConfig::default();
        assert!(config.validate().is_ok());

        let bad_config = MemoryConfig {
            warning_threshold: 0.9,
            critical_threshold: 0.8,
            ..Default::default()
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_memory_pressure_ordering() {
        assert!(MemoryPressure::Normal < MemoryPressure::Warning);
        assert!(MemoryPressure::Warning < MemoryPressure::Critical);
        assert!(MemoryPressure::Critical < MemoryPressure::Emergency);
    }

    #[test]
    fn test_memory_stats() {
        let stats = MemoryStats {
            gpu_used: 4_000_000_000,
            gpu_total: 8_000_000_000,
            system_used: 8_000_000_000,
            system_total: 16_000_000_000,
        };

        assert!((stats.gpu_usage() - 0.5).abs() < 0.001);
        assert!((stats.system_usage() - 0.5).abs() < 0.001);
        assert_eq!(stats.gpu_available(), 4_000_000_000);
    }

    #[test]
    fn test_recovery_manager() {
        let manager = RecoveryManager::new();
        assert!(!manager.strategies().is_empty());
        assert!(manager.get_strategy(0).is_some());
        assert!(!manager.needs_recovery()); // No monitor attached
    }

    #[test]
    fn test_monitor_creation() {
        let monitor = MemoryMonitor::with_defaults();
        assert_eq!(monitor.pressure(), MemoryPressure::Normal);
        assert!(!monitor.is_running());
    }
}
