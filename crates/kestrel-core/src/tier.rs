//! Tiering configuration, level enumeration, and metrics counters for KV cache management.

use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Storage tiers for session KV cache and context state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TierLevel {
    /// In-memory active tier (GPU VRAM / live backend handle resident).
    Hot,
    /// In-memory snapshot tier (System RAM, backend handle evicted to release VRAM).
    Warm,
    /// Persistent disk storage tier.
    Cold,
}

/// Configuration policy governing tiering limits and memory budgets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierPolicy {
    /// Maximum number of active sessions resident in hot tier.
    pub max_hot_sessions: usize,
    /// Maximum number of sessions resident in warm tier.
    pub max_warm_sessions: usize,
    /// Maximum aggregate KV cache budget in bytes for hot tier.
    pub hot_budget_bytes: usize,
    /// Maximum aggregate KV cache budget in bytes for warm tier.
    pub warm_budget_bytes: usize,
}

impl Default for TierPolicy {
    fn default() -> Self {
        Self {
            max_hot_sessions: 16,
            max_warm_sessions: 64,
            hot_budget_bytes: 2048 * 1024 * 1024,  // 2 GiB
            warm_budget_bytes: 4096 * 1024 * 1024, // 4 GiB
        }
    }
}

impl TierPolicy {
    /// Creates a new tier policy with specified session limits and byte budgets.
    pub fn new(
        max_hot_sessions: usize,
        max_warm_sessions: usize,
        hot_budget_bytes: usize,
        warm_budget_bytes: usize,
    ) -> Self {
        Self {
            max_hot_sessions,
            max_warm_sessions,
            hot_budget_bytes,
            warm_budget_bytes,
        }
    }

    /// Checks whether adding `needed` bytes or another item violates hot tier limits.
    pub fn exceeds_hot(&self, count: usize, current_bytes: usize, needed: usize) -> bool {
        count >= self.max_hot_sessions || current_bytes + needed > self.hot_budget_bytes
    }

    /// Checks whether adding `needed` bytes or another item violates warm tier limits.
    pub fn exceeds_warm(&self, count: usize, current_bytes: usize, needed: usize) -> bool {
        count >= self.max_warm_sessions || current_bytes + needed > self.warm_budget_bytes
    }

    /// Creates a policy using default capacities and custom byte budgets.
    pub fn with_budgets_bytes(hot_budget_bytes: usize, warm_budget_bytes: usize) -> Self {
        Self {
            hot_budget_bytes,
            warm_budget_bytes,
            ..Default::default()
        }
    }

    /// Creates a policy using default capacities and byte budgets specified in megabytes.
    pub fn with_budgets_mb(hot_mb: usize, warm_mb: usize) -> Self {
        Self::with_budgets_bytes(hot_mb * 1024 * 1024, warm_mb * 1024 * 1024)
    }

    /// Constructs a policy reading environment overrides:
    /// - `KESTREL_HOT_BUDGET_MB`
    /// - `KESTREL_WARM_BUDGET_MB`
    pub fn from_env() -> Self {
        let mut policy = Self::default();

        if let Ok(val) = env::var("KESTREL_HOT_BUDGET_MB") {
            if let Ok(mb) = val.trim().parse::<usize>() {
                policy.hot_budget_bytes = mb * 1024 * 1024;
            }
        }

        if let Ok(val) = env::var("KESTREL_WARM_BUDGET_MB") {
            if let Ok(mb) = val.trim().parse::<usize>() {
                policy.warm_budget_bytes = mb * 1024 * 1024;
            }
        }

        policy
    }
}

/// Point-in-time snapshot of session tiering performance and utilization metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TierMetricsSnapshot {
    /// Requests satisfied directly by hot tier (VRAM).
    pub hot_hits: u64,
    /// Requests satisfied by warm tier (RAM).
    pub warm_hits: u64,
    /// Requests loaded from cold tier (disk).
    pub cold_hits: u64,
    /// Requests requiring creation of fresh sessions.
    pub misses: u64,
    /// Total number of session resolution requests.
    pub total_requests: u64,
    /// Overall cache hit rate (`(hot + warm + cold) / total_requests`).
    pub hit_rate: f64,
    /// Hot tier hit rate (`hot_hits / total_requests`).
    pub hot_hit_rate: f64,
    /// Promotions from warm tier into hot tier.
    pub promotions_warm_to_hot: u64,
    /// Promotions from cold tier into hot tier.
    pub promotions_cold_to_hot: u64,
    /// Evictions from hot tier into warm tier.
    pub demotions_hot_to_warm: u64,
    /// Evictions from warm tier into cold storage.
    pub demotions_warm_to_cold: u64,
    /// Current number of sessions in hot tier.
    pub hot_sessions: usize,
    /// Current number of sessions in warm tier.
    pub warm_sessions: usize,
    /// Total KV bytes currently used in hot tier.
    pub hot_used_bytes: usize,
    /// Total KV bytes currently used in warm tier.
    pub warm_used_bytes: usize,
    /// Configured maximum byte budget for hot tier.
    pub hot_budget_bytes: usize,
    /// Configured maximum byte budget for warm tier.
    pub warm_budget_bytes: usize,
    /// Continuous batching scheduler metrics, if active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_metrics: Option<crate::scheduler::BatchMetricsSnapshot>,
}

/// Thread-safe atomic counters tracking tiering events and transitions.
#[derive(Debug, Default, Clone)]
pub struct TierMetricsCounters {
    inner: Arc<TierMetricsCountersInner>,
}

#[derive(Debug, Default)]
struct TierMetricsCountersInner {
    hot_hits: AtomicU64,
    warm_hits: AtomicU64,
    cold_hits: AtomicU64,
    misses: AtomicU64,
    promotions_warm_to_hot: AtomicU64,
    promotions_cold_to_hot: AtomicU64,
    demotions_hot_to_warm: AtomicU64,
    demotions_warm_to_cold: AtomicU64,
}

impl TierMetricsCounters {
    /// Creates a new empty counters registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a hit in hot cache.
    pub fn record_hot_hit(&self) {
        self.inner.hot_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a hit in warm cache.
    pub fn record_warm_hit(&self) {
        self.inner.warm_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a hit in cold storage.
    pub fn record_cold_hit(&self) {
        self.inner.cold_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a cache miss (new session created).
    pub fn record_miss(&self) {
        self.inner.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a promotion from warm tier into hot tier.
    pub fn record_promotion_warm_to_hot(&self) {
        self.inner
            .promotions_warm_to_hot
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Records a promotion from cold tier into hot tier.
    pub fn record_promotion_cold_to_hot(&self) {
        self.inner
            .promotions_cold_to_hot
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Records a demotion from hot tier into warm tier.
    pub fn record_demotion_hot_to_warm(&self) {
        self.inner
            .demotions_hot_to_warm
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Records a demotion from warm tier into cold storage.
    pub fn record_demotion_warm_to_cold(&self) {
        self.inner
            .demotions_warm_to_cold
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Computes an aggregated point-in-time metrics snapshot.
    pub fn snapshot(
        &self,
        hot_sessions: usize,
        warm_sessions: usize,
        hot_used_bytes: usize,
        warm_used_bytes: usize,
        policy: &TierPolicy,
    ) -> TierMetricsSnapshot {
        let hot_hits = self.inner.hot_hits.load(Ordering::Relaxed);
        let warm_hits = self.inner.warm_hits.load(Ordering::Relaxed);
        let cold_hits = self.inner.cold_hits.load(Ordering::Relaxed);
        let misses = self.inner.misses.load(Ordering::Relaxed);
        let total_requests = hot_hits + warm_hits + cold_hits + misses;

        let hit_rate = if total_requests > 0 {
            (hot_hits + warm_hits + cold_hits) as f64 / total_requests as f64
        } else {
            0.0
        };

        let hot_hit_rate = if total_requests > 0 {
            hot_hits as f64 / total_requests as f64
        } else {
            0.0
        };

        TierMetricsSnapshot {
            hot_hits,
            warm_hits,
            cold_hits,
            misses,
            total_requests,
            hit_rate,
            hot_hit_rate,
            promotions_warm_to_hot: self.inner.promotions_warm_to_hot.load(Ordering::Relaxed),
            promotions_cold_to_hot: self.inner.promotions_cold_to_hot.load(Ordering::Relaxed),
            demotions_hot_to_warm: self.inner.demotions_hot_to_warm.load(Ordering::Relaxed),
            demotions_warm_to_cold: self.inner.demotions_warm_to_cold.load(Ordering::Relaxed),
            hot_sessions,
            warm_sessions,
            hot_used_bytes,
            warm_used_bytes,
            hot_budget_bytes: policy.hot_budget_bytes,
            warm_budget_bytes: policy.warm_budget_bytes,
            batch_metrics: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_policy_defaults_and_helpers() {
        let default_policy = TierPolicy::default();
        assert_eq!(default_policy.max_hot_sessions, 16);
        assert_eq!(default_policy.max_warm_sessions, 64);
        assert_eq!(default_policy.hot_budget_bytes, 2 * 1024 * 1024 * 1024);
        assert_eq!(default_policy.warm_budget_bytes, 4 * 1024 * 1024 * 1024);

        let custom_mb = TierPolicy::with_budgets_mb(512, 1024);
        assert_eq!(custom_mb.hot_budget_bytes, 512 * 1024 * 1024);
        assert_eq!(custom_mb.warm_budget_bytes, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_tier_metrics_snapshot_calculation() {
        let counters = TierMetricsCounters::new();
        counters.record_hot_hit();
        counters.record_hot_hit();
        counters.record_warm_hit();
        counters.record_cold_hit();
        counters.record_miss();

        let policy = TierPolicy::with_budgets_mb(100, 200);
        let snap = counters.snapshot(2, 5, 2048, 4096, &policy);

        assert_eq!(snap.total_requests, 5);
        assert_eq!(snap.hot_hits, 2);
        assert_eq!(snap.warm_hits, 1);
        assert_eq!(snap.cold_hits, 1);
        assert_eq!(snap.misses, 1);
        assert!((snap.hit_rate - 0.8).abs() < 1e-6);
        assert!((snap.hot_hit_rate - 0.4).abs() < 1e-6);
        assert_eq!(snap.hot_sessions, 2);
        assert_eq!(snap.warm_sessions, 5);
        assert_eq!(snap.hot_used_bytes, 2048);
        assert_eq!(snap.warm_used_bytes, 4096);
    }
}
