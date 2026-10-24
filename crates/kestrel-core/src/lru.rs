//! In-memory Least Recently Used (LRU) cache implementation.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

/// An in-memory Least Recently Used (LRU) cache with a fixed maximum capacity.
///
/// When the cache exceeds its capacity, inserting a new key evicts the least recently
/// accessed entry, returning both the key and the value so callers can persist it.
#[derive(Debug, Clone)]
pub struct LruCache<K, V> {
    capacity: usize,
    map: HashMap<K, V>,
    order: VecDeque<K>,
}

impl<K: Hash + Eq + Clone, V> LruCache<K, V> {
    /// Creates a new LRU cache with the specified capacity.
    ///
    /// # Panics
    /// Panics if `capacity == 0`.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "LruCache capacity must be greater than zero");
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
        }
    }

    /// Returns the maximum capacity of the cache.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of items currently in the cache.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if the cache contains no items.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns true if the cache contains the specified key without promoting it.
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Returns a reference to the value corresponding to the key and marks it as most recently used.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.map.contains_key(key) {
            self.promote(key);
            self.map.get(key)
        } else {
            None
        }
    }

    /// Returns a mutable reference to the value corresponding to the key and marks it as most recently used.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        if self.map.contains_key(key) {
            self.promote(key);
            self.map.get_mut(key)
        } else {
            None
        }
    }

    /// Returns a reference to the value without marking it as most recently used.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    /// Inserts a key-value pair into the cache.
    ///
    /// If the key already exists, its value is updated and moved to the most recently used position.
    /// If the key did not exist and the cache is at capacity, the least recently used entry is evicted
    /// and returned as `Some((evicted_key, evicted_value))`.
    pub fn put(&mut self, key: K, value: V) -> Option<(K, V)> {
        if self.map.contains_key(&key) {
            self.promote(&key);
            self.map.insert(key, value);
            None
        } else {
            let evicted = if self.map.len() >= self.capacity {
                self.order.pop_front().and_then(|oldest_key| {
                    self.map.remove(&oldest_key).map(|val| (oldest_key, val))
                })
            } else {
                None
            };

            self.order.push_back(key.clone());
            self.map.insert(key, value);
            evicted
        }
    }

    /// Removes a key from the cache, returning the value if the key was present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(val) = self.map.remove(key) {
            self.order.retain(|k| k != key);
            Some(val)
        } else {
            None
        }
    }

    /// Removes and returns the least recently used key-value pair from the cache.
    pub fn pop_lru(&mut self) -> Option<(K, V)> {
        self.order
            .pop_front()
            .and_then(|oldest_key| self.map.remove(&oldest_key).map(|val| (oldest_key, val)))
    }

    /// Returns an iterator visiting all values in arbitrary order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.map.values()
    }

    /// Clears all entries from the cache.
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    fn promote(&mut self, key: &K) {
        if let Some(idx) = self.order.iter().position(|k| k == key) {
            self.order.remove(idx);
            self.order.push_back(key.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_basic_put_and_get() {
        let mut cache = LruCache::new(2);
        assert!(cache.is_empty());
        assert_eq!(cache.capacity(), 2);

        assert_eq!(cache.put("a", 1), None);
        assert_eq!(cache.put("b", 2), None);
        assert_eq!(cache.len(), 2);

        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.get(&"c"), None);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = LruCache::new(2);
        assert_eq!(cache.put("a", 1), None);
        assert_eq!(cache.put("b", 2), None);

        // Access "a" to make "b" the least recently used
        assert_eq!(cache.get(&"a"), Some(&1));

        // Inserting "c" should evict "b"
        let evicted = cache.put("c", 3);
        assert_eq!(evicted, Some(("b", 2)));

        assert_eq!(cache.get(&"b"), None);
        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"c"), Some(&3));
    }

    #[test]
    fn test_lru_update_existing_key() {
        let mut cache = LruCache::new(2);
        assert_eq!(cache.put("a", 1), None);
        assert_eq!(cache.put("b", 2), None);

        // Update "a", shouldn't evict anything
        assert_eq!(cache.put("a", 10), None);
        assert_eq!(cache.len(), 2);

        // "a" was updated so "b" is now LRU
        let evicted = cache.put("c", 3);
        assert_eq!(evicted, Some(("b", 2)));
        assert_eq!(cache.get(&"a"), Some(&10));
    }

    #[test]
    fn test_lru_remove() {
        let mut cache = LruCache::new(2);
        cache.put("a", 1);
        cache.put("b", 2);

        assert_eq!(cache.remove(&"a"), Some(1));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&"a"), None);
        assert_eq!(cache.get(&"b"), Some(&2));
    }

    #[test]
    fn test_lru_peek_does_not_promote() {
        let mut cache = LruCache::new(2);
        cache.put("a", 1);
        cache.put("b", 2);

        // Peek "a" without promoting
        assert_eq!(cache.peek(&"a"), Some(&1));

        // "a" was not promoted, so inserting "c" evicts "a"
        let evicted = cache.put("c", 3);
        assert_eq!(evicted, Some(("a", 1)));
        assert_eq!(cache.get(&"a"), None);
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.get(&"c"), Some(&3));
    }

    #[test]
    fn test_lru_pop_lru_and_values() {
        let mut cache = LruCache::new(3);
        cache.put("a", 10);
        cache.put("b", 20);
        cache.put("c", 30);

        // Sum of values
        let sum: i32 = cache.values().sum();
        assert_eq!(sum, 60);

        // Access "a" so "b" becomes the oldest
        assert_eq!(cache.get(&"a"), Some(&10));

        let popped = cache.pop_lru();
        assert_eq!(popped, Some(("b", 20)));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&"b"), None);

        let popped_next = cache.pop_lru();
        assert_eq!(popped_next, Some(("c", 30)));

        let popped_last = cache.pop_lru();
        assert_eq!(popped_last, Some(("a", 10)));

        assert_eq!(cache.pop_lru(), None);
        assert!(cache.is_empty());
    }
}
