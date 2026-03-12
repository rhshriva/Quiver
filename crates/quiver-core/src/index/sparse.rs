//! Sparse inverted index for hybrid dense+sparse search.
//!
//! Stores sparse vectors as `(index, value)` pairs. Search uses posting list
//! traversal with dot-product scoring (higher = more similar).
//!
//! # Intended use
//!
//! This index is not used standalone — it is paired with a dense vector index
//! inside [`Collection`][crate::collection::Collection] to support hybrid
//! dense+sparse search via weighted score fusion.
//!
//! # Memory
//!
//! Posting lists are inverted: for each term/dimension `d`, we store a list of
//! `(vector_id, weight)` pairs. This is very memory-efficient for typical
//! sparse embeddings (BM25, SPLADE, etc.) where vectors have ~50–200 non-zero
//! entries out of 30,000+ dimensions.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::VectorDbError;

// ── Sparse vector ────────────────────────────────────────────────────────────

/// A sparse vector represented as parallel arrays of indices and values.
///
/// Indices should be sorted for efficient merge-join operations, but this is
/// not strictly required — the search implementation handles unsorted input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    /// Dimension indices of non-zero entries.
    pub indices: Vec<u32>,
    /// Corresponding values (weights/scores).
    pub values: Vec<f32>,
}

impl SparseVector {
    /// Create a new sparse vector from parallel index/value arrays.
    ///
    /// # Panics
    /// Panics if `indices.len() != values.len()`.
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Self {
        assert_eq!(
            indices.len(),
            values.len(),
            "sparse vector: indices and values must have the same length"
        );
        Self { indices, values }
    }

    /// Create from a `HashMap<u32, f32>` (sorted by index).
    pub fn from_map(map: &HashMap<u32, f32>) -> Self {
        let mut pairs: Vec<(u32, f32)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        pairs.sort_by_key(|(k, _)| *k);
        let (indices, values): (Vec<u32>, Vec<f32>) = pairs.into_iter().unzip();
        Self { indices, values }
    }

    /// Dot product between two sparse vectors (merge-join on sorted indices).
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let mut score = 0.0f32;
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    score += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        score
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Is the vector empty?
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

// ── Sparse index ─────────────────────────────────────────────────────────────

/// A scored search result from the sparse index.
#[derive(Debug, Clone)]
pub struct SparseSearchResult {
    pub id: u64,
    /// Dot-product score (higher = more similar).
    pub score: f32,
}

/// An inverted index for sparse vectors.
///
/// Posting lists map each dimension index to a list of `(vector_id, value)`
/// pairs. A forward index maps each vector ID back to its sparse vector for
/// efficient deletion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseIndex {
    /// Inverted posting lists: dimension → [(id, weight), ...]
    posting_lists: HashMap<u32, Vec<(u64, f32)>>,
    /// Forward index: id → SparseVector (for deletion and iteration).
    forward: HashMap<u64, SparseVector>,
}

impl SparseIndex {
    /// Create an empty sparse index.
    pub fn new() -> Self {
        Self {
            posting_lists: HashMap::new(),
            forward: HashMap::new(),
        }
    }

    /// Add (or replace) a sparse vector for the given ID.
    pub fn upsert(&mut self, id: u64, vector: SparseVector) {
        // Remove old entries if this ID already exists.
        self.delete(id);

        // Add to posting lists.
        for (&dim, &val) in vector.indices.iter().zip(vector.values.iter()) {
            self.posting_lists
                .entry(dim)
                .or_insert_with(Vec::new)
                .push((id, val));
        }

        // Store in forward index.
        self.forward.insert(id, vector);
    }

    /// Delete a sparse vector by ID. Returns `true` if found.
    pub fn delete(&mut self, id: u64) -> bool {
        if let Some(vec) = self.forward.remove(&id) {
            for &dim in &vec.indices {
                if let Some(list) = self.posting_lists.get_mut(&dim) {
                    if let Some(pos) = list.iter().position(|(vid, _)| *vid == id) {
                        list.swap_remove(pos);
                    }
                    // Clean up empty posting lists.
                    if list.is_empty() {
                        self.posting_lists.remove(&dim);
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Search for the top-k vectors most similar to `query` by dot product.
    ///
    /// Returns results sorted by score descending (highest first).
    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<SparseSearchResult> {
        if k == 0 || query.is_empty() {
            return vec![];
        }

        // Accumulate scores via posting list traversal.
        let mut scores: HashMap<u64, f32> = HashMap::new();
        for (&dim, &q_val) in query.indices.iter().zip(query.values.iter()) {
            if let Some(list) = self.posting_lists.get(&dim) {
                for &(id, doc_val) in list {
                    *scores.entry(id).or_insert(0.0) += q_val * doc_val;
                }
            }
        }

        // Sort by score descending and take top-k.
        let mut results: Vec<SparseSearchResult> = scores
            .into_iter()
            .map(|(id, score)| SparseSearchResult { id, score })
            .collect();
        results.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.forward.len()
    }

    /// Is the index empty?
    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }

    /// Check if a vector with the given ID exists.
    pub fn contains(&self, id: u64) -> bool {
        self.forward.contains_key(&id)
    }

    /// Get the sparse vector for a given ID.
    pub fn get(&self, id: u64) -> Option<&SparseVector> {
        self.forward.get(&id)
    }

    /// Iterate over all (id, sparse_vector) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &SparseVector)> {
        self.forward.iter().map(|(&id, v)| (id, v))
    }

    // ── Persistence ─────────────────────────────────────────────────────────

    /// Serialize this index to a binary file (bincode format).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), VectorDbError> {
        let bytes = bincode::serialize(self)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Deserialize a previously saved index.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, VectorDbError> {
        let bytes = std::fs::read(path)?;
        bincode::deserialize(&bytes)
            .map_err(|e| VectorDbError::Serialization(e.to_string()))
    }
}

impl Default for SparseIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_vector_dot_product() {
        let a = SparseVector::new(vec![0, 2, 5], vec![1.0, 0.5, 0.3]);
        let b = SparseVector::new(vec![0, 3, 5], vec![2.0, 0.1, 0.4]);
        // Overlapping indices: 0 and 5
        // dot = 1.0*2.0 + 0.3*0.4 = 2.12
        let dot = a.dot(&b);
        assert!((dot - 2.12).abs() < 1e-5, "dot={dot}");
    }

    #[test]
    fn sparse_vector_dot_no_overlap() {
        let a = SparseVector::new(vec![0, 1], vec![1.0, 2.0]);
        let b = SparseVector::new(vec![2, 3], vec![3.0, 4.0]);
        assert_eq!(a.dot(&b), 0.0);
    }

    #[test]
    fn sparse_vector_from_map() {
        let mut map = HashMap::new();
        map.insert(5u32, 0.5f32);
        map.insert(1, 0.1);
        map.insert(10, 1.0);
        let sv = SparseVector::from_map(&map);
        assert_eq!(sv.indices, vec![1, 5, 10]); // sorted
        assert_eq!(sv.nnz(), 3);
    }

    #[test]
    fn sparse_index_upsert_and_search() {
        let mut idx = SparseIndex::new();
        idx.upsert(1, SparseVector::new(vec![0, 1, 2], vec![1.0, 0.5, 0.3]));
        idx.upsert(2, SparseVector::new(vec![0, 3], vec![0.2, 0.8]));
        idx.upsert(3, SparseVector::new(vec![1, 2, 3], vec![0.9, 0.1, 0.7]));

        let query = SparseVector::new(vec![0, 1], vec![1.0, 1.0]);
        let results = idx.search(&query, 2);
        assert_eq!(results.len(), 2);
        // vec1 score: 1.0*1.0 + 0.5*1.0 = 1.5
        // vec3 score: 0.9*1.0 = 0.9
        // vec2 score: 0.2*1.0 = 0.2
        assert_eq!(results[0].id, 1);
        assert!((results[0].score - 1.5).abs() < 1e-5);
        assert_eq!(results[1].id, 3);
    }

    #[test]
    fn sparse_index_delete() {
        let mut idx = SparseIndex::new();
        idx.upsert(1, SparseVector::new(vec![0, 1], vec![1.0, 0.5]));
        idx.upsert(2, SparseVector::new(vec![0, 2], vec![0.3, 0.8]));
        assert_eq!(idx.len(), 2);

        assert!(idx.delete(1));
        assert_eq!(idx.len(), 1);
        assert!(!idx.delete(1)); // already deleted

        let query = SparseVector::new(vec![0, 1], vec![1.0, 1.0]);
        let results = idx.search(&query, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 2);
    }

    #[test]
    fn sparse_index_upsert_replaces() {
        let mut idx = SparseIndex::new();
        idx.upsert(1, SparseVector::new(vec![0], vec![1.0]));
        idx.upsert(1, SparseVector::new(vec![1], vec![2.0]));
        assert_eq!(idx.len(), 1);

        let query = SparseVector::new(vec![0, 1], vec![1.0, 1.0]);
        let results = idx.search(&query, 1);
        assert_eq!(results[0].id, 1);
        assert!((results[0].score - 2.0).abs() < 1e-5); // should use new vector
    }

    #[test]
    fn sparse_index_empty_search() {
        let idx = SparseIndex::new();
        let query = SparseVector::new(vec![0], vec![1.0]);
        let results = idx.search(&query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn sparse_index_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sparse.bin");

        let mut idx = SparseIndex::new();
        idx.upsert(1, SparseVector::new(vec![0, 1], vec![1.0, 0.5]));
        idx.upsert(2, SparseVector::new(vec![1, 2], vec![0.3, 0.8]));
        idx.save(&path).unwrap();

        let loaded = SparseIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        let query = SparseVector::new(vec![0, 1], vec![1.0, 1.0]);
        let results = loaded.search(&query, 1);
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn sparse_index_iter() {
        let mut idx = SparseIndex::new();
        idx.upsert(10, SparseVector::new(vec![0], vec![1.0]));
        idx.upsert(20, SparseVector::new(vec![1], vec![2.0]));
        let mut ids: Vec<u64> = idx.iter().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, vec![10, 20]);
    }

    #[test]
    fn sparse_index_k_zero() {
        let mut idx = SparseIndex::new();
        idx.upsert(1, SparseVector::new(vec![0], vec![1.0]));
        let results = idx.search(&SparseVector::new(vec![0], vec![1.0]), 0);
        assert!(results.is_empty());
    }

    #[test]
    fn sparse_vector_empty_dot() {
        let a = SparseVector::new(vec![], vec![]);
        let b = SparseVector::new(vec![0], vec![1.0]);
        assert_eq!(a.dot(&b), 0.0);
        assert!(a.is_empty());
    }
}
