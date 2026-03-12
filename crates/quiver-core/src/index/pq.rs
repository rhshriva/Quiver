//! Product Quantization (PQ) codebook — sub-vector quantization for memory-efficient
//! approximate nearest-neighbour search.
//!
//! # How it works
//!
//! A `D`-dimensional vector is split into `m` sub-vectors of length `sub_dim = D / m`.
//! Each sub-vector is independently quantized to one of `k_sub` centroids (typically 256
//! so the code fits in a single `u8`).
//!
//! The result is an `m`-byte code per vector — a compression ratio of `4D / m` for
//! `f32` vectors (e.g. 96× for D=1536, m=64).
//!
//! # Asymmetric Distance Computation (ADC)
//!
//! At query time, a *distance table* is precomputed: `table[sub][code]` holds the
//! distance between the query's sub-vector and the corresponding codebook centroid.
//! The approximate distance to any encoded vector is then `sum(table[sub][code[sub]])`
//! — an O(m) operation instead of O(D).

use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::distance::Metric;

// ── Configuration ────────────────────────────────────────────────────────────

/// Product Quantization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqConfig {
    /// Number of sub-vectors (sub-quantizers). Must divide `dimensions`.
    /// Common values: 8, 16, 32, 64.
    pub m: usize,
    /// Number of centroids per sub-quantizer.
    /// Must be ≤ 256 so codes fit in a `u8`. Typically 256.
    pub k_sub: usize,
    /// Maximum k-means iterations for codebook training.
    pub max_iter: usize,
}

impl Default for PqConfig {
    fn default() -> Self {
        Self {
            m: 8,
            k_sub: 256,
            max_iter: 25,
        }
    }
}

// ── PQ code ──────────────────────────────────────────────────────────────────

/// A PQ-encoded vector: `m` bytes, one code per sub-quantizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqCode {
    pub codes: Vec<u8>, // length = m
}

// ── Codebook ─────────────────────────────────────────────────────────────────

/// Trained PQ codebook: `m` sub-quantizers, each with `k_sub` centroids.
///
/// Centroids are stored in a flattened layout:
/// `centroids[(sub * k_sub + centroid_idx) * sub_dim .. +sub_dim]`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqCodebook {
    pub m: usize,
    pub k_sub: usize,
    pub sub_dim: usize,
    /// Flattened centroid data: length = `m * k_sub * sub_dim`.
    pub centroids: Vec<f32>,
}

impl PqCodebook {
    /// Train the codebook from a set of training vectors using per-subspace k-means.
    ///
    /// # Panics
    ///
    /// Panics if `vectors` is empty or `dims % config.m != 0`.
    pub fn train(vectors: &[&[f32]], config: &PqConfig) -> Self {
        assert!(!vectors.is_empty(), "cannot train PQ on empty set");
        let dims = vectors[0].len();
        assert_eq!(
            dims % config.m, 0,
            "dimensions ({dims}) must be divisible by m ({})",
            config.m
        );
        let sub_dim = dims / config.m;
        let k_sub = config.k_sub.min(vectors.len()); // can't have more centroids than vectors

        let mut all_centroids = Vec::with_capacity(config.m * k_sub * sub_dim);

        for sub in 0..config.m {
            let offset = sub * sub_dim;

            // Extract sub-vectors for this sub-quantizer.
            let sub_vectors: Vec<&[f32]> = vectors
                .iter()
                .map(|v| &v[offset..offset + sub_dim])
                .collect();

            // Run k-means on the sub-vectors.
            let sub_centroids = kmeans_subvectors(&sub_vectors, k_sub, config.max_iter);

            // Pad with zeros if k_sub was clamped.
            for c in &sub_centroids {
                all_centroids.extend_from_slice(c);
            }
            for _ in sub_centroids.len()..config.k_sub {
                all_centroids.extend(std::iter::repeat(0.0f32).take(sub_dim));
            }
        }

        PqCodebook {
            m: config.m,
            k_sub: config.k_sub,
            sub_dim,
            centroids: all_centroids,
        }
    }

    /// Get a centroid slice for sub-quantizer `sub`, centroid index `k`.
    #[inline]
    fn centroid(&self, sub: usize, k: usize) -> &[f32] {
        let start = (sub * self.k_sub + k) * self.sub_dim;
        &self.centroids[start..start + self.sub_dim]
    }

    /// Encode a vector into PQ codes.
    pub fn encode(&self, vector: &[f32]) -> PqCode {
        let mut codes = Vec::with_capacity(self.m);
        for sub in 0..self.m {
            let sub_vec = &vector[sub * self.sub_dim..(sub + 1) * self.sub_dim];
            let mut best_code = 0u8;
            let mut best_dist = f32::MAX;
            for k in 0..self.k_sub {
                let centroid = self.centroid(sub, k);
                let dist = l2_squared(sub_vec, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_code = k as u8;
                }
            }
            codes.push(best_code);
        }
        PqCode { codes }
    }

    /// Decode a PQ code back to an approximate vector.
    pub fn decode(&self, code: &PqCode) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.m * self.sub_dim);
        for (sub, &c) in code.codes.iter().enumerate() {
            let centroid = self.centroid(sub, c as usize);
            vec.extend_from_slice(centroid);
        }
        vec
    }

    /// Precompute distance table for asymmetric distance computation (ADC).
    ///
    /// Returns a flattened table of size `m * k_sub`:
    /// `table[sub * k_sub + code] = squared_l2(query_sub, codebook_centroid)`.
    ///
    /// **Important:** ADC requires additive decomposition across sub-vectors.
    /// We always use squared L2 here because `||q-x||² = Σ ||q_sub - x_sub||²`.
    /// The final ADC distance is the sum of squared sub-distances. Callers
    /// that need Euclidean distance can take the `sqrt` of the result if needed.
    pub fn compute_distance_table(&self, query: &[f32], _metric: Metric) -> Vec<f32> {
        let mut table = Vec::with_capacity(self.m * self.k_sub);
        for sub in 0..self.m {
            let query_sub = &query[sub * self.sub_dim..(sub + 1) * self.sub_dim];
            for k in 0..self.k_sub {
                let centroid = self.centroid(sub, k);
                table.push(l2_squared(query_sub, centroid));
            }
        }
        table
    }

    /// Compute approximate distance using a precomputed distance table.
    /// This is O(m) instead of O(D) — the main speedup of PQ.
    #[inline]
    pub fn asymmetric_distance(&self, table: &[f32], code: &PqCode) -> f32 {
        code.codes
            .iter()
            .enumerate()
            .map(|(sub, &c)| table[sub * self.k_sub + c as usize])
            .sum()
    }
}

// ── K-means for sub-vectors ──────────────────────────────────────────────────

/// Simple L2 squared distance (no SIMD, sub-vectors are short).
#[inline]
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Lloyd's k-means on sub-vectors. Returns `k` centroids.
fn kmeans_subvectors(vectors: &[&[f32]], k: usize, max_iter: usize) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let dim = vectors[0].len();
    let k = k.min(n);

    // Random initialization.
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    let mut centroids: Vec<Vec<f32>> = indices[..k].iter().map(|&i| vectors[i].to_vec()).collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        let mut changed = false;

        // Assignment step.
        for (i, v) in vectors.iter().enumerate() {
            let best = centroids
                .iter()
                .enumerate()
                .map(|(ci, c)| (ci, l2_squared(v, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(ci, _)| ci)
                .unwrap_or(0);
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step.
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, v) in vectors.iter().enumerate() {
            let c = assignments[i];
            for (d, &x) in v.iter().enumerate() {
                sums[c][d] += x;
            }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                let cnt = counts[c] as f32;
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / cnt;
                }
            }
        }
    }

    centroids
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_and_encode_decode() {
        // 8-dim vectors, m=2 sub-quantizers, k_sub=4
        let config = PqConfig {
            m: 2,
            k_sub: 4,
            max_iter: 10,
        };
        let vecs: Vec<Vec<f32>> = (0..20)
            .map(|i| (0..8).map(|d| (i * 8 + d) as f32 * 0.01).collect())
            .collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

        let codebook = PqCodebook::train(&refs, &config);
        assert_eq!(codebook.m, 2);
        assert_eq!(codebook.sub_dim, 4);
        assert_eq!(codebook.centroids.len(), 2 * 4 * 4); // m * k_sub * sub_dim

        // Encode and decode should produce an approximation.
        let original = &vecs[5];
        let code = codebook.encode(original);
        assert_eq!(code.codes.len(), 2);

        let reconstructed = codebook.decode(&code);
        assert_eq!(reconstructed.len(), 8);

        // Reconstruction error should be bounded (not exact, but reasonable).
        let err: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        // With 4 centroids per subspace on 20 training vectors, error should be small.
        assert!(err < 1.0, "reconstruction error too large: {err}");
    }

    #[test]
    fn distance_table_matches_brute_force() {
        let config = PqConfig {
            m: 2,
            k_sub: 4,
            max_iter: 10,
        };
        let vecs: Vec<Vec<f32>> = (0..20)
            .map(|i| (0..8).map(|d| (i * 8 + d) as f32 * 0.01).collect())
            .collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let codebook = PqCodebook::train(&refs, &config);

        let query = &vecs[0];
        let code = codebook.encode(&vecs[5]);

        // ADC distance via table (uses squared L2 internally).
        let table = codebook.compute_distance_table(query, Metric::L2);
        let adc_dist = codebook.asymmetric_distance(&table, &code);

        // Brute-force squared L2 on reconstructed vector (must match ADC's space).
        let reconstructed = codebook.decode(&code);
        let bf_dist = l2_squared(query, &reconstructed);

        // They should be close (both approximate the same thing in squared-L2 space).
        assert!(
            (adc_dist - bf_dist).abs() < 0.1,
            "ADC={adc_dist} vs brute-force={bf_dist}"
        );
    }

    #[test]
    fn encode_produces_valid_codes() {
        let config = PqConfig {
            m: 4,
            k_sub: 8,
            max_iter: 5,
        };
        let vecs: Vec<Vec<f32>> = (0..30)
            .map(|i| (0..16).map(|d| (i * 16 + d) as f32 * 0.01).collect())
            .collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let codebook = PqCodebook::train(&refs, &config);

        for v in &vecs {
            let code = codebook.encode(v);
            assert_eq!(code.codes.len(), 4);
            for &c in &code.codes {
                assert!((c as usize) < 8, "code {c} exceeds k_sub=8");
            }
        }
    }

    #[test]
    fn small_training_set() {
        // When fewer vectors than k_sub, k_sub is clamped.
        let config = PqConfig {
            m: 2,
            k_sub: 256,
            max_iter: 5,
        };
        let vecs: Vec<Vec<f32>> = (0..5)
            .map(|i| (0..4).map(|d| (i * 4 + d) as f32).collect())
            .collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let codebook = PqCodebook::train(&refs, &config);
        // Should not panic; k_sub is internally clamped to 5.
        let code = codebook.encode(&vecs[0]);
        assert_eq!(code.codes.len(), 2);
    }
}
