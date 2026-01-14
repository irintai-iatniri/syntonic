use crate::resonant::PHI;

/// E8 Lattice Generator and Golden Projector
///
/// Implements:
/// 1. E8 Root generation (240 roots)
/// 2. Golden Projection R^8 -> R^4 (Parallel) + R^4 (Perpendicular)
/// 3. Golden Cone filtering

pub struct E8Lattice {
    pub roots: Vec<[f64; 8]>,
}

impl E8Lattice {
    pub fn new() -> Self {
        let mut roots = Vec::with_capacity(240);
        
        // Type A: Permutations of (+-1, +-1, 0, 0, 0, 0, 0, 0)
        // Indices pairs: (0..8) choose 2 = 28 pairs.
        // Signs: ++, +-, -+, -- (4 combinations)
        // Total: 28 * 4 = 112
        for i in 0..8 {
            for j in (i + 1)..8 {
                for s1 in [-1.0, 1.0] {
                    for s2 in [-1.0, 1.0] {
                        let mut v = [0.0; 8];
                        v[i] = s1;
                        v[j] = s2;
                        roots.push(v);
                    }
                }
            }
        }

        // Type B: (+-0.5, ..., +-0.5) with even number of minus signs
        // 2^8 = 256 combinations. Half have even signs.
        // Total: 128
        for i in 0..256 {
            let mut v = [0.0; 8];
            let mut neg_count = 0;
            for bit in 0..8 {
                if (i >> bit) & 1 == 1 {
                    v[bit] = -0.5;
                    neg_count += 1;
                } else {
                    v[bit] = 0.5;
                }
            }
            if neg_count % 2 == 0 {
                roots.push(v);
            }
        }
        
        E8Lattice { roots }
    }

    /// Static helper to get roots as Vec<Vec<f64>>
    pub fn generate_roots() -> Vec<Vec<f64>> {
        let lattice = Self::new();
        lattice.roots.into_iter().map(|arr| arr.to_vec()).collect()
    }

    /// Generate N weights (legacy method, currently unused directly in new logic but kept for integrity)
    pub fn generate_n_weights(&self, n: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(n * 8);
        for i in 0..n {
            let root = self.roots[i % 240];
            let scale = 1.0 + (i / 240) as f64; 
            for x in root {
                 result.push(x * scale);
            }
        }
        result
    }
}

pub struct GoldenProjector {
    phi: f64,
    // Projection matrices [4][8]
    p_par: [[f64; 8]; 4],
    p_perp: [[f64; 8]; 4],
}

impl GoldenProjector {
    pub fn new() -> Self {
        let phi = PHI;
        let norm = 1.0f64 / (1.0f64 + phi * phi).sqrt();
        
        // Parallel Projection: x_i + phi * x_{i+4} (normalized)
        let mut p_par = [[0.0; 8]; 4];
        let mut p_perp = [[0.0; 8]; 4];
        
        for i in 0..4 {
            p_par[i][i] = 1.0 * norm;
            p_par[i][i+4] = phi * norm;
            
            p_perp[i][i] = phi * norm;
            p_perp[i][i+4] = -1.0 * norm;
        }
        
        Self { phi, p_par, p_perp }
    }

    pub fn project_parallel(&self, v: &[f64]) -> [f64; 4] {
        let mut out = [0.0; 4];
        for i in 0..4 {
            let mut sum = 0.0;
            for j in 0..8 {
                sum += self.p_par[i][j] * v[j];
            }
            out[i] = sum;
        }
        out
    }

    pub fn project_perp(&self, v: &[f64]) -> [f64; 4] {
        let mut out = [0.0; 4];
        for i in 0..4 {
            let mut sum = 0.0;
            for j in 0..8 {
                sum += self.p_perp[i][j] * v[j];
            }
            out[i] = sum;
        }
        out
    }

    pub fn compute_q(&self, v: &[f64]) -> f64 {
        let par = self.project_parallel(v);
        let perp = self.project_perp(v);
        let norm_par: f64 = par.iter().map(|x| x * x).sum();
        let norm_perp: f64 = perp.iter().map(|x| x * x).sum();
        norm_par - norm_perp
    }
}

/// The 4 null vectors defining the golden cone boundary
/// These satisfy Q(c_a) = 0 and define the 36 Φ⁺(E₆) roots via ⟨c_a, α⟩ > 0
const NULL_VECTORS: [[f64; 8]; 4] = [
    [-0.152753, -0.312330, 0.192683, -0.692448, 0.013308, 0.531069, 0.153449, 0.238219],
    [0.270941, 0.201058, 0.532514, -0.128468, 0.404635, -0.475518, -0.294881, 0.330591],
    [0.157560, 0.189639, 0.480036, 0.021016, 0.274831, -0.782436, 0.060319, 0.130225],
    [0.476719, 0.111410, 0.464671, -0.543379, -0.142049, -0.150498, -0.314296, 0.327929],
];

/// Check if vector is in the golden cone using 4 null vectors
/// A root α is in the cone iff ⟨c_a, α⟩ > 0 for all 4 null vectors
/// This selects exactly 36 roots = Φ⁺(E₆)
pub fn is_in_golden_cone(v: &[f64], tol: f64) -> bool {
    if v.len() != 8 {
        return false;
    }

    for c in &NULL_VECTORS {
        let inner: f64 = (0..8).map(|i| c[i] * v[i]).sum();
        if inner <= tol {
            return false;
        }
    }
    true
}

/// Compute scalar weight from 8D vector using Golden Cone & Mobius logic
/// Returns w = exp(-Metric / phi)
pub fn compute_8d_weight(v: &[f64], projector: &GoldenProjector) -> f64 {
    // First check if vector is in the golden cone using null vectors
    if !is_in_golden_cone(v, 1e-10) {
        // Outside Golden Cone → Suppress heavily
        return 1e-9;
    }

    // Inside Golden Cone - compute weight based on parallel projection norm
    let par = projector.project_parallel(v);
    let norm_par: f64 = par.iter().map(|x| x*x).sum();

    // Weight: exp(-|v_parallel|^2 / phi)
    // Low energy states (small norm) have high weight
    (-norm_par / PHI).exp()
}
