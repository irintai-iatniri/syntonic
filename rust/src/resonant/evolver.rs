//! Resonant Evolution Strategy (RES) for discrete lattice-based optimization.
//!
//! The RES is a population-based evolution algorithm that exploits the insight that
//! lattice syntony (geometric beauty) correlates with task fitness. Key principle:
//! 80% of candidates die cheaply on CPU (lattice syntony check); only geometrically
//! promising candidates get expensive GPU evaluation.
//!
//! # Algorithm Overview
//!
//! 4-Step Generation Loop:
//! 1. **Spawn mutants** (CPU): Generate population_size mutants via lattice perturbation
//! 2. **Filter by lattice syntony** (CPU): Keep top survivor_count (~25%)
//! 3. **GPU evaluation** (GPU→CPU): D-phase + crystallize survivors only
//! 4. **Select winner** (CPU): score = fitness + λ × flux_syntony
//!
//! # Example
//!
//! ```ignore
//! use syntonic::resonant::{ResonantEvolver, RESConfig, ResonantTensor};
//!
//! let config = RESConfig::default();
//! let template = ResonantTensor::from_floats(&[1.0, 2.0, 3.0, 4.0], vec![4], 100)?;
//! let mut evolver = ResonantEvolver::from_template(&template, config);
//!
//! // Run evolution
//! for _ in 0..100 {
//!     let best_syntony = evolver.step()?;
//!     println!("Best syntony: {:.4}", best_syntony);
//! }
//!
//! let result = evolver.result();
//! ```

use pyo3::prelude::*;
use rand::Rng;
use std::cmp::Ordering;

use crate::exact::GoldenExact;
use super::tensor::{ResonantTensor, ResonantError};
use super::attractor::AttractorMemory;
use super::retrocausal::harmonize_with_attractor_pull;
use super::PHI;

/// Universal syntony deficit q - NOT a hyperparameter!
/// This is a fundamental constant from SRT.
pub const Q_DEFICIT: f64 = 0.027395146920;

/// Configuration for the Resonant Evolution Strategy.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RESConfig {
    /// Population size (number of mutants per generation)
    #[pyo3(get, set)]
    pub population_size: usize,

    /// Number of survivors after lattice syntony filtering (~25% of population)
    #[pyo3(get, set)]
    pub survivor_count: usize,

    /// Syntony weight in scoring: score = fitness + lambda * syntony
    /// Default: Q_DEFICIT = 0.027395146920 (NOT a hyperparameter!)
    #[pyo3(get, set)]
    pub lambda_val: f64,

    /// Scale of mutations in Q(φ) space
    #[pyo3(get, set)]
    pub mutation_scale: f64,

    /// Precision for crystallization (max coefficient)
    #[pyo3(get, set)]
    pub precision: i64,

    /// Noise scale for D-phase
    #[pyo3(get, set)]
    pub noise_scale: f64,

    /// Maximum generations before stopping
    #[pyo3(get, set)]
    pub max_generations: usize,

    /// Convergence threshold (stop if improvement < threshold)
    #[pyo3(get, set)]
    pub convergence_threshold: f64,

    // === Retrocausal RES Parameters ===
    /// Enable retrocausal attractor-guided harmonization
    #[pyo3(get, set)]
    pub enable_retrocausal: bool,

    /// Maximum number of attractors to store
    #[pyo3(get, set)]
    pub attractor_capacity: usize,

    /// Retrocausal pull strength (0.0 = disabled, 1.0 = full pull)
    #[pyo3(get, set)]
    pub attractor_pull_strength: f64,

    /// Minimum syntony threshold for attractor storage
    #[pyo3(get, set)]
    pub attractor_min_syntony: f64,

    /// Temporal decay rate for attractors (per generation)
    #[pyo3(get, set)]
    pub attractor_decay_rate: f64,
}

impl Default for RESConfig {
    fn default() -> Self {
        RESConfig {
            population_size: 64,
            survivor_count: 16,
            lambda_val: Q_DEFICIT,
            mutation_scale: 0.1,
            precision: 100,
            noise_scale: 0.01,
            max_generations: 1000,
            convergence_threshold: 1e-6,
            // Retrocausal defaults (opt-in)
            enable_retrocausal: false,
            attractor_capacity: 32,
            attractor_pull_strength: 0.3,
            attractor_min_syntony: 0.7,
            attractor_decay_rate: 0.98,
        }
    }
}

#[pymethods]
impl RESConfig {
    /// Create a new RESConfig with default values.
    #[new]
    #[pyo3(signature = (
        population_size=64,
        survivor_count=16,
        lambda_val=None,
        mutation_scale=0.1,
        precision=100,
        noise_scale=0.01,
        max_generations=1000,
        convergence_threshold=1e-6,
        enable_retrocausal=false,
        attractor_capacity=32,
        attractor_pull_strength=0.3,
        attractor_min_syntony=0.7,
        attractor_decay_rate=0.98
    ))]
    fn py_new(
        population_size: usize,
        survivor_count: usize,
        lambda_val: Option<f64>,
        mutation_scale: f64,
        precision: i64,
        noise_scale: f64,
        max_generations: usize,
        convergence_threshold: f64,
        enable_retrocausal: bool,
        attractor_capacity: usize,
        attractor_pull_strength: f64,
        attractor_min_syntony: f64,
        attractor_decay_rate: f64,
    ) -> Self {
        RESConfig {
            population_size,
            survivor_count,
            lambda_val: lambda_val.unwrap_or(Q_DEFICIT),
            mutation_scale,
            precision,
            noise_scale,
            max_generations,
            convergence_threshold,
            enable_retrocausal,
            attractor_capacity,
            attractor_pull_strength,
            attractor_min_syntony,
            attractor_decay_rate,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RESConfig(pop={}, survivors={}, λ={:.6}, mut_scale={}, prec={})",
            self.population_size, self.survivor_count, self.lambda_val,
            self.mutation_scale, self.precision
        )
    }
}

/// Result of a RES evolution run.
#[pyclass]
#[derive(Clone)]
pub struct RESResult {
    /// The winning tensor with highest syntony
    #[pyo3(get)]
    pub winner: ResonantTensor,

    /// Final syntony of the winner
    #[pyo3(get)]
    pub final_syntony: f64,

    /// Number of generations run
    #[pyo3(get)]
    pub generations: usize,

    /// Whether the evolution converged
    #[pyo3(get)]
    pub converged: bool,

    /// History of best syntony per generation
    #[pyo3(get)]
    pub syntony_history: Vec<f64>,
}

#[pymethods]
impl RESResult {
    fn __repr__(&self) -> String {
        format!(
            "RESResult(syntony={:.4}, generations={}, converged={})",
            self.final_syntony, self.generations, self.converged
        )
    }
}

/// Resonant Evolution Strategy (RES) evolver.
///
/// A population-based evolution algorithm that uses lattice syntony
/// as a cheap proxy for fitness, filtering 75-80% of candidates on CPU
/// before expensive GPU evaluation.
#[pyclass]
pub struct ResonantEvolver {
    /// Configuration
    config: RESConfig,

    /// Current best tensor (parent for next generation)
    best_tensor: Option<ResonantTensor>,

    /// Best syntony achieved so far
    best_syntony: f64,

    /// Current generation number
    generation: usize,

    /// History of best syntony per generation
    syntony_history: Vec<f64>,

    /// Template tensor (for shape and mode norms)
    template: Option<ResonantTensor>,

    /// Convergence window for checking stagnation
    convergence_window: Vec<f64>,

    /// Attractor memory for retrocausal influence
    attractor_memory: AttractorMemory,
}

impl ResonantEvolver {
    /// Create a new evolver with configuration.
    pub fn new(config: RESConfig) -> Self {
        let attractor_memory = AttractorMemory::new(
            config.attractor_capacity,
            config.attractor_min_syntony,
            config.attractor_decay_rate,
        );

        ResonantEvolver {
            config,
            best_tensor: None,
            best_syntony: 0.0,
            generation: 0,
            syntony_history: Vec::new(),
            template: None,
            convergence_window: Vec::new(),
            attractor_memory,
        }
    }

    /// Create an evolver from a template tensor.
    pub fn from_template(template: &ResonantTensor, config: RESConfig) -> Self {
        let syntony = template.syntony();
        let attractor_memory = AttractorMemory::new(
            config.attractor_capacity,
            config.attractor_min_syntony,
            config.attractor_decay_rate,
        );

        ResonantEvolver {
            config,
            best_tensor: Some(template.clone()),
            best_syntony: syntony,
            generation: 0,
            syntony_history: vec![syntony],
            template: Some(template.clone()),
            convergence_window: vec![syntony],
            attractor_memory,
        }
    }

    /// Spawn mutants from a parent tensor.
    ///
    /// Each mutant is created by adding random perturbations in Q(φ) space
    /// to each lattice element.
    pub fn spawn_mutants(&self, parent: &ResonantTensor) -> Vec<ResonantTensor> {
        let mut rng = rand::thread_rng();
        let mut mutants = Vec::with_capacity(self.config.population_size);

        for _ in 0..self.config.population_size {
            let mutant = self.mutate_lattice(parent, &mut rng);
            mutants.push(mutant);
        }

        mutants
    }

    /// Mutate a single tensor's lattice.
    fn mutate_lattice(&self, parent: &ResonantTensor, rng: &mut impl Rng) -> ResonantTensor {
        let lattice = parent.lattice();
        let mode_norm_sq = parent.mode_norm_sq();
        let shape = parent.shape().to_vec();

        let mutated: Vec<GoldenExact> = lattice
            .iter()
            .map(|g| {
                // Random perturbation in Q(φ) space
                // delta = delta_a + delta_b * φ
                let delta_a = (rng.gen::<f64>() - 0.5) * self.config.mutation_scale;
                let delta_b = (rng.gen::<f64>() - 0.5) * self.config.mutation_scale;

                // Snap perturbation to golden lattice
                let perturbation = GoldenExact::find_nearest(
                    delta_a + delta_b * PHI,
                    self.config.precision
                );

                // Add perturbation to original
                *g + perturbation
            })
            .collect();

        // Create new tensor from mutated lattice
        ResonantTensor::from_lattice(mutated, shape, mode_norm_sq.to_vec())
            .unwrap_or_else(|_| parent.clone())
    }

    /// Filter candidates by lattice syntony (CPU, cheap).
    ///
    /// Returns the top `survivor_count` candidates sorted by descending syntony.
    pub fn filter_by_lattice_syntony(&self, candidates: Vec<ResonantTensor>) -> Vec<ResonantTensor> {
        let mut scored: Vec<(ResonantTensor, f64)> = candidates
            .into_iter()
            .map(|t| {
                let syntony = t.syntony();
                (t, syntony)
            })
            .collect();

        // Sort by syntony descending
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        });

        // Take top survivors
        scored
            .into_iter()
            .take(self.config.survivor_count)
            .map(|(t, _)| t)
            .collect()
    }

    /// Evaluate survivors using CPU D-phase cycle.
    ///
    /// For each survivor:
    /// 1. Wake flux with noise (simulated D-phase)
    /// 2. Crystallize back to lattice
    /// 3. Compute final syntony
    ///
    /// Returns survivors with their scores (syntony after D→H cycle).
    pub fn evaluate_survivors_cpu(&self, survivors: Vec<ResonantTensor>) -> Vec<(ResonantTensor, f64)> {
        survivors
            .into_iter()
            .map(|mut tensor| {
                // CPU D→H cycle
                let syntony = tensor.run_cpu_cycle(self.config.noise_scale, self.config.precision)
                    .unwrap_or(tensor.syntony());
                (tensor, syntony)
            })
            .collect()
    }

    /// Apply retrocausal harmonization to survivors.
    ///
    /// Uses attractor memory to bias harmonization toward proven high-syntony states.
    fn apply_retrocausal_harmonization(
        &self,
        survivors: Vec<ResonantTensor>
    ) -> Result<Vec<ResonantTensor>, ResonantError> {
        survivors
            .into_iter()
            .map(|mut t| {
                harmonize_with_attractor_pull(
                    &mut t,
                    &self.attractor_memory,
                    self.config.attractor_pull_strength
                )?;
                Ok(t)
            })
            .collect()
    }

    /// Select the winner from evaluated candidates.
    ///
    /// Winner is selected by: score = syntony (since we don't have external fitness yet)
    pub fn select_winner(&self, evaluated: Vec<(ResonantTensor, f64)>) -> Option<ResonantTensor> {
        evaluated
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .map(|(t, _)| t)
    }

    /// Select the winner from evaluated candidates, returning both tensor and score.
    ///
    /// Winner is selected by: score = syntony (since we don't have external fitness yet)
    fn select_winner_with_score(&self, evaluated: &[(ResonantTensor, f64)]) -> Option<(ResonantTensor, f64)> {
        evaluated
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .map(|(t, s)| (t.clone(), *s))
    }

    /// Run one generation of evolution.
    ///
    /// Returns the best syntony achieved this generation.
    pub fn step(&mut self) -> Result<f64, ResonantError> {
        // Get parent (or create random if none)
        let parent = match &self.best_tensor {
            Some(t) => t.clone(),
            None => return Err(ResonantError::InvalidPhaseTransition(
                "No parent tensor set. Use from_template() to initialize.".to_string()
            )),
        };

        // Step 1: Spawn mutants (CPU)
        let mutants = self.spawn_mutants(&parent);

        // Step 2: Filter by lattice syntony (CPU, cheap)
        let mut survivors = self.filter_by_lattice_syntony(mutants);

        // Step 3: RETROCAUSAL HARMONIZATION (if enabled)
        if self.config.enable_retrocausal && !self.attractor_memory.is_empty() {
            survivors = self.apply_retrocausal_harmonization(survivors)?;
        }

        // Step 4: Evaluate survivors (CPU D→H cycle)
        let evaluated = self.evaluate_survivors_cpu(survivors);

        // Step 5: Select winner and update attractors
        if let Some((winner, _score)) = self.select_winner_with_score(&evaluated) {
            let winner_syntony = winner.syntony();

            // Store high-syntony candidates as attractors
            if self.config.enable_retrocausal {
                for (tensor, _) in &evaluated {
                    let syntony = tensor.syntony();
                    self.attractor_memory.maybe_add(tensor, syntony, self.generation);
                }
            }

            // Update best if improved
            if winner_syntony > self.best_syntony {
                self.best_syntony = winner_syntony;
                self.best_tensor = Some(winner);
            }
        }

        // Apply temporal decay to attractors
        if self.config.enable_retrocausal {
            self.attractor_memory.apply_decay();
        }

        // Update tracking
        self.generation += 1;
        self.syntony_history.push(self.best_syntony);
        self.update_convergence_window(self.best_syntony);

        Ok(self.best_syntony)
    }

    /// Update convergence tracking window.
    fn update_convergence_window(&mut self, syntony: f64) {
        self.convergence_window.push(syntony);
        if self.convergence_window.len() > 10 {
            self.convergence_window.remove(0);
        }
    }

    /// Check if evolution has converged.
    pub fn is_converged(&self) -> bool {
        if self.convergence_window.len() < 10 {
            return false;
        }

        let min = self.convergence_window.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.convergence_window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        (max - min) < self.config.convergence_threshold
    }

    /// Run evolution until convergence or max generations.
    pub fn run(&mut self) -> Result<RESResult, ResonantError> {
        while self.generation < self.config.max_generations && !self.is_converged() {
            self.step()?;
        }

        let winner = self.best_tensor.clone()
            .ok_or_else(|| ResonantError::InvalidPhaseTransition(
                "No winner found after evolution".to_string()
            ))?;

        Ok(RESResult {
            winner,
            final_syntony: self.best_syntony,
            generations: self.generation,
            converged: self.is_converged(),
            syntony_history: self.syntony_history.clone(),
        })
    }

    /// Get the current best tensor.
    pub fn best(&self) -> Option<&ResonantTensor> {
        self.best_tensor.as_ref()
    }

    /// Get the current generation number.
    pub fn current_generation(&self) -> usize {
        self.generation
    }

    /// Get the best syntony achieved.
    pub fn best_syntony(&self) -> f64 {
        self.best_syntony
    }
}

// =========================================================================
// PyO3 Methods
// =========================================================================

#[pymethods]
impl ResonantEvolver {
    /// Create a new evolver from a template tensor.
    #[new]
    #[pyo3(signature = (template, config=None))]
    fn py_new(template: &ResonantTensor, config: Option<RESConfig>) -> Self {
        let cfg = config.unwrap_or_default();
        ResonantEvolver::from_template(template, cfg)
    }

    /// Run one generation. Returns best syntony.
    #[pyo3(name = "step")]
    fn py_step(&mut self) -> PyResult<f64> {
        self.step().map_err(|e| e.into())
    }

    /// Run evolution until convergence. Returns RESResult.
    #[pyo3(name = "run")]
    fn py_run(&mut self) -> PyResult<RESResult> {
        self.run().map_err(|e| e.into())
    }

    /// Get the current best tensor.
    #[getter]
    fn get_best(&self) -> Option<ResonantTensor> {
        self.best_tensor.clone()
    }

    /// Get the current generation number.
    #[getter]
    fn get_generation(&self) -> usize {
        self.generation
    }

    /// Get the best syntony achieved.
    #[getter]
    fn get_best_syntony(&self) -> f64 {
        self.best_syntony
    }

    /// Check if evolution has converged.
    #[getter]
    fn get_is_converged(&self) -> bool {
        self.is_converged()
    }

    /// Get the syntony history.
    #[getter]
    fn get_syntony_history(&self) -> Vec<f64> {
        self.syntony_history.clone()
    }

    /// Get the configuration.
    #[getter]
    fn get_config(&self) -> RESConfig {
        self.config.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ResonantEvolver(generation={}, best_syntony={:.4}, converged={})",
            self.generation, self.best_syntony, self.is_converged()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = RESConfig::default();
        assert_eq!(config.population_size, 64);
        assert_eq!(config.survivor_count, 16);
        assert!((config.lambda_val - Q_DEFICIT).abs() < 1e-10);
    }

    #[test]
    fn test_evolver_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mode_norms = vec![0.0, 1.0, 4.0, 9.0];
        let tensor = ResonantTensor::from_floats(&data, vec![4], mode_norms, 100).unwrap();

        let evolver = ResonantEvolver::from_template(&tensor, RESConfig::default());
        assert_eq!(evolver.current_generation(), 0);
        assert!(evolver.best().is_some());
    }

    #[test]
    fn test_spawn_mutants() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mode_norms = vec![0.0, 1.0, 4.0, 9.0];
        let tensor = ResonantTensor::from_floats(&data, vec![4], mode_norms, 100).unwrap();

        let evolver = ResonantEvolver::from_template(&tensor, RESConfig::default());
        let mutants = evolver.spawn_mutants(&tensor);

        assert_eq!(mutants.len(), 64);
    }

    #[test]
    fn test_filter_by_syntony() {
        let config = RESConfig {
            population_size: 10,
            survivor_count: 3,
            ..Default::default()
        };

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mode_norms = vec![0.0, 1.0, 4.0, 9.0];
        let tensor = ResonantTensor::from_floats(&data, vec![4], mode_norms, 100).unwrap();

        let evolver = ResonantEvolver::from_template(&tensor, config);
        let mutants = evolver.spawn_mutants(&tensor);
        let survivors = evolver.filter_by_lattice_syntony(mutants);

        assert_eq!(survivors.len(), 3);
    }

    #[test]
    fn test_single_step() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mode_norms = vec![0.0, 1.0, 4.0, 9.0];
        let tensor = ResonantTensor::from_floats(&data, vec![4], mode_norms, 100).unwrap();

        let config = RESConfig {
            population_size: 10,
            survivor_count: 3,
            ..Default::default()
        };

        let mut evolver = ResonantEvolver::from_template(&tensor, config);
        let syntony = evolver.step().unwrap();

        assert!(syntony >= 0.0 && syntony <= 1.0);
        assert_eq!(evolver.current_generation(), 1);
    }
}
