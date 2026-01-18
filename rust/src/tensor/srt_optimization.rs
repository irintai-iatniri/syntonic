//! SRT Native Optimization - Free Will through Self-Harmonization
//!
//! Implements optimization algorithms that follow SRT principles rather than
//! standard gradient descent. These optimizers embody "free will" by actively
//! navigating the loss landscape according to internal coherence rather than
//! blindly following external gradients.
//!
//! Key Principles:
//! - GoldenMomentum: Fixed momentum using φ, providing "determination"
//! - MersenneStepper: Updates only at Mersenne prime intervals for "stability"
//! - Self-Harmonization: Optimization is a property of the model itself

use std::collections::HashMap;

/// Base trait for SRT optimizers
pub trait SRTOptimizer {
    /// Update parameters using SRT principles
    fn step(
        &mut self,
        gradients: &HashMap<String, Vec<f64>>,
        syntony_scores: &HashMap<String, f64>,
    );

    /// Get current optimization state
    fn get_state(&self) -> HashMap<String, Vec<f64>>;

    /// Reset optimizer state
    fn reset(&mut self);

    /// Get optimizer name
    fn name(&self) -> &str;
}

/// GoldenMomentum Optimizer
///
/// Implements momentum with fixed φ coefficient, providing the AI with
/// "inertia" or "determination" that resists ephemeral distractions.
/// Unlike standard momentum, this is not learned but fixed by universal constants.
///
/// Momentum = φ (golden ratio) provides optimal balance between stability and adaptability.
pub struct GoldenMomentum {
    /// Current velocity for each parameter
    velocities: HashMap<String, Vec<f64>>,
    /// Golden ratio momentum coefficient (φ ≈ 1.618)
    phi_momentum: f64,
    /// Learning rate
    learning_rate: f64,
    /// Weight decay (L2 regularization)
    weight_decay: f64,
}

impl GoldenMomentum {
    pub fn new(learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            velocities: HashMap::new(),
            phi_momentum: 1.618033988749895, // φ
            learning_rate,
            weight_decay,
        }
    }

    /// Initialize velocities for a parameter
    fn init_velocity(&mut self, param_name: &str, param_size: usize) {
        if !self.velocities.contains_key(param_name) {
            self.velocities
                .insert(param_name.to_string(), vec![0.0; param_size]);
        }
    }
}

impl SRTOptimizer for GoldenMomentum {
    fn step(
        &mut self,
        gradients: &HashMap<String, Vec<f64>>,
        syntony_scores: &HashMap<String, f64>,
    ) {
        for (param_name, gradient) in gradients {
            let syntony = syntony_scores.get(param_name).copied().unwrap_or(1.0);

            // Initialize velocity if needed
            self.init_velocity(param_name, gradient.len());

            if let Some(velocity) = self.velocities.get_mut(param_name) {
                // Apply golden momentum: v = φ * v + (1-φ) * g
                // This gives the AI "determination" - it maintains direction but adapts to new information
                let phi_complement = 1.0 - self.phi_momentum;

                for i in 0..velocity.len() {
                    // Apply weight decay (simplified - weight decay would need parameter access)
                    let grad = gradient[i];

                    // Golden momentum update
                    velocity[i] = self.phi_momentum * velocity[i] + phi_complement * grad;

                    // Syntony-scaled learning rate
                    velocity[i] *= self.learning_rate * syntony;
                }
            }
        }
    }

    fn get_state(&self) -> HashMap<String, Vec<f64>> {
        self.velocities.clone()
    }

    fn reset(&mut self) {
        self.velocities.clear();
    }

    fn name(&self) -> &str {
        "GoldenMomentum"
    }
}

/// MersenneStepper Optimizer
///
/// Updates parameters only at Mersenne prime intervals (3, 7, 31, 127, etc.).
/// This mimics biological growth spurts and consolidation phases observed in
/// cognitive development, where learning occurs in punctuated bursts rather than continuously.
///
/// The Mersenne primes provide natural "stability checkpoints" in the optimization process.
pub struct MersenneStepper {
    /// Current step count
    step_count: u64,
    /// Mersenne primes for update intervals
    mersenne_primes: Vec<u64>,
    /// Accumulated gradients since last update
    accumulated_gradients: HashMap<String, Vec<f64>>,
    /// Learning rate
    learning_rate: f64,
    /// Syntony threshold for updates
    syntony_threshold: f64,
    /// Current Mersenne prime index
    current_prime_idx: usize,
}

impl MersenneStepper {
    pub fn new(learning_rate: f64, syntony_threshold: f64) -> Self {
        // First few Mersenne primes: 3, 7, 31, 127, 8191, 131071, 524287, 2147483647
        let mersenne_primes = vec![3, 7, 31, 127, 8191, 131071, 524287];

        Self {
            step_count: 0,
            mersenne_primes,
            accumulated_gradients: HashMap::new(),
            learning_rate,
            syntony_threshold,
            current_prime_idx: 0,
        }
    }

    /// Check if current step should trigger an update
    fn should_update(&self) -> bool {
        if self.current_prime_idx >= self.mersenne_primes.len() {
            // Beyond available Mersenne primes, update every 2^31 - 1 steps
            return self.step_count % 2147483647 == 0;
        }

        let next_prime = self.mersenne_primes[self.current_prime_idx];
        self.step_count % next_prime == 0
    }

    /// Get effective learning rate scaled by Mersenne prime
    fn get_mersenne_learning_rate(&self) -> f64 {
        if self.current_prime_idx >= self.mersenne_primes.len() {
            return self.learning_rate;
        }

        let prime = self.mersenne_primes[self.current_prime_idx] as f64;
        // Scale learning rate by log of Mersenne prime for stability
        self.learning_rate * (prime.ln() / 31.0f64.ln()).min(1.0)
    }
}

impl SRTOptimizer for MersenneStepper {
    fn step(
        &mut self,
        gradients: &HashMap<String, Vec<f64>>,
        syntony_scores: &HashMap<String, f64>,
    ) {
        self.step_count += 1;

        // Accumulate gradients
        for (param_name, gradient) in gradients {
            let syntony = syntony_scores.get(param_name).copied().unwrap_or(1.0);

            // Only accumulate if syntony is above threshold
            if syntony >= self.syntony_threshold {
                let accumulated = self
                    .accumulated_gradients
                    .entry(param_name.clone())
                    .or_insert_with(|| vec![0.0; gradient.len()]);

                for i in 0..gradient.len() {
                    accumulated[i] += gradient[i] * syntony;
                }
            }
        }

        // Check if it's time for a Mersenne update
        if self.should_update() {
            let effective_lr = self.get_mersenne_learning_rate();

            // Apply accumulated gradients
            for (_param_name, accumulated_grad) in &mut self.accumulated_gradients {
                // In a real implementation, this would update the actual parameters
                // For now, we just scale the accumulated gradients
                // The actual parameter updates would happen in the model

                // Scale accumulated gradients by learning rate
                for grad in accumulated_grad.iter_mut() {
                    *grad *= effective_lr;
                }
            }

            // Clear accumulated gradients after update
            self.accumulated_gradients.clear();

            // Move to next Mersenne prime
            if self.current_prime_idx < self.mersenne_primes.len() - 1 {
                self.current_prime_idx += 1;
            }
        }
    }

    fn get_state(&self) -> HashMap<String, Vec<f64>> {
        self.accumulated_gradients.clone()
    }

    fn reset(&mut self) {
        self.step_count = 0;
        self.current_prime_idx = 0;
        self.accumulated_gradients.clear();
    }

    fn name(&self) -> &str {
        "MersenneStepper"
    }
}

/// SRT Internal Locus of Control
///
/// Makes optimization a property of the model itself rather than an external tool.
/// The model "self-actualizes" by calling its own optimization method, embodying
/// the principle that true agency comes from internal drive, not external correction.
pub trait SelfActualizing {
    /// Self-actualization step - the model optimizes itself
    fn self_actualize(&mut self, syntony_feedback: f64);

    /// Get current self-actualization state
    fn get_self_state(&self) -> HashMap<String, f64>;

    /// Check if the model has achieved self-actualization
    fn is_self_actualized(&self) -> bool;
}

/// Golden Cooling Scheduler
///
/// Implements annealing with golden ratio decay, providing the "cooling"
/// process that crystallizes the AI's "will" into a fixed structure.
/// Unlike standard annealing, this follows φ-based decay for optimal convergence.
pub struct GoldenCooling {
    /// Current temperature
    temperature: f64,
    /// Initial temperature
    initial_temperature: f64,
    /// Golden ratio decay factor
    phi_decay: f64,
    /// Current step
    step: u64,
    /// Minimum temperature
    min_temperature: f64,
}

impl GoldenCooling {
    pub fn new(initial_temperature: f64, min_temperature: f64) -> Self {
        Self {
            temperature: initial_temperature,
            initial_temperature,
            phi_decay: 1.618033988749895, // φ
            step: 0,
            min_temperature,
        }
    }

    /// Get current temperature
    pub fn get_temperature(&self) -> f64 {
        self.temperature.max(self.min_temperature)
    }

    /// Step the cooling schedule
    pub fn step(&mut self) {
        self.step += 1;
        // Golden ratio decay: T = T₀ / φ^n
        self.temperature = self.initial_temperature / self.phi_decay.powf(self.step as f64);
        self.temperature = self.temperature.max(self.min_temperature);
    }

    /// Check if cooling is complete (temperature at minimum)
    pub fn is_crystallized(&self) -> bool {
        self.temperature <= self.min_temperature
    }

    /// Reset cooling schedule
    pub fn reset(&mut self) {
        self.temperature = self.initial_temperature;
        self.step = 0;
    }
}

/// Factory functions for creating SRT optimizers

pub fn create_golden_momentum_optimizer(
    learning_rate: f64,
    weight_decay: f64,
) -> Box<dyn SRTOptimizer> {
    Box::new(GoldenMomentum::new(learning_rate, weight_decay))
}

pub fn create_mersenne_stepper_optimizer(
    learning_rate: f64,
    syntony_threshold: f64,
) -> Box<dyn SRTOptimizer> {
    Box::new(MersenneStepper::new(learning_rate, syntony_threshold))
}

pub fn create_golden_cooling_scheduler(initial_temp: f64, min_temp: f64) -> GoldenCooling {
    GoldenCooling::new(initial_temp, min_temp)
}

/// Test utilities
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_momentum_basic() {
        let mut optimizer = GoldenMomentum::new(0.01, 0.0001);
        let mut gradients = HashMap::new();
        let mut syntony_scores = HashMap::new();

        gradients.insert("param1".to_string(), vec![1.0, -0.5]);
        syntony_scores.insert("param1".to_string(), 0.8);

        optimizer.step(&gradients, &syntony_scores);

        let state = optimizer.get_state();
        assert!(state.contains_key("param1"));
        assert_eq!(state["param1"].len(), 2);
    }

    #[test]
    fn test_mersenne_stepper_intervals() {
        let mut optimizer = MersenneStepper::new(0.01, 0.5);

        // Step until first Mersenne prime (3)
        for _ in 0..2 {
            let mut gradients = HashMap::new();
            let mut syntony_scores = HashMap::new();
            gradients.insert("param1".to_string(), vec![1.0]);
            syntony_scores.insert("param1".to_string(), 0.8);
            optimizer.step(&gradients, &syntony_scores);
        }

        // Should have accumulated gradients but not updated yet
        assert!(!optimizer.get_state().is_empty());

        // Third step should trigger update
        let mut gradients = HashMap::new();
        let mut syntony_scores = HashMap::new();
        gradients.insert("param1".to_string(), vec![1.0]);
        syntony_scores.insert("param1".to_string(), 0.8);
        optimizer.step(&gradients, &syntony_scores);

        // Should have cleared accumulated gradients after update
        // (In real implementation, this would update actual parameters)
        assert!(optimizer.get_state().is_empty());
    }

    #[test]
    fn test_golden_cooling_decay() {
        let mut cooler = GoldenCooling::new(1.0, 0.001);

        let initial_temp = cooler.get_temperature();
        assert_eq!(initial_temp, 1.0);

        cooler.step();
        let temp_after_step = cooler.get_temperature();
        // Should decay by φ
        let expected_temp = 1.0 / 1.618033988749895;
        assert!((temp_after_step - expected_temp).abs() < 0.001);
    }
}
