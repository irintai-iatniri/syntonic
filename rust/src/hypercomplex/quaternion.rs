use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct Quaternion {
    #[pyo3(get, set)]
    pub a: f64,
    #[pyo3(get, set)]
    pub b: f64,
    #[pyo3(get, set)]
    pub c: f64,
    #[pyo3(get, set)]
    pub d: f64,
}

#[pymethods]
impl Quaternion {
    #[new]
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Quaternion { a, b, c, d }
    }
    
    #[getter]
    pub fn real(&self) -> f64 { self.a }
    
    #[getter]
    pub fn imag(&self) -> Vec<f64> { vec![self.b, self.c, self.d] }
    
    pub fn conjugate(&self) -> Self {
        Quaternion::new(self.a, -self.b, -self.c, -self.d)
    }
    
    pub fn norm(&self) -> f64 {
        (self.a*self.a + self.b*self.b + self.c*self.c + self.d*self.d).sqrt()
    }
    
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n == 0.0 {
            // Handle zero quaternion gracefully? Or return zero/error?
            // Standard convention: keep 0.
            return *self
        }
        Quaternion::new(self.a/n, self.b/n, self.c/n, self.d/n)
    }
    
    pub fn inverse(&self) -> Self {
        let n2 = self.a*self.a + self.b*self.b + self.c*self.c + self.d*self.d;
        if n2 == 0.0 {
            // Technically undefined, but returning zero often useful or raising error.
            // Let's implement division by zero panic or similar in caller/wrapper usually,
            // but here returning NaN/Inf components is standard float behavior.
             return Quaternion::new(self.a/n2, -self.b/n2, -self.c/n2, -self.d/n2)
        }
        let conj = self.conjugate();
        Quaternion::new(conj.a/n2, conj.b/n2, conj.c/n2, conj.d/n2)
    }
    
    // Arithmetic
    
    fn __add__(&self, other: &Quaternion) -> Self {
        Quaternion::new(self.a+other.a, self.b+other.b, self.c+other.c, self.d+other.d)
    }
    
    fn __sub__(&self, other: &Quaternion) -> Self {
        Quaternion::new(self.a-other.a, self.b-other.b, self.c-other.c, self.d-other.d)
    }
    
    fn __neg__(&self) -> Self {
        Quaternion::new(-self.a, -self.b, -self.c, -self.d)
    }

    /// Hamilton product
    fn __mul__(&self, other: &Quaternion) -> Self {
        Quaternion::new(
            self.a*other.a - self.b*other.b - self.c*other.c - self.d*other.d,
            self.a*other.b + self.b*other.a + self.c*other.d - self.d*other.c,
            self.a*other.c - self.b*other.d + self.c*other.a + self.d*other.b,
            self.a*other.d + self.b*other.c - self.c*other.b + self.d*other.a,
        )
    }
    
    fn __truediv__(&self, other: &Quaternion) -> Self {
        // q1 / q2 = q1 * q2^-1
        self.__mul__(&other.inverse())
    }
    
    fn __repr__(&self) -> String {
        format!("Quaternion(a={:.4}, b={:.4}, c={:.4}, d={:.4})", self.a, self.b, self.c, self.d)
    }

    // Advanced Ops
    
    pub fn to_rotation_matrix(&self) -> Vec<Vec<f64>> {
        let q = self.normalize();
        let (a, b, c, d) = (q.a, q.b, q.c, q.d);
        
        vec![
            vec![1.0-2.0*(c*c+d*d), 2.0*(b*c-d*a),     2.0*(b*d+c*a)],
            vec![2.0*(b*c+d*a),     1.0-2.0*(b*b+d*d), 2.0*(c*d-b*a)],
            vec![2.0*(b*d-c*a),     2.0*(c*d+b*a),     1.0-2.0*(b*b+c*c)],
        ]
    }
    
    #[staticmethod]
    pub fn from_axis_angle(axis: Vec<f64>, theta: f64) -> PyResult<Self> {
        if axis.len() != 3 {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Axis must be length 3"));
        }
        let half = theta / 2.0;
        let s = half.sin();
        let norm = (axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]).sqrt();
        if norm == 0.0 {
            // Return identity if axis is zero
            return Ok(Quaternion::new(1.0, 0.0, 0.0, 0.0));
        }
        Ok(Quaternion::new(
            half.cos(),
            s * axis[0] / norm,
            s * axis[1] / norm,
            s * axis[2] / norm,
        ))
    }
    
    #[staticmethod]
    pub fn slerp(q1: &Quaternion, q2: &Quaternion, t: f64) -> Self {
        let mut dot = q1.a*q2.a + q1.b*q2.b + q1.c*q2.c + q1.d*q2.d;
        
        // Handle negative dot (opposite hemispheres) - take shortest path
        let q2_adj = if dot < 0.0 {
            dot = -dot;
            Quaternion::new(-q2.a, -q2.b, -q2.c, -q2.d)
        } else {
            *q2
        };
        
        let dot = dot.min(1.0).max(-1.0);
        let theta = dot.acos();
        let sin_theta = theta.sin();
        
        if sin_theta.abs() < 1e-6 {
            // Linear interpolation for small angles
            Quaternion::new(
                q1.a + t*(q2_adj.a - q1.a),
                q1.b + t*(q2_adj.b - q1.b),
                q1.c + t*(q2_adj.c - q1.c),
                q1.d + t*(q2_adj.d - q1.d),
            ).normalize()
        } else {
            let s1 = ((1.0-t)*theta).sin() / sin_theta;
            let s2 = (t*theta).sin() / sin_theta;
            Quaternion::new(
                s1*q1.a + s2*q2_adj.a,
                s1*q1.b + s2*q2_adj.b,
                s1*q1.c + s2*q2_adj.c,
                s1*q1.d + s2*q2_adj.d,
            )
        }
    }
}
