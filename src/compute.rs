use rustfft::num_complex::Complex;

use rand_xoshiro::rand_core::RngCore;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

/// hann window function
/// - x: input
/// - i: index
/// - l_div: reciprocal of window length
pub fn win_hann(x: Complex<f32>, i: usize, l_div: f32) -> Complex<f32> {
    let sin_i = (i as f32 * std::f32::consts::PI * l_div).sin();
    return x * sin_i * sin_i;
}

/// sqrt hann (cosine) window function
/// - x: input
/// - i: index
/// - l_div: reciprocal of window length
pub fn win_hann_sqrt(x: Complex<f32>, i: usize, l_div: f32) -> Complex<f32> {
    return x * (i as f32 * std::f32::consts::PI * l_div).sin();
}

/// random float [0, 1)
/// - rng: provide random number engine Xoshiro256Plus
pub fn randf(rng: &mut Xoshiro256Plus) -> f32 {
    return rng.next_u64() as f32 / u64::MAX as f32;
}
