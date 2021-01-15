use rustfft::num_complex::Complex;

/// hann window function
/// - x: input
/// - i: index
/// - l_div: reciprocal of window length
pub fn win_hann(x: Complex<f32>, i: usize, l_div: f32) -> Complex<f32>{
    let cos_i = (i as f32  *  std::f32::consts::PI  *  l_div).cos();
    return Complex::new(cos_i*cos_i, 0.0)*x;
}