use rustfft::num_complex::Complex;

use rand_xoshiro::rand_core::RngCore;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use std::f32::consts;

/// hann window function
/// - x: input
/// - i: index
/// - l_div: reciprocal of window length
pub fn win_hann(x: Complex<f32>, i: usize, l_div: f32) -> Complex<f32> {
    let sin_i = (i as f32 * std::f32::consts::PI * l_div).sin();
    return x * sin_i * sin_i;
}

/// triangular window function
/// - x: input
/// - i: index
/// - l_div: reciprocal of window length
pub fn win_tri(x: Complex<f32>, i: usize, l_div: f32) -> Complex<f32> {
    let tri_i = 1.0 - (2.0 * i as f32 * l_div - 1.0).abs();
    return x * tri_i;
}

/// blackman window function
/// - x: input
/// - i: index
/// - l_div: reciprocal of window length
pub fn win_black(x: Complex<f32>, i: usize, l_div: f32) -> Complex<f32> {
    let a_0 = 0.42659;
    let a_1 = 0.49656;
    let a_2 = 0.076849;
    let win = a_0 - a_1 * (consts::TAU * i as f32 * l_div).cos()
        + a_2 * (2.0 * consts::TAU * i as f32 * l_div).cos();
    return x * win;
}

/// nuttal window function
/// - x: input
/// - i: index
/// - l_div: reciprocal of window length
pub fn win_nutt(x: Complex<f32>, i: usize, l_div: f32) -> Complex<f32> {
    let a_0 = 0.355768;
    let a_1 = 0.487396;
    let a_2 = 0.144232;
    let a_3 = 0.012604;
    let win = a_0 - a_1 * (consts::TAU * i as f32 * l_div).cos()
        + a_2 * (2.0 * consts::TAU * i as f32 * l_div).cos()
        - a_3 * (3.0 * consts::TAU * i as f32 * l_div).cos();
    return x * win;
}

/// flat top window function
/// - x: input
/// - i: index
/// - l_div: reciprocal of window length
pub fn win_flat(x: Complex<f32>, i: usize, l_div: f32) -> Complex<f32> {
    let a_0 = 0.21557895;
    let a_1 = 0.41663158;
    let a_2 = 0.277263158;
    let a_3 = 0.083578947;
    let a_4 = 0.006947368;
    let win = a_0 - a_1 * (consts::TAU * i as f32 * l_div).cos()
        + a_2 * (2.0 * consts::TAU * i as f32 * l_div).cos()
        - a_3 * (3.0 * consts::TAU * i as f32 * l_div).cos()
        + a_4 * (4.0 * consts::TAU * i as f32 * l_div).cos();
    return x * win;
}

/// random float [0, 1)
/// - rng: provide random number engine Xoshiro256Plus
pub fn randf(rng: &mut Xoshiro256Plus) -> f32 {
    return rng.next_u64() as f32 / u64::MAX as f32;
}

// === Freeze modes ============================================================
/// normal freeze
pub fn flat_freeze(
    size: usize,
    l_bins: &mut Vec<Complex<f32>>,
    r_bins: &mut Vec<Complex<f32>>,
    l_prev: &mut Vec<Complex<f32>>,
    r_prev: &mut Vec<Complex<f32>>,
    amount: f32,
    diffuse: f32,
    rng: &mut Xoshiro256Plus,
) {
    for i in 0..size {
        // convert bins to polar
        // input bins
        let (mut l_r, mut l_phi) = l_bins[i].to_polar();
        let (mut r_r, mut r_phi) = r_bins[i].to_polar();
        // previous output bins
        let (mut l_r_z1, mut l_phi_z1) = l_prev[i].to_polar();
        let (mut r_r_z1, mut r_phi_z1) = r_prev[i].to_polar();

        // amplitude freezing
        l_r *= (1.0 - amount);
        r_r *= (1.0 - amount);
        l_r += l_r_z1 * amount;
        r_r += r_r_z1 * amount;

        // generate random phase offsets
        let rand_aux_1 = (randf(rng) - 0.5) * diffuse * std::f32::consts::TAU;
        let rand_aux_2 = (randf(rng) - 0.5) * diffuse * std::f32::consts::TAU;

        // phase freeze
        l_phi *= (1.0 - amount);
        r_phi *= (1.0 - amount);
        l_phi += l_phi_z1 * amount + rand_aux_1 * amount;
        r_phi += r_phi_z1 * amount + rand_aux_2 * amount;

        // save result
        l_bins[i] = Complex::from_polar(l_r, l_phi);
        r_bins[i] = Complex::from_polar(r_r, r_phi);

        // update previous output bins
        l_prev[i] = l_bins[i];
        r_prev[i] = r_bins[i];
    }
}

/// glitchy freeze
/// chanche of a single bin to freeze from one frame to the next
pub fn glitch_freeze(
    size: usize,
    l_bins: &mut Vec<Complex<f32>>,
    r_bins: &mut Vec<Complex<f32>>,
    l_prev: &mut Vec<Complex<f32>>,
    r_prev: &mut Vec<Complex<f32>>,
    amount: f32,
    diffuse: f32,
    rng: &mut Xoshiro256Plus,
) {
    for i in 0..size {
        // convert bins to polar
        // input bins
        let (mut l_r, mut l_phi) = l_bins[i].to_polar();
        let (mut r_r, mut r_phi) = r_bins[i].to_polar();
        // previous output bins
        let (mut l_r_z1, mut l_phi_z1) = l_prev[i].to_polar();
        let (mut r_r_z1, mut r_phi_z1) = r_prev[i].to_polar();

        // amplitude freezing
        if (randf(rng) < amount) {
            l_r = l_r_z1;
        }
        if (randf(rng) < amount) {
            r_r = r_r_z1;
        }

        // generate random phase offsets
        let rand_aux_1 = (randf(rng) - 0.5) * diffuse * std::f32::consts::TAU;
        let rand_aux_2 = (randf(rng) - 0.5) * diffuse * std::f32::consts::TAU;

        // phase freeze
        if (randf(rng) < amount) {
            l_phi = l_phi_z1 + rand_aux_1;
        }
        if (randf(rng) < amount) {
            r_phi = r_phi_z1 + rand_aux_2;
        }

        // save result
        l_bins[i] = Complex::from_polar(l_r, l_phi);
        r_bins[i] = Complex::from_polar(r_r, r_phi);

        // update previous output bins
        l_prev[i] = l_bins[i];
        r_prev[i] = r_bins[i];
    }
}

/// random freeze
/// chance of entire frame to freeze from one frame to the next
pub fn random_freeze(
    size: usize,
    l_bins: &mut Vec<Complex<f32>>,
    r_bins: &mut Vec<Complex<f32>>,
    l_prev: &mut Vec<Complex<f32>>,
    r_prev: &mut Vec<Complex<f32>>,
    amount: f32,
    diffuse: f32,
    rng: &mut Xoshiro256Plus,
) {
    let choice_1 = randf(rng) < amount;
    let choice_2 = randf(rng) < amount;
    for i in 0..size {
        // convert bins to polar
        // input bins
        let (mut l_r, mut l_phi) = l_bins[i].to_polar();
        let (mut r_r, mut r_phi) = r_bins[i].to_polar();
        // previous output bins
        let (mut l_r_z1, mut l_phi_z1) = l_prev[i].to_polar();
        let (mut r_r_z1, mut r_phi_z1) = r_prev[i].to_polar();

        // amplitude freezing
        if choice_1 {
            l_r = l_r_z1;
        }
        if choice_2 {
            r_r = r_r_z1;
        }

        // generate random phase offsets
        let rand_aux_1 = (randf(rng) - 0.5) * diffuse * std::f32::consts::TAU;
        let rand_aux_2 = (randf(rng) - 0.5) * diffuse * std::f32::consts::TAU;

        // phase freeze
        if choice_1 {
            l_phi = l_phi_z1 + rand_aux_1;
        }
        if choice_2 {
            r_phi = r_phi_z1 + rand_aux_2;
        }

        // save result
        l_bins[i] = Complex::from_polar(l_r, l_phi);
        r_bins[i] = Complex::from_polar(r_r, r_phi);

        // update previous output bins
        l_prev[i] = l_bins[i];
        r_prev[i] = r_bins[i];
    }
}

/// resonant freeze
/// freeze% is proportional to bin loudness, at 100% all bins are frozen, but
/// halfway there, the tension at which the freezing curve goes is dependant
/// on relative loudness
pub fn reso_freeze(
    size: usize,
    l_bins: &mut Vec<Complex<f32>>,
    r_bins: &mut Vec<Complex<f32>>,
    l_prev: &mut Vec<Complex<f32>>,
    r_prev: &mut Vec<Complex<f32>>,
    amount: f32,
    diffuse: f32,
    rng: &mut Xoshiro256Plus,
) {
    // compute max and min of previous
    let mut max: f32 = 0.0;
    let mut min: f32 = f32::MAX;
    for i in 0..size {
        // previous output bins to polar
        let (mut l_r_z1, _) = l_prev[i].to_polar();
        let (mut r_r_z1, _) = r_prev[i].to_polar();

        if l_r_z1 > max {
            max = l_r_z1;
        }
        if r_r_z1 > max {
            max = r_r_z1;
        }
        if l_r_z1 < min {
            min = l_r_z1;
        }
        if r_r_z1 < min {
            min = r_r_z1;
        }

        // post-condition: min <= max
        // debug_assert!(min <= max);
    }

    // apply freeze
    for i in 0..size {
        // convert bins to polar
        // input bins
        let (mut l_r, mut l_phi) = l_bins[i].to_polar();
        let (mut r_r, mut r_phi) = r_bins[i].to_polar();
        // previous output bins
        let (mut l_r_z1, mut l_phi_z1) = l_prev[i].to_polar();
        let (mut r_r_z1, mut r_phi_z1) = r_prev[i].to_polar();

        // scale amount based on relative loudness
        let l_amount = loud_scale(amount, map_normal(l_r_z1, min, max));
        let r_amount = loud_scale(amount, map_normal(r_r_z1, min, max));

        // amplitude freezing
        l_r *= (1.0 - l_amount);
        r_r *= (1.0 - r_amount);
        l_r += l_r_z1*l_amount;
        r_r += r_r_z1*r_amount;

        // generate random phase offsets
        let rand_aux_1 = (randf(rng) - 0.5)*diffuse*std::f32::consts::TAU;
        let rand_aux_2 = (randf(rng) - 0.5)*diffuse*std::f32::consts::TAU;

        // phase freeze
        l_phi *= (1.0 - l_amount);
        r_phi *= (1.0 - r_amount);
        l_phi += l_phi_z1*l_amount + rand_aux_1*l_amount;
        r_phi += r_phi_z1*r_amount + rand_aux_2*r_amount;

        // save result
        l_bins[i] = Complex::from_polar(l_r, l_phi);
        r_bins[i] = Complex::from_polar(r_r, r_phi);

        // update previous output bins
        l_prev[i] = l_bins[i];
        r_prev[i] = r_bins[i];
    }
}

/*
/// fuzzy freeze
/// diffusion is applied to amplitude as well as phase


/// dull


/// mashup
/// picks a random effect for each frame 😎
*/

/// maps x such that: x in [a, b] -> y in [c, d]
fn map_range(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return (d - c) / (b - a) * (x - a) + c;
}

/// maps x such that: x in [a, b] -> y in [0, 1]
fn map_normal(x: f32, a: f32, mut b: f32) -> f32 {
    b = b*0.99999999 + 0.00000001;  // prevent division by zero
    // precondition: x in [a, b]
    // debug_assert!(a <= x);
    // debug_assert!(x <= b);
    return 1.0/(b - a)*(x - a);
}

/// relative loudness and global amount of freeze to local amount of freeze
/// takes the overall freeze amount, and scales it based on the loudness of an
/// individual bin, used for resonant freeze and dull freeze
fn loud_scale(amt: f32, loud: f32) -> f32 {
    // precodition: amt <= 1.0, loud <= 1.0
    // debug_assert!(amt <= 1.0 && loud <= 1.0);
    return if amt < 0.5 {
        2.0*loud*amt
    } else {
        1.0 + 2.0*(loud - 1.0)*(1.0 - amt)
    };
}