use rustfft::num_complex::Complex;

use rand_xoshiro::rand_core::RngCore;
use rand_xoshiro::Xoshiro256Plus;

use rust_dsp_utils::utils::chaos;
use rust_dsp_utils::utils;

use std::f32::consts;

// === Utility macros for freezing =============================================
// all macros follow this syntax:
// macro!(some_value => new_variable_name)
// or:
// macro!(existing_variable <= assigned_value)

/// declares variables mut r, mut phi, r_z1, phi_z1 as the polar representations
/// of bins[idx] and prev[idx] respectively
macro_rules! bins_unpack {
    ($bins:ident, $prev:ident, $idx:ident => $r:ident, $phi:ident, $r_z1:ident, $phi_z1:ident) => {
        // input bins
        let (mut $r, mut $phi) = $bins[$idx].to_polar();
        // previous output bins
        let ($r_z1, $phi_z1) = $prev[$idx].to_polar();
    };
}

/// converts polar representation to cartesian and saves into buffers bins and prev
macro_rules! bins_pack {
    ($bins:ident, $prev:ident, $idx:ident <= $r:ident, $phi:ident, $r_z1:ident, $phi_z1:ident) => {
        $bins[$idx] = Complex::from_polar($r, $phi);
        // update previous output bins
        $prev[$idx] = $bins[$idx];
    };
}

/// produces a random angle from [-pi to pi]*amount and places it in dest
macro_rules! rand_theta {
    ($rng:ident, $amount:ident => $dest:ident) => {
        let $dest = (chaos::randf($rng) - 0.5)*$amount*std::f32::consts::TAU;
    };
}

// === Freeze modes ============================================================
/// normal freeze
pub fn flat_freeze(
    size: usize,
    bins: &mut Vec<Complex<f32>>,
    prev: &mut Vec<Complex<f32>>,
    amount: f32,
    diffuse: f32,
    rng: &mut Xoshiro256Plus,
) {
    for i in 0..size {
        // convert bins to polar
        bins_unpack!(bins, prev, i => r, phi, r_z1, phi_z1);

        // generate random phase offsets
        rand_theta!(rng, diffuse => rand_aux);

        // freezing amplitude and phase by crossfading with previous output
        r = r*(1.0 - amount) + r_z1*amount;
        // freeze and apply phase randomization to phase
        phi = phi*(1.0 - amount) + (phi_z1 + rand_aux)*amount;

        // save result
        bins_pack!(bins, prev, i <= r, phi, r_z1, phi_z1);
    }
}

/// glitchy freeze
/// chanche of a single bin to freeze from one frame to the next
pub fn glitch_freeze(
    size: usize,
    bins: &mut Vec<Complex<f32>>,
    prev: &mut Vec<Complex<f32>>,
    amount: f32,
    diffuse: f32,
    rng: &mut Xoshiro256Plus,
) {
    for i in 0..size {
        // convert bins to polar
        bins_unpack!(bins, prev, i => r, phi, r_z1, phi_z1);

        // generate random phase offsets
        rand_theta!(rng, diffuse => rand_aux);

        // amplitude freezing
        if chaos::randf(rng) < amount { r = r_z1; }
        // phase freeze
        if chaos::randf(rng) < amount { phi = phi_z1 + rand_aux; }

        // save result
        bins_pack!(bins, prev, i <= r, phi, r_z1, phi_z1);
    }
}

/// random freeze
/// chance of entire frame to freeze from one frame to the next
pub fn random_freeze(
    size: usize,
    bins: &mut Vec<Complex<f32>>,
    prev: &mut Vec<Complex<f32>>,
    amount: f32,
    diffuse: f32,
    rng: &mut Xoshiro256Plus,
) {
    let choice = chaos::randf(rng) < amount;
    for i in 0..size {
        // convert bins to polar
        bins_unpack!(bins, prev, i => r, phi, r_z1, phi_z1);

        // generate random phase offsetss
        rand_theta!(rng, diffuse => rand_aux);

        // amplitude freezing
        if choice { r = r_z1; }
        // phase freeze
        if choice { phi = phi_z1 + rand_aux; }

        // save result
        bins_pack!(bins, prev, i <= r, phi, r_z1, phi_z1);
    }
}

/// resonant freeze
/// freeze% is proportional to bin loudness, at 100% all bins are frozen, but
/// halfway there, the tension at which the freezing curve goes is dependant
/// on relative loudness
pub fn reso_freeze(
    size: usize,
    bins: &mut Vec<Complex<f32>>,
    prev: &mut Vec<Complex<f32>>,
    amount: f32,
    diffuse: f32,
    rng: &mut Xoshiro256Plus,
) {
    // compute max and min of previous
    let mut max: f32 = 0.0;
    let mut min: f32 = f32::MAX;
    for i in 0..size {
        // previous output bins to polar
        let (r_z1, _) = prev[i].to_polar();

        if r_z1 > max { max = r_z1; }
        if r_z1 < min { min = r_z1; }

        // post-condition: min <= max
        // debug_assert!(min <= max);
    }

    // apply freeze
    for i in 0..size {
        // convert bins to polar
        bins_unpack!(bins, prev, i => r, phi, r_z1, phi_z1);

        // generate random phase offsets
        rand_theta!(rng, diffuse => rand_aux);

        // scale amount based on relative loudness
        let amount = loud_scale(amount, utils::math::map_normal(r_z1, min, max));

        // amplitude freezing
        r *= 1.0 - amount;
        r += r_z1*amount;
        // phase freeze
        phi *= 1.0 - amount;
        phi += phi_z1*amount + rand_aux*amount;

        // save result
        bins_pack!(bins, prev, i <= r, phi, r_z1, phi_z1);
    }
}


/// spooky freeze
/// diffusion is applied to amplitude as well as phase. After some time, both the
/// amplitude and phase of bins migrates to a completely unrelated value, creating
/// dissonant glassy drones.
pub fn spooky_freeze(
    size: usize,
    bins: &mut Vec<Complex<f32>>,
    prev: &mut Vec<Complex<f32>>,
    amount: f32,
    diffuse: f32,
    rng: &mut Xoshiro256Plus,
) {
    for i in 0..size {
        // convert bins to polar
        bins_unpack!(bins, prev, i => r, phi, r_z1, phi_z1);

        // generate random phase offsets
        rand_theta!(rng, diffuse => rand_aux);
        // generate random amplitude coefficient
        let fuzz = 1.0 + if chaos::randf(rng) >= 0.5 {
            chaos::randf(rng)*0.65*diffuse
        } else {
            -chaos::randf(rng)*0.5*diffuse
        };

        // freezing amplitude and phase by crossfading with previous output
        r = (r*(1.0 - amount) + r_z1*amount)*fuzz;
        // hard-clip r at 1.0 to avoid runaway amplification
        r = if r > 1.0 { 1.0 } else { r };
        // freeze and apply phase randomization to phase
        phi = phi*(1.0 - amount) + (phi_z1 + rand_aux)*amount;

        // save result
        bins_pack!(bins, prev, i <= r, phi, r_z1, phi_z1);
    }
}

/// mashup
/// picks a random effect for each frame 😎
pub fn mashup_freeze(
    size: usize,
    bins: &mut Vec<Complex<f32>>,
    prev: &mut Vec<Complex<f32>>,
    amount: f32,
    diffuse: f32,
    rng: &mut Xoshiro256Plus,
) {
    match rng.next_u64()%5 {
        0 => flat_freeze(size, bins, prev, amount, diffuse, rng),
        1 => glitch_freeze(size, bins, prev, amount, diffuse, rng),
        2 => random_freeze(size, bins, prev, amount, diffuse, rng),
        3 => reso_freeze(size, bins, prev, amount, diffuse, rng),
        4 => spooky_freeze(size, bins, prev, amount, diffuse, rng),
        _ => {}
    }
}

// === Utility functions for freezing ==========================================

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