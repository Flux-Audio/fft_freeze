#[macro_use]
extern crate vst;

use vst::buffer::AudioBuffer;
use vst::plugin::{Category, Info, Plugin, PluginParameters};
use vst::util::AtomicFloat;

use std::collections::VecDeque;
use std::sync::Arc;

use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use rustfft::algorithm::Radix4;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFT;

mod compute;

const SIZE: usize = 4096; // must be of the form 2^k >= 32
const L_DIV: f32 = 1.0 / (SIZE as f32);
const OVERLAP: usize = 4; // what proportion of the fft is overlapped (i.e. 4 -> 3/4)
const _NORM: f32 = 2.0 / 3.0;

struct Effect {
    // Store a handle to the plugin's parameter object.
    params: Arc<EffectParameters>,

    // meta variables
    rng: Xoshiro256Plus,
    sr: f32,
    scale: f64, // scaling factor for sr independence of integrals

    // FFT engines
    fft: Radix4<f32>,
    ifft: Radix4<f32>,

    // other variables
    env_z1: f32, // envelope follower

    // FFT variables
    // sample counter (counts to SIZE/4)
    count: usize,

    // time-domain variables
    xl_samp: VecDeque<Complex<f32>>,
    xr_samp: VecDeque<Complex<f32>>,

    // frequency-domain variables
    yl_bins_z1: Vec<Complex<f32>>,
    yr_bins_z1: Vec<Complex<f32>>,

    // output buffer
    yl_samp: VecDeque<f32>,
    yr_samp: VecDeque<f32>,
}

struct EffectParameters {
    window_mode: AtomicFloat,
    freeze_mode: AtomicFloat,
    freeze: AtomicFloat,
    diffuse: AtomicFloat,
    env_amt: AtomicFloat,
    env_time: AtomicFloat,
}

impl Default for Effect {
    fn default() -> Effect {
        Effect {
            params: Arc::new(EffectParameters::default()),

            // meta variables
            rng: Xoshiro256Plus::seed_from_u64(58249537),
            sr: 44100.0,
            scale: 1.0,

            // FFT engines
            fft: Radix4::new(SIZE, false),
            ifft: Radix4::new(SIZE, true),

            // pre-fill misc variables
            env_z1: 0.0,

            // pre-fill FFT variables
            count: 0,
            xl_samp: VecDeque::from(vec![Complex::zero(); SIZE]),
            xr_samp: VecDeque::from(vec![Complex::zero(); SIZE]),
            yl_bins_z1: vec![Complex::zero(); SIZE],
            yr_bins_z1: vec![Complex::zero(); SIZE],
            yl_samp: VecDeque::from(vec![0.0; SIZE]),
            yr_samp: VecDeque::from(vec![0.0; SIZE]),
        }
    }
}

impl Default for EffectParameters {
    fn default() -> EffectParameters {
        EffectParameters {
            window_mode: AtomicFloat::new(0.0),
            freeze_mode: AtomicFloat::new(0.0),
            freeze: AtomicFloat::new(0.0),
            diffuse: AtomicFloat::new(0.0),
            env_amt: AtomicFloat::new(0.5),
            env_time: AtomicFloat::new(0.0),
        }
    }
}

// All plugins using `vst` also need to implement the `Plugin` trait.  Here, we
// define functions that give necessary info to our host.
impl Plugin for Effect {
    fn get_info(&self) -> Info {
        Info {
            name: "FFT_FREEZE".to_string(),
            vendor: "Flux-Audio".to_string(),
            unique_id: 72763875,
            version: 020,
            inputs: 2,
            outputs: 2,
            // This `parameters` bit is important; without it, none of our
            // parameters will be shown!
            parameters: 6,
            category: Category::Effect,
            initial_delay: 1024,
            ..Default::default()
        }
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sr = rate;
        self.scale = 44100.0 / rate as f64;
    }

    // called once
    fn init(&mut self) {}

    // Here is where the bulk of our audio processing code goes.
    fn process(&mut self, buffer: &mut AudioBuffer<f32>) {
        let (inputs, outputs) = buffer.split();

        // Iterate over inputs as (&f32, &f32)
        let (l, r) = inputs.split_at(1);
        let stereo_in = l[0].iter().zip(r[0].iter());

        // Iterate over outputs as (&mut f32, &mut f32)
        let (mut l, mut r) = outputs.split_at_mut(1);
        let stereo_out = l[0].iter_mut().zip(r[0].iter_mut());

        // process
        for ((left_in, right_in), (left_out, right_out)) in stereo_in.zip(stereo_out) {
            // === get params ==================================================
            let mut freeze = self.params.freeze.get().sqrt();
            let diffuse = self.params.diffuse.get();
            let env_amt = self.params.env_amt.get() * 4.0 - 2.0;
            let env_time = self.params.env_time.get().sqrt();
            let window_mode = (self.params.window_mode.get() * 3.0).round() as u16;
            let freeze_mode = (self.params.freeze_mode.get() * 6.0).round() as u16;

            // === buffering inputs ============================================
            // rotate buffer, pushing new samples and discarding front
            let xl = *left_in;
            let xr = *right_in;
            self.xl_samp.pop_front();
            self.xl_samp.push_back(Complex::new(xl, 0.0));
            self.xr_samp.pop_front();
            self.xr_samp.push_back(Complex::new(xr, 0.0));
            self.count += 1;
            // !!! invariant: xl_samp.size == SIZE
            // !!! invariant: xr_samp.size == SIZE

            // === perform FFT on inputs =======================================
            // if buffer has advanced by 25% (75% overlap), perform FFT
            if self.count >= SIZE / OVERLAP {
                self.count = 0;

                // deep-copy input buffers into fft input
                let mut xl_fft: Vec<Complex<f32>> = Vec::with_capacity(SIZE);
                let mut xr_fft: Vec<Complex<f32>> = Vec::with_capacity(SIZE);
                for i in 0..SIZE {
                    xl_fft.push(self.xl_samp[i]);
                    xr_fft.push(self.xr_samp[i]);
                }

                // apply windowing function
                for i in 0..SIZE {
                    match window_mode {
                        0 => {
                            xl_fft[i] = compute::win_hann(xl_fft[i], i, L_DIV);
                            xr_fft[i] = compute::win_hann(xr_fft[i], i, L_DIV);
                        }
                        1 => {
                            xl_fft[i] = compute::win_tri(xl_fft[i], i, L_DIV);
                            xr_fft[i] = compute::win_tri(xr_fft[i], i, L_DIV);
                        }
                        2 => {
                            xl_fft[i] = compute::win_black(xl_fft[i], i, L_DIV);
                            xr_fft[i] = compute::win_black(xr_fft[i], i, L_DIV);
                        }
                        3 => {
                            xl_fft[i] = compute::win_flat(xl_fft[i], i, L_DIV);
                            xr_fft[i] = compute::win_flat(xr_fft[i], i, L_DIV);
                        }
                        _ => {}
                    }
                }

                // forward fft
                let mut xl_bins: Vec<Complex<f32>> = vec![Complex::zero(); SIZE];
                let mut xr_bins: Vec<Complex<f32>> = vec![Complex::zero(); SIZE];
                self.fft.process(&mut xl_fft, &mut xl_bins);
                self.fft.process(&mut xr_fft, &mut xr_bins);

                // normalize
                xl_bins.iter_mut().for_each(|elem| *elem *= L_DIV);
                xr_bins.iter_mut().for_each(|elem| *elem *= L_DIV);

                // === envelope follower =======================================
                // take max of absolute values
                let mut max = 0.0;
                for i in 0..SIZE {
                    let aux = xl_fft[i].re.abs();
                    if aux > max {
                        max = aux;
                    }
                    let aux = xr_fft[i].re.abs();
                    if aux > max {
                        max = aux;
                    }
                }
                // envelope is moving average with previous max
                self.env_z1 = self.env_z1 * env_time + max * (1.0 - env_time);
                freeze = (freeze + self.env_z1 * env_amt).clamp(0.0, 1.0);

                // === spectral freeze =========================================
                match freeze_mode {
                    0 => {
                        compute::flat_freeze(
                            SIZE,
                            &mut xl_bins,
                            &mut xr_bins,
                            &mut self.yl_bins_z1,
                            &mut self.yr_bins_z1,
                            freeze,
                            diffuse,
                            &mut self.rng,
                        );
                    }
                    1 => {
                        compute::glitch_freeze(
                            SIZE,
                            &mut xl_bins,
                            &mut xr_bins,
                            &mut self.yl_bins_z1,
                            &mut self.yr_bins_z1,
                            freeze,
                            diffuse,
                            &mut self.rng,
                        );
                    }
                    2 => {
                        compute::random_freeze(
                            SIZE,
                            &mut xl_bins,
                            &mut xr_bins,
                            &mut self.yl_bins_z1,
                            &mut self.yr_bins_z1,
                            freeze,
                            diffuse,
                            &mut self.rng,
                        );
                    }
                    3 => {
                        compute::reso_freeze(
                            SIZE,
                            &mut xl_bins,
                            &mut xr_bins,
                            &mut self.yl_bins_z1,
                            &mut self.yr_bins_z1,
                            freeze,
                            diffuse,
                            &mut self.rng,
                        );
                    }
                    _ => {}
                }

                // === inverse FFT =============================================
                // inverse fft
                let mut xl_samp_i: Vec<Complex<f32>> = vec![Complex::zero(); SIZE];
                let mut xr_samp_i: Vec<Complex<f32>> = vec![Complex::zero(); SIZE];
                self.ifft.process(&mut xl_bins, &mut xl_samp_i);
                self.ifft.process(&mut xr_bins, &mut xr_samp_i);

                // apply window to output twice (total window = hann^4)
                for i in 0..SIZE {
                    match window_mode {
                        0 => {
                            xl_samp_i[i] = compute::win_hann(xl_samp_i[i], i, L_DIV);
                            xr_samp_i[i] = compute::win_hann(xr_samp_i[i], i, L_DIV);
                        }
                        1 => {
                            xl_samp_i[i] = compute::win_tri(xl_samp_i[i], i, L_DIV);
                            xr_samp_i[i] = compute::win_tri(xr_samp_i[i], i, L_DIV);
                        }
                        2 => {
                            xl_samp_i[i] = compute::win_black(xl_samp_i[i], i, L_DIV);
                            xr_samp_i[i] = compute::win_black(xr_samp_i[i], i, L_DIV);
                        }
                        3 => {
                            xl_samp_i[i] = compute::win_flat(xl_samp_i[i], i, L_DIV);
                            xr_samp_i[i] = compute::win_flat(xr_samp_i[i], i, L_DIV);
                        }
                        _ => {}
                    }

                    // sum output into output buffer
                    let auxl = self.yl_samp.pop_front().unwrap();
                    let auxr = self.yr_samp.pop_front().unwrap();
                    self.yl_samp.push_back(auxl + xl_samp_i[i].re * _NORM);
                    self.yr_samp.push_back(auxr + xr_samp_i[i].re * _NORM);
                    // NOTE: all samples are popped and pushed, so they return
                    // to their initial positions. Ordering is not affected
                }
            }

            // === output ======================================================
            // compensate for freeze gain loss
            *left_out = self.yl_samp.pop_front().unwrap_or(0.0);
            *right_out = self.yr_samp.pop_front().unwrap_or(0.0);
            self.yl_samp.push_back(0.0);
            self.yr_samp.push_back(0.0);
        }
    }

    // Return the parameter object. This method can be omitted if the
    // plugin has no parameters.
    fn get_parameter_object(&mut self) -> Arc<dyn PluginParameters> {
        Arc::clone(&self.params) as Arc<dyn PluginParameters>
    }
}

impl PluginParameters for EffectParameters {
    // the `get_parameter` function reads the value of a parameter.
    fn get_parameter(&self, index: i32) -> f32 {
        match index {
            0 => self.freeze.get(),
            1 => self.diffuse.get(),
            2 => self.env_amt.get(),
            3 => self.env_time.get(),
            4 => self.window_mode.get(),
            5 => self.freeze_mode.get(),
            _ => 0.0,
        }
    }

    // the `set_parameter` function sets the value of a parameter.
    fn set_parameter(&self, index: i32, val: f32) {
        #[allow(clippy::single_match)]
        match index {
            0 => self.freeze.set(val),
            1 => self.diffuse.set(val),
            2 => self.env_amt.set(val),
            3 => self.env_time.set(val),
            4 => self.window_mode.set(val),
            5 => self.freeze_mode.set(val),
            _ => (),
        }
    }

    // This is what will display underneath our control.  We can
    // format it into a string that makes the most since.
    fn get_parameter_text(&self, index: i32) -> String {
        match index {
            0 => format!("{:.2}", self.freeze.get()),
            1 => format!("{:.2}", self.diffuse.get()),
            2 => format!("{:.2}", self.env_amt.get() * 4.0 - 2.0),
            3 => format!("{:.2}", self.env_time.get()),
            4 => match (self.window_mode.get() * 3.0).round() as u16 {
                    0 => "Balanced",
                    1 => "Smear",
                    2 => "Clean",
                    3 => "Flutter",
                    _ => "",
                }.to_string(),
            5 => match (self.freeze_mode.get() * 6.0).round() as u16 {
                    0 => "Normal",
                    1 => "Glitchy",
                    2 => "Random",
                    3 => "Resonant",
                    4 => "Fuzzy",
                    5 => "Dull",
                    6 => "Mashup",
                    _ => "",
                }.to_string(),
            _ => "".to_string(),
        }
    }

    // This shows the control's name.
    fn get_parameter_name(&self, index: i32) -> String {
        match index {
            0 => "freeze",
            1 => "diffuse",
            2 => "envelope amount",
            3 => "envelope time",
            4 => "window mode",
            5 => "freeze mode",
            _ => "",
        }
        .to_string()
    }
}

// This part is important! Without it, our plugin won't work.
plugin_main!(Effect);