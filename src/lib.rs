#[macro_use]
extern crate vst;

use vst::buffer::AudioBuffer;
use vst::plugin::{Category, Info, Plugin, PluginParameters};
use vst::util::AtomicFloat;

use std::sync::Arc;
use std::collections::VecDeque;

use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use rand_xoshiro::rand_core::RngCore;

use rustfft::algorithm::Radix4;
use rustfft::FFT;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

mod compute;

const SIZE: usize = 256;   // must be of the form 2^k >= 32
const L_DIV: f32 = 1.0/ (SIZE as f32);
const _2_DIV_3: f32 = 2.0/3.0;

struct Effect {
    // Store a handle to the plugin's parameter object.
    params: Arc<EffectParameters>,

    // meta variables
    rng: Xoshiro256Plus,
    sr: f32,
    scale: f64,     // scaling factor for sr independence of integrals

    // FFT engines
    fft: Radix4<f32>,
    ifft: Radix4<f32>,

    // other variables
    env: VecDeque<f32>,         // envelope follower

    // FFT variables
    // sample counter (counts to SIZE/4)
    count: usize,

    // time-domain variables
    xl_samp: VecDeque<Complex<f32>>,
    xr_samp: VecDeque<Complex<f32>>,

    // frequency-domain variables
    xl_bins: Vec<Complex<f32>>,
    xr_bins: Vec<Complex<f32>>,
    // TODO: do the variables for the freezing thing. Deltas and non-deltas

    // output buffer
    yl_samp: VecDeque<f32>,
    yr_samp: VecDeque<f32>,
}

struct EffectParameters {
    freeze: AtomicFloat,
    diffuse: AtomicFloat,
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

            // misc variables
            env: VecDeque::from(vec![0.0; SIZE]),

            // FFT variables
            count: 0,
            xl_samp: VecDeque::from(vec![Complex::zero(); SIZE]),     // pre-fill padding buffer
            xr_samp: VecDeque::from(vec![Complex::zero(); SIZE]),     // pre-fill padding buffer
            xl_bins: vec![Complex::zero(); SIZE],
            xr_bins: vec![Complex::zero(); SIZE],
            xl_bins_in_dly_1: vec![Complex::zero(); SIZE],
            xr_bins_in_dly_1: vec![Complex::zero(); SIZE],
            d_xl_bins_out_dly_1: vec![Complex::zero(); SIZE],
            d_xr_bins_out_dly_1: vec![Complex::zero(); SIZE],
            yl_samp: VecDeque::from(vec![0.0; SIZE]),
            yr_samp: VecDeque::from(vec![0.0; SIZE]),
        }
    }
}

impl Default for EffectParameters {
    fn default() -> EffectParameters {
        EffectParameters {
            freeze: AtomicFloat::new(0.0),
            diffuse: AtomicFloat::new(0.0),
        }
    }
}

// All plugins using `vst` also need to implement the `Plugin` trait.  Here, we
// define functions that give necessary info to our host.
impl Plugin for Effect {
    fn get_info(&self) -> Info {
        Info {
            name: "SPEC_FREEZE".to_string(),
            vendor: "Flux-Audio".to_string(),
            unique_id: 72763875,
            version: 010,
            inputs: 2,
            outputs: 2,
            // This `parameters` bit is important; without it, none of our
            // parameters will be shown!
            parameters: 2,
            category: Category::Generator,
            ..Default::default()
        }
    }

    fn set_sample_rate(&mut self, rate: f32){
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
            let freeze = self.params.freeze.get();
            let diffuse = self.params.diffuse.get();

            // === envelope follower ===========================================

            // === buffering inputs ============================================
            // apply windowing function based on count and push to buffer
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
            if self.count >= SIZE/4{
                self.count = 0;

                let mut xl_fft: Vec<Complex<f32>> = Vec::with_capacity(SIZE);
                let mut xr_fft: Vec<Complex<f32>> = Vec::with_capacity(SIZE);
                for i in 0..SIZE{
                    xl_fft.push(self.xl_samp[i]);
                    xr_fft.push(self.xr_samp[i]);
                }

                // apply windowing function
                for i in 0..SIZE{
                    xl_fft[i] = compute::win_hann(xl_fft[i], i, L_DIV);
                    xr_fft[i] = compute::win_hann(xr_fft[i], i, L_DIV);
                }

                self.fft.process(&mut xl_fft, &mut self.xl_bins);
                self.fft.process(&mut xr_fft, &mut self.xr_bins);

                // normalize
                self.xl_bins.iter_mut().for_each(|elem| *elem *= L_DIV);
                self.xr_bins.iter_mut().for_each(|elem| *elem *= L_DIV);

                // clean up
                // self.xl.clear();
                // self.xr.clear();

                // === spectral freeze =========================================
                // process bins
                for i in 0..SIZE{
                    
                    // amplitude
                    let r_left_res = r_left*(1.0 - freeze) + r_left_p*freeze;
                    let r_right_res = r_right*(1.0 - freeze) + r_right_p*freeze;
                    
                    // phase
                    let mut rand_aux_1 = self.rng.next_u64() as f32 / u64::MAX as f32;
                    let mut rand_aux_2 = self.rng.next_u64() as f32 / u64::MAX as f32;
                    rand_aux_1 = (rand_aux_1 - 0.5)*diffuse;
                    rand_aux_2 = (rand_aux_2 - 0.5)*diffuse;

                    // save result
                    self.xl_bins[i] = Complex::from_polar(r_left_res, phi_left_res);
                    self.xr_bins[i] = Complex::from_polar(r_right_res, phi_right_res);

                    // apply delay
                }
                

                // === inverse FFT & deinterlacing =============================

                // inverse fft
                let mut xl_samp_i: Vec<Complex<f32>> = vec![Complex::zero(); SIZE];
                let mut xr_samp_i: Vec<Complex<f32>> = vec![Complex::zero(); SIZE];
                self.ifft.process(&mut self.xl_bins, &mut xl_samp_i);
                self.ifft.process(&mut self.xr_bins, &mut xr_samp_i);

                // apply window to output (total window = hann^2)
                for i in 0..SIZE{
                    xl_samp_i[i] = compute::win_hann(xl_samp_i[i], i, L_DIV);
                    xr_samp_i[i] = compute::win_hann(xr_samp_i[i], i, L_DIV);

                    // sum output into output buffer
                    let auxl = self.yl_samp.pop_front().unwrap();
                    let auxr = self.yr_samp.pop_front().unwrap();
                    self.yl_samp.push_back(auxl + xl_samp_i[i].re*_2_DIV_3);
                    self.yr_samp.push_back(auxr + xr_samp_i[i].re*_2_DIV_3);
                    // NOTE: all samples are popped and pushed, so they return
                    // to their initial positions. Ordering is not affected
                }
            }

            // === output ======================================================
            *left_out = match self.yl_samp.pop_front(){
                Some(val) => val,
                None => 0.0,
            };
            *right_out = match self.yr_samp.pop_front(){
                Some(val) => val,
                None => 0.0,
            };
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
            // Pendulum parameters
            //0 => self.len_ratio.get(),
            0 => self.freeze.get(),
            1 => self.diffuse.get(),
            _ => 0.0,
        }
    }

    // the `set_parameter` function sets the value of a parameter.
    fn set_parameter(&self, index: i32, val: f32) {
        #[allow(clippy::single_match)]
        match index {
            // Pendulum parameters
            //0 => self.len_ratio.set(val),
            0 => self.freeze.set(val),
            1 => self.diffuse.set(val),
            _ => (),
        }
    }

    // This is what will display underneath our control.  We can
    // format it into a string that makes the most since.
    fn get_parameter_text(&self, index: i32) -> String {
        match index {
            // Pendulum parameters
            //0 => format!("L1: {:.2}, L2: {:.2}", 1.0-self.len_ratio.get(), self.len_ratio.get()),
            0 => format!("{:.2}", self.freeze.get()),
            1 => format!("{:.2}", self.diffuse.get()),
            _ => "".to_string(),
        }
    }

    // This shows the control's name.
    fn get_parameter_name(&self, index: i32) -> String {
        match index {
            //0 => "L1 <=> L2",
            0 => "freeze",
            1 => "diffuse",
            _ => "",
        }
        .to_string()
    }
}

// This part is important! Without it, our plugin won't work.
plugin_main!(Effect);