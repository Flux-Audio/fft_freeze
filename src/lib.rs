#![feature(tau_constant)]
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

const SIZE: usize = 4096;   // must be of the form 2^k >= 32
const L_DIV: f32 = 1.0/ (SIZE as f32);
const _NORM: f32 = 2.0/3.0;

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
    xl_bins_z1: Vec<Complex<f32>>,
    xr_bins_z1: Vec<Complex<f32>>,
    yl_bins_z1: Vec<Complex<f32>>,
    yr_bins_z1: Vec<Complex<f32>>,
    yl_bins_z2: Vec<Complex<f32>>,
    yr_bins_z2: Vec<Complex<f32>>,
    yl_bins_z3: Vec<Complex<f32>>,
    yr_bins_z3: Vec<Complex<f32>>,
    yl_bins_z4: Vec<Complex<f32>>,
    yr_bins_z4: Vec<Complex<f32>>,
    yl_bins_z: Vec<Vec<Complex<f32>>>,
    yr_bins_z: Vec<Vec<Complex<f32>>>,
    d_yl_phi_z1: Vec<f32>,
    d_yr_phi_z1: Vec<f32>,

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

            // pre-fill misc variables
            env: VecDeque::from(vec![0.0; SIZE]),

            // pre-fill FFT variables
            count: 0,
            xl_samp: VecDeque::from(vec![Complex::zero(); SIZE]),
            xr_samp: VecDeque::from(vec![Complex::zero(); SIZE]),
            xl_bins_z1: vec![Complex::zero(); SIZE],
            xr_bins_z1: vec![Complex::zero(); SIZE],
            yl_bins_z1: vec![Complex::zero(); SIZE],
            yr_bins_z1: vec![Complex::zero(); SIZE],
            yl_bins_z2: vec![Complex::zero(); SIZE],
            yr_bins_z2: vec![Complex::zero(); SIZE],
            yl_bins_z3: vec![Complex::zero(); SIZE],
            yr_bins_z3: vec![Complex::zero(); SIZE],
            yl_bins_z4: vec![Complex::zero(); SIZE],
            yr_bins_z4: vec![Complex::zero(); SIZE],
            yl_bins_z: vec![vec![Complex::zero(); SIZE]; 32],
            yr_bins_z: vec![vec![Complex::zero(); SIZE]; 32],
            d_yl_phi_z1: vec![0.0; SIZE],
            d_yr_phi_z1: vec![0.0; SIZE],
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
            let freeze = self.params.freeze.get().sqrt();
            let diffuse = freeze*self.params.diffuse.get();
            let freeze_gain = 1.0 + 3.0*freeze;

            // === envelope follower ===========================================

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
            // if buffer has advanced by 12.5% (87.5% overlap), perform FFT
            // TODO: run fft processing on separate thread, join previous fft
            // at start of next fft frame
            if self.count >= SIZE/4{
                self.count = 0;

                // deep-copy input buffers into fft input
                let mut xl_fft: Vec<Complex<f32>> = Vec::with_capacity(SIZE);
                let mut xr_fft: Vec<Complex<f32>> = Vec::with_capacity(SIZE);
                for i in 0..SIZE{
                    xl_fft.push(self.xl_samp[i]);
                    xr_fft.push(self.xr_samp[i]);
                }

                // apply windowing function twice
                for i in 0..SIZE{
                    xl_fft[i] = compute::win_hann(xl_fft[i], i, L_DIV);
                    xr_fft[i] = compute::win_hann(xr_fft[i], i, L_DIV);
                }

                // forward fft
                let mut xl_bins: Vec<Complex<f32>> = vec![Complex::zero(); SIZE];
                let mut xr_bins: Vec<Complex<f32>> = vec![Complex::zero(); SIZE];
                self.fft.process(&mut xl_fft, &mut xl_bins);
                self.fft.process(&mut xr_fft, &mut xr_bins);

                // normalize
                xl_bins.iter_mut().for_each(|elem| *elem *= L_DIV);
                xr_bins.iter_mut().for_each(|elem| *elem *= L_DIV);

                // === spectral freeze =========================================
                // interpolate previous output's amplitude with input amplitude
                // interpolate previous output's delta phase with input delta phase
                // add interpolated delta phase to previous phase
                for i in 0..SIZE{

                    /*
                    // average bins, first get coprime set of delay taps
                    let yl_p = Complex::new(self.yl_bins_z[2][i].re, self.yl_bins_z[4][i].im);
                    let yr_p = Complex::new(self.yr_bins_z[2][i].re, self.yr_bins_z[4][i].im);

                    xl_bins[i] = xl_bins[i]*(1.0 - freeze) + yl_p*freeze;
                    xr_bins[i] = xr_bins[i]*(1.0 - freeze) + yr_p*freeze;

                    // apply phase randomization
                    let mut rand_aux_1 = compute::randf(&mut self.rng);
                    let mut rand_aux_2 = compute::randf(&mut self.rng);
                    rand_aux_1 = (rand_aux_1 - 0.5)*diffuse*std::f32::consts::PI;
                    rand_aux_2 = (rand_aux_2 - 0.5)*diffuse*std::f32::consts::PI;
                    
                    let (yl_r, yl_phi) = xl_bins[i].to_polar();
                    let (yr_r, yr_phi) = xr_bins[i].to_polar();

                    xl_bins[i] = Complex::from_polar(yl_r, yl_phi + rand_aux_1);
                    xr_bins[i] = Complex::from_polar(yr_r, yr_phi + rand_aux_2);

                    // update delay line
                    for j in 0..31{
                        self.yl_bins_z[31 - j][i] = self.yl_bins_z[30 - j][i];
                        self.yr_bins_z[31 - j][i] = self.yr_bins_z[30 - j][i];
                    }
                    self.yl_bins_z[0][i] = xl_bins[i];
                    self.yr_bins_z[0][i] = xr_bins[i];

                    */
                    
                    
                    // convert bins to polar
                    // input bins
                    let (xl_r, xl_phi) = xl_bins[i].to_polar();
                    let (xr_r, xr_phi) = xr_bins[i].to_polar();
                    // previous input bins
                    let (_, xl_phi_z1) = self.xl_bins_z1[i].to_polar();
                    let (_, xr_phi_z1) = self.xr_bins_z1[i].to_polar();
                    // previous output bins
                    let (yl_r_z1, yl_phi_z1) = self.yl_bins_z[0][i].to_polar();
                    let (yr_r_z1, yr_phi_z1) = self.yr_bins_z[0][i].to_polar();

                    // update previous input bins
                    self.xl_bins_z1[i] = xl_bins[i];
                    self.xr_bins_z1[i] = xr_bins[i];

                    // get input deltas
                    let d_xl_phi = xl_phi - xl_phi_z1;
                    let d_xr_phi = xr_phi - xr_phi_z1;
                    
                    // amplitude freezing
                    let yl_r = xl_r*(1.0 - freeze) + yl_r_z1*freeze;
                    let yr_r = xr_r*(1.0 - freeze) + yr_r_z1*freeze;
                    
                    // generate random phase offsets
                    let mut rand_aux_1 = compute::randf(&mut self.rng);
                    let mut rand_aux_2 = compute::randf(&mut self.rng);
                    rand_aux_1 = (rand_aux_1 - 0.5)*diffuse*std::f32::consts::TAU;
                    rand_aux_2 = (rand_aux_2 - 0.5)*diffuse*std::f32::consts::TAU;

                    // phase freeze
                    /*
                    let d_yl_phi = d_xl_phi*(1.0 - freeze) + self.d_yl_phi_z1[i]*freeze*0.75;
                    let d_yr_phi = d_xr_phi*(1.0 - freeze) + self.d_yr_phi_z1[i]*freeze*0.75;
                    */
                    let yl_phi = xl_phi*(1.0 - freeze) + yl_phi_z1*freeze + rand_aux_1;
                    let yr_phi = xr_phi*(1.0 - freeze) + yr_phi_z1*freeze + rand_aux_2;

                    // save result
                    xl_bins[i] = Complex::from_polar(yl_r, yl_phi);
                    xr_bins[i] = Complex::from_polar(yr_r, yr_phi);

                    // update previous output bins
                    self.yl_bins_z[3][i] = self.yl_bins_z[2][i];
                    self.yl_bins_z[2][i] = self.yl_bins_z[1][i];
                    self.yl_bins_z[1][i] = self.yl_bins_z[0][i];
                    self.yl_bins_z[0][i] = xl_bins[i];

                    self.yr_bins_z[3][i] = self.yr_bins_z[2][i];
                    self.yr_bins_z[2][i] = self.yr_bins_z[1][i];
                    self.yr_bins_z[1][i] = self.yr_bins_z[0][i];
                    self.yr_bins_z[0][i] = xr_bins[i];
                    //self.d_yl_phi_z1[i] = d_yl_phi;
                    //self.d_yr_phi_z1[i] = d_yr_phi;
                }

                // === inverse FFT =============================================
                // inverse fft
                let mut xl_samp_i: Vec<Complex<f32>> = vec![Complex::zero(); SIZE];
                let mut xr_samp_i: Vec<Complex<f32>> = vec![Complex::zero(); SIZE];
                self.ifft.process(&mut xl_bins, &mut xl_samp_i);
                self.ifft.process(&mut xr_bins, &mut xr_samp_i);

                // apply window to output twice (total window = hann^4)
                for i in 0..SIZE{
                    xl_samp_i[i] = compute::win_hann(xl_samp_i[i], i, L_DIV);
                    xr_samp_i[i] = compute::win_hann(xr_samp_i[i], i, L_DIV);

                    // sum output into output buffer
                    let auxl = self.yl_samp.pop_front().unwrap();
                    let auxr = self.yr_samp.pop_front().unwrap();
                    self.yl_samp.push_back(auxl + xl_samp_i[i].re*_NORM);
                    self.yr_samp.push_back(auxr + xr_samp_i[i].re*_NORM);
                    // NOTE: all samples are popped and pushed, so they return
                    // to their initial positions. Ordering is not affected
                }
            }

            // === output ======================================================
            // compensate for freeze gain loss
            *left_out = match self.yl_samp.pop_front(){
                Some(val) => val*freeze_gain,
                None => 0.0,
            };
            *right_out = match self.yr_samp.pop_front(){
                Some(val) => val*freeze_gain,
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