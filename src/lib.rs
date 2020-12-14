#[macro_use]
extern crate vst;

use vst::buffer::AudioBuffer;
use vst::plugin::{Category, Info, Plugin, PluginParameters};
use vst::util::AtomicFloat;

use std::sync::Arc;
use std::collections::VecDeque;

use rustfft::algorithm::Radix4;
use rustfft::FFT;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

mod compute;

const SIZE: usize = 4096;   // must be of the form 2^k >= 32
const PADDING: usize = SIZE/32;
const ACTIVE: usize = SIZE - 2*PADDING;

struct Effect {
    // Store a handle to the plugin's parameter object.
    params: Arc<EffectParameters>,

    // meta variables
    sr: f32,
    scale: f64,     // scaling factor for sr independence of integrals

    // FFT engines
    fft: Radix4<f32>,
    ifft: Radix4<f32>,

    // FFT variables
    xl_1: Vec<Complex<f32>>,
    xr_1: Vec<Complex<f32>>,
    xl_2: Vec<Complex<f32>>,
    xr_2: Vec<Complex<f32>>,
    env: VecDeque<f32>,
    fft_l: Vec<Complex<f32>>,
    fft_r: Vec<Complex<f32>>,
    p_ifft_l: Vec<Complex<f32>>,
    p_ifft_r: Vec<Complex<f32>>,
    ifft_l: Vec<Complex<f32>>,
    ifft_r: Vec<Complex<f32>>,
    yl: VecDeque<Complex<f32>>,
    yr: VecDeque<Complex<f32>>,
    count1: usize,
    count2: usize,
    x_1_enable: bool,
    x_2_enable: bool,
}

struct EffectParameters {
    freeze: AtomicFloat,
}

impl Default for Effect {
    fn default() -> Effect {
        Effect {
            params: Arc::new(EffectParameters::default()),

            // meta variables
            sr: 44100.0,
            scale: 1.0,

            // FFT engines
            fft: Radix4::new(SIZE, false),
            ifft: Radix4::new(SIZE, true),

            // FFT variables
            xl_1: vec![Complex::zero(); PADDING],     // pre-fill padding buffer
            xr_1: vec![Complex::zero(); PADDING],     // pre-fill padding buffer
            xl_2: Vec::with_capacity(SIZE),
            xr_2: Vec::with_capacity(SIZE),
            env: VecDeque::from(vec![0.0; SIZE]),
            fft_l: vec![Complex::zero(); SIZE],
            fft_r: vec![Complex::zero(); SIZE],
            ifft_l: vec![Complex::zero(); SIZE],
            ifft_r: vec![Complex::zero(); SIZE],
            p_ifft_l: vec![Complex::zero(); SIZE],
            p_ifft_r: vec![Complex::zero(); SIZE],
            yl: VecDeque::from(vec![Complex::zero(); SIZE]),
            yr: VecDeque::from(vec![Complex::zero(); SIZE]),
            count1: PADDING,     // pre-filled padding buffer
            count2: 0,
            x_1_enable: true,
            x_2_enable: false,
        }
    }
}

impl Default for EffectParameters {
    fn default() -> EffectParameters {
        EffectParameters {
            freeze: AtomicFloat::new(0.0),
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
            parameters: 1,
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

            // === envelope follower ===========================================

            // === buffer interlacing ==========================================
            // the incoming audio is divided into ACTIVE-sized blocks, but each 
            // block contains an additional PADDING samples at the beginning and
            // end that overlap with the next block, bringing the total size
            // of each block to SIZE. This is done to avoid FFT edge artifacts

            // enable correct buffers
            if self.count1 == ACTIVE{ self.x_2_enable = true; }
            if self.count2 == ACTIVE{ self.x_1_enable = true; }
            // invariants:  count1 >= ACTIVE =>  x_2_enable
            //              count1 <  ACTIVE => !x_2_enable
            //              count2 >= ACTIVE =>  x_1_enable
            //              count2 <  ACTIVE => !x_1_enable 

            // load samples into enabled buffers
            if self.x_1_enable{
                self.xl_1.push(Complex::new(*left_in, 0.0));
                self.xr_1.push(Complex::new(*right_in, 0.0));
                self.count1 += 1;
            }
            if self.x_2_enable{
                self.xl_2.push(Complex::new(*left_in, 0.0));
                self.xr_2.push(Complex::new(*right_in, 0.0));
                self.count1 += 2;
            }

            // disable full buffers & prepare for fft
            let mut fft_ready = false;
            let mut xl_fft: Vec<Complex<f32>> = Vec::new();
            let mut xr_fft: Vec<Complex<f32>> = Vec::new();
            if self.count1 == SIZE{ 
                self.x_1_enable = false;
                xl_fft = self.xl_1.to_vec();
                xr_fft = self.xr_1.to_vec();
                self.count1 = 0;
                fft_ready = true;
            } else if self.count2 == SIZE{
                self.x_2_enable = false;
                xl_fft = self.xl_2.to_vec();
                xr_fft = self.xr_2.to_vec();
                self.count2 = 0;
                fft_ready = true;
            }
            // invariants:  !x_1_enable => x_2_enable
            //              !x_2_enable => x_1_enable
            //              count1 <= SIZE
            //              count2 <= SIZE

            // === perform FFT on inputs =======================================
            // if buffer is full, perform FFT
            if fft_ready{
                self.fft.process(&mut xl_fft, &mut self.fft_l);
                self.fft.process(&mut xr_fft, &mut self.fft_r);

                // normalize
                self.fft_l.iter_mut().for_each(|elem| *elem /= SIZE as f32);
                self.fft_r.iter_mut().for_each(|elem| *elem /= SIZE as f32);

                // clean up
                // self.xl.clear();
                // self.xr.clear();

                // === spectral freeze =========================================
                for i in 0..SIZE{
                    self.ifft_l[i] = {
                        let (x_amp, x_phi) = self.fft_l[i].to_polar();
                        let (p_amp, p_phi) = self.p_ifft_l[i].to_polar();
                        let y_amp = x_amp * (1.0 - freeze) + p_amp * freeze;
                        let y_phi = x_phi * (1.0 - freeze) + p_phi * freeze;
                        Complex::from_polar(y_amp, y_phi)
                    };
                    self.ifft_r[i] = {
                        let (x_amp, x_phi) = self.fft_r[i].to_polar();
                        let (p_amp, p_phi) = self.p_ifft_r[i].to_polar();
                        let y_amp = x_amp * (1.0 - freeze) + p_amp * freeze;
                        let y_phi = x_phi * (1.0 - freeze) + p_phi * freeze;
                        Complex::from_polar(y_amp, y_phi)
                    };
                    self.p_ifft_l[i] = self.ifft_l[i];
                    self.p_ifft_r[i] = self.ifft_r[i];
                }

                // === inverse FFT & deinterlacing =============================

                // inverse fft
                let mut auxl = vec![Complex::zero(); SIZE];
                let mut auxr = vec![Complex::zero(); SIZE];
                self.ifft.process(&mut self.ifft_l, &mut auxl);
                self.ifft.process(&mut self.ifft_r, &mut auxr);

                // convert to deques
                self.yl = VecDeque::from(auxl);
                self.yr = VecDeque::from(auxr);

                // discard first and last PADDING samples (deinterlace)
                for i in 0..PADDING{
                    self.yl.pop_front();
                    self.yr.pop_front();
                    self.yl.pop_back();
                    self.yr.pop_back();
                }
            }

            // === output ======================================================
            *left_out = match self.yl.pop_front(){
                Some(val) => val.re,
                None => 0.0,
            };
            *right_out = match self.yr.pop_front(){
                Some(val) => val.re,
                None => 0.0,
            };
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
            _ => "".to_string(),
        }
    }

    // This shows the control's name.
    fn get_parameter_name(&self, index: i32) -> String {
        match index {
            //0 => "L1 <=> L2",
            0 => "freeze",
            _ => "",
        }
        .to_string()
    }
}

// This part is important! Without it, our plugin won't work.
plugin_main!(Effect);