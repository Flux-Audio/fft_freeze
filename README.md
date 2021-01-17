# FFT_FREEZE v0.1.0

## Intallation
_**Disclaimer:** this plugin will only work on 64-bit Windows machines!_ \
Download the `.dll` file in the `bin\` directory and place it into your DAW's VST
folder.
## Compiling the source code
_**Disclaimer:** you don't need to compile the source code if you just want to use
the plugin, just download the `.dll`_ \
Make sure you have Cargo installed on your computer (the Rust compiler). Then in
the root of the repository run `cargo build`. Once Cargo is done building, there
should be a `FFT_FREEZE_v0_1_0.dll` file in the newly created `debug/` directory.
Place this file into your DAW's VST folder.

# What is FFT_FREEZE ?
FFT_FREEZE is a spectral effect, which allows you to freeze a small slice of
sound and smear it over time. What makes it special is that the incoming sound
is spliced into 4096 frequency bands (called bins), each of which is averaged over
time to produce the freezing effect.

FFT_FREEZE is an attempt at reverse-engineering the amazing plugin ["SpecOps" by
Unfiltered Audio](https://www.unfilteredaudio.com/products/specops); specifically 
the "speed" knob. While my attempt was somewhat succesful, I highly recomend checking 
out the original, which has tons of other interesting spectral effects.

This implementation of the effect is quite resonant at a period length of
1024 samples (about 43Hz at a 44.1kHz sample-rate). To mitigate this a "diffusion"
parameter was added to the original effect. This randomizes the phase of each bin,
smoothing out the resonant artifacts, compromising pitch accuracy for timbral
accuracy. With 0% diffusion, the effect resembles Karplus-Strong synthesis.

A simple envelope follower was added, this is to re-create my favorite patch in
SpecOps, where you use the envelope follower to modulate the speed knob. You can
apply positive or negative modulation to the freeze knob, and adjust the response
time of the envelope (both attack and release) with the time knob.

For the nerds reading this, the spectral analysis is performed with a
Fast-Furier-Transform (Bjarke, if you're reading this, hello). With 75% overlap
4096-sample chunks, so an effective latency of 1024 samples. Each chunk is windowed
with a Hann window (squared sine) before the FFT transformation, and is windowed
again after the iFFT transformation, to remove edge-artifacts. This effectively
means we're using a squared Hann window. Outputs of the iFFT are overlapped and 
added, so that the gain loss due to windowing disappears. With no freezing or 
diffusion, the effect is almost completely transparent. 

# Controls Explained
- Freeze: how much spectral averaging is applied, e.g. how much the audio is
smeared. At 100% the audio is completely frozen.
- Diffusion: how much phase randomization is applied to each bin. Note that left
and right channels have separate random values, so this effect also widens the
stereo field, and can sound a bit like reverb.
- Envelope Amount: how much the freeze knob is modulated by the envelope follower.
Negative values move the freeze knob to the left, and positive to the right.
- Envelope Time: how much attack and release the envelope follower has. 0% is 
instantaneous (as fast as technically possible, i.e. an arbitrary value between
0 and 1024 samples, based on audio-slice allignment), 100% is infinite, i.e. the
envelope itself is frozen. Note that at 100% the envelope might not be 100% closed
but rather in any arbitrary state, this might have some interesting use cases.

# Planned Features
- Multiple windowing functions: the squared Hann function, while very clean for
no processing, becomes very resonant at 100% freeze. Other windowing functions
can be considered, and might become part of a list of windowing choices. Note
that windowing functions exist in a trade-off between pitch accuracy and timbral
accuracy. The pick will be from this list:
    - Hann with 50% overlap (moderate side-lobes)
    - Hann with 75% overlap
    - ~~Boxcar with 50% overlap:~~ this was tried and sounds very harsh
    - ~~Boxcar with 75% overlap:~~ this was tried and sounds very harsh
    - Triangular with 50% overlap (heavy side-lobes)
    - Triangular with 75% overlap
    - Blackman with 75% overlap (moderate side-lobes)
    - Hamming with 50% overlap (heavy side-lobes)
    - Hamming with 75% overlap
    - Nuttal with 75% overlap (light side-lobes)
    - Flattop with 87.5% overlap (negative side-lobes, spacey)
    - Tukey with 75% overlap (good compromise between Hann and Boxcar)
- Multiple freezing modes: the various freeze modes from SpecOps can be easily
added. These include:
    - Glitchy freeze: freeze% controls chance of a bin to freeze completely from
    one frame to the next.
    - Random freeze: freeze% controls change of entire frame to freeze completely
    from one frame to the next.
    - ~~Threshold freeze~~: won't be implemented, because the envelope follower achieves
    the same effect.
    - Resonant freeze: freeze% is proportional to bin loudness, so the resonant
    frequencies ring out for longer.
    - Fuzzy freeze: diffusion randomization is applied to amplitude, as well as
    phase.
- Feedback EQ: allows for EQ'ing the feedback frame, EQ'ing is achieved by attenuating
a bin based on its index value. Low pass, high pass, band pass and band stop filters
available. EQ curve can be mixed, turning LP and HP into shelf functions at 50%
and BP and BS filters into peak EQ. The notch filter in particular would be interesting
to achieve the "blackhole" preset from FabFilter Pro-R.
- Effect mask: allows to specify the bin index range where the effect is applied,
like in SpecOps.
## Planned plugins using the same architecture
- FFT_SHIMMER: uses the FFT engine for diffusion (blurring) and for pitch shifting,
has an added delay network in the feedback path, as well as a traditional time-domain
vibrato-like effect for a more chorus-y sound.
- FFT_DYNAMO: uses the FFT engine for spectral dynamic effects. Has these features:
    - Spectral compander: pushes all bin loudnesses towards a constant value, so
    the ideal output has the same energy per frequency as pink noise at -6dB peak.
    At negative levels of compander, bins are pushed away from the ideal -6dB.
    - Attack and release for all gain changes applied to bins.
- FFT_SMEAR: various methods of smearing bins with nearby bins. This is implemented
with a variety of filters:
    - Box average
    - Gaussian average
    - Median of amplitudes
    - Min of amplitdes
    - Max of amplitudes
    - Sharpen
- FFT_PHASER: a phaser implemented with FFT. Each bin has an associated phasor
which modulates the phase. The phasors can be offset from each other in phase and 
frequency, creating various interesting effects. Each bin has also
an associated sine wave oscillator, which modulates the amplitude, again phase,
amplitude and frequency of the oscillator can be changed.
- FFT_FUCK: a collection of weird glitchy spectral effects, beyond what SpecOps
can do, including a more in-depth take on the "mp3-ify" effect, which applies
bin-wise bit reduction. Effects where FFT and iFFT sizes don't match and the input
is mapped in various interesting ways to fit the size of the iFFT. A bin-wise
saturation effect, bin clustering in various ways (similar bins are combined),
bin sorting or partial sorting (a halted merge sort or quick-sort on bins by
amplitude), imaginary-real swapping. Various ways of permutating bins. Various
ways to smear bins (averaging with neighbors) or other convolutions borrowed from
the image processing world. Speaking of image processing:
- WAV_TO_PNG: a stand-alone application that converts a wave file into a png.
It collapses the 4-dimensions of the FFT (left_phase, right_phase, left_amplitude,
right_amplitude) into three dimensions (mid_phase, mid_amp, amp_panning) and then
turns each bin into a pixel, where the three dimensions are represented by RGB
values. The full wave file is thus converted into a matrix of pixels, and is
stored as a PNG, it also allows the reverse to happen. This can be used to
experiment with image filters on audio as well as using convolutional neural
networks for audio processing.
Specifically the conversion happens as follows:
    - mid_amp: lightness
    - pan: red-green balance
    - phase: yellow-blue balance