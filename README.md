# FFT_FREEZE v0.2.0

***UPDATE NOTICE:** this update breaks compatibility with the previous version
(0.1.0). Backup your previous version and its presets before upgrading, or keep
both versions as separate plugins. Changelist at the bottom of the page.* 

## Intallation
_**Disclaimer:** this plugin will only work on 64-bit Windows machines!_ \
Download the `.dll` file in the `bin\` directory and place it into your DAW's VST
folder.
## Compiling the source code
_**Disclaimer:** you don't need to compile the source code if you just want to use
the plugin, just download the `.dll`_ \
Make sure you have Cargo installed on your computer (the Rust compiler). Then in
the root of the repository run `cargo build`. Once Cargo is done building, there
should be a `FFT_FREEZE_v0_2_0.dll` file in the newly created `debug/` directory.
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

FFT_FREEZE has various different modes of operation. They are explained below.

For the nerds reading this, the spectral analysis is performed with a
Fast-Furier-Transform (Bjarke, if you're reading this, hello). With 75% overlap
4096-sample chunks, so an effective latency of 1024 samples. Each chunk is windowed
with a Hann window (squared sine) before the FFT transformation, and is windowed
again after the iFFT transformation, to remove edge-artifacts. This effectively
means we're using a squared Hann window. Outputs of the iFFT are overlapped and 
added, so that the gain loss due to windowing disappears. With no freezing or 
diffusion, the effect is almost completely transparent. 

## Window Modes
The audio coming in is split into frames, which are processed separately. The
frames overlap, and so a crossfading function is needed to smoothly weave the
frames back together after the processing. This is what the window function does.

The window modes are different shapes of crossfading, each has its advantages and
disadvantages:

- Balanced: Hann window. Good balance between frequency accuracy and time accuracy.
- Smear: Triangular window. Slightly worse time accuracy, better frequency accuracy.
- Clean: Blackmann window. Better time accuracy, worse frequency accuracy.
- Flutter: Broken window, does not overlap correctly, producing a fluttering sound
that might be desirable. Sounds good on drums.
## Freeze Modes
There are various ways of freezing the sound. Each one interacts with the freeze
amount and diffusion faders in interesting ways.

- Normal: crossfades current incoming frame with previously outputted frame. Has
no special interaction with the diffusion slider.
- Glitchy: freeze amount is the probability that any bin will be completely frozen
from one frame to the next. Sounds most unique at 50%, normal freeze at 100%.
- Random: freeze amount is the probability that the entire frame will be completely
frozen. Sounds most unique between 60% and 95%. Normal freeze at 100%.
- Resonant: freeze amount is also dependant on the relative loudness of a bin in
respect to the other bins. This effect is quite subtle (and might be changed in
future versions) but it can be best heard at 50%.
- Spooky: This one randomizes the amplitude as well as the phase when diffusion
is turned up. It didn't work as intended, but ended up sounding even cooler, so
I kept it. There might be an error in there causing it to sound the way it does
so I cannot give a very precise description, other than the fact that it sounds
really spooky.
- Mashup: picks a random freeze mode for each frame.

# Controls Explained
- Freeze: how much spectral averaging is applied, e.g. how much the audio is
smeared. At 100% the audio is completely frozen.
- Diffusion: how much phase randomization is applied to the phase of each bin. Note that left
and right channels have separate random values, so this effect also widens the
stereo field, and can sound a bit like reverb.
- Envelope Amount: how much the freeze knob is modulated by the envelope follower.
Negative values move the freeze knob to the left, and positive to the right.
- Envelope Time: how much attack and release the envelope follower has. 0% is 
instantaneous (as fast as technically possible, i.e. an arbitrary value between
0 and 1024 samples, based on audio-slice allignment), 100% is infinite, i.e. the
envelope itself is frozen. Note that at 100% the envelope might not be 100% closed
but rather in any arbitrary state, this might have some interesting use cases.
- Window Mode: shape of the grains (see previous chapter)
- Freeze Mode: freezing algorithm (see previous chapter)

# Changelist
- Added: new freeze modes.
- Added: new window modes.
- Changed: improved freeze quality.
- Changed: slightly improved performance.
- Changed: now relies on external library for some of the processing.

# Planned Features
- EQ in the feedback loop.
- UI.