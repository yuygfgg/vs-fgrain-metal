# vs-fgrain-metal
[Realistic Film Grain Rendering](https://www.ipol.im/pub/art/2017/192/) for VapourSynth, implemented in Metal. A port of https://github.com/AmusementClub/vs-fgrain-cuda/
## Usage
Prototype:

`core.fgrain_metal.Add(clip clip[, int num_iterations = 800, float grain_radius_mean = 0.1, float grain_radius_std=0.0, float sigma = 0.8, int seed = 0])`

Currently only grays is supported.

## Compilation

Build with Xcode and copy dylib and default.metallib to /usr/local/lib/vapoursynth
