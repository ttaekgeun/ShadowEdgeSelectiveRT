# Sogang University, Graphics Lab.

## Assets

\\\163.239.24.71\GraphicsLab\Project\2024_Samsung_Project\SamsungVulkanRT_assets


## Building

1. Clone this repository.
2. Copy the 'assets' folder to your cloned root.
3. Make build system using cmake.
4. Build using Visual Studio IDE.

## Running

Once built, examples can be run from the bin directory. The list of available command line options can be brought up with `--help`:
```
 -v, --validation: Enable validation layers
 -br, --benchruntime: Set duration time for benchmark mode in seconds
 -vs, --vsync: Enable V-Sync
 -w, --width: Set window width
 -f, --fullscreen: Start in fullscreen mode
 --help: Show help
 -h, --height: Set window height
 -bt, --benchframetimes: Save frame times to benchmark results file
 -s, --shaders: Select shader type to use (glsl or hlsl)
 -b, --benchmark: Run example in benchmark mode
 -g, --gpu: Select GPU to run on
 -bf, --benchfilename: Set file name for benchmark results
 -gl, --listgpus: Display a list of available Vulkan devices
 -bw, --benchwarmup: Set warmup time for benchmark mode in seconds
```

Note that some examples require specific device features, and if you are on a multi-gpu system you might need to use the `-gl` and `-g` to select a gpu that supports them.

## Shaders

Vulkan consumes shaders in an intermediate representation called SPIR-V. This makes it possible to use different shader languages by compiling them to that bytecode format. The primary shader language used here is [GLSL](shaders/glsl).


## Examples

#### [Vulkan Ray Tracing Main](projects/VulkanRTMain/)



## References

We refer to the repository shown below.

https://github.com/SaschaWillems/Vulkan.git


## Camera Interface
1. 알파벳 키
-W : forward
-S : backward
-A : left
-D : right
-E : up
-Q : down
-R: reset original position, rotation

2. Mouse
- left button: pitch + yaw
- right button: roll(갸우뚱)
- middle button: move x,y축
- wheel : forward, backward

3. 방향키(mouse left button 기능과 같음)
→ : yaw +
← : yaw - 
↓ : pitch -
↑ : pitch +

4. 키보드, 마우스 속도 제어
<W,A,S,D,E,Q>
- : 감소
+ : 증가

<Mouse left,middle,right>
{ : 감소
} : 증가

5. 카메라 version
version: {SG_camera, firstperson, lookat}
F2: SG_camera 설정이 안되어있을 경우, runtime에서 변경가능
