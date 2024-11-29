#pragma once

#define DRAW_WORLD_AXIS 0

#define ASSET 7
#define VIEW 0

#if ASSET == 0
#define ASSET_PATH "models/sponza_tiger/Sponza.gltf"

#elif ASSET == 1
#define ASSET_PATH "models/sponza_sphere/Sponza.gltf"

#elif ASSET == 2
#define ASSET_PATH "models/sponza_multi_blas_opaque/Sponza.gltf"

#elif ASSET == 3
#define ASSET_PATH "models/sponza_opaque/Sponza.gltf"

#elif ASSET == 4
#define ASSET_PATH "models/sponza_multi_blas_transparent/Sponza.gltf"

#elif ASSET == 5
#define ASSET_PATH "models/sponza_transparent/Sponza.gltf"

#elif ASSET == 6
#define ASSET_PATH "models/cornellBox/CornellBox.gltf"

#elif ASSET == 7
#define ASSET_PATH "models/checkerboard/CheckerBoard.gltf"
#endif

#define CUBEMAP_TEXTURE_PATH "cubeMapTextures/blueSky.ktx"

#define LIGHT_OFFSET 4