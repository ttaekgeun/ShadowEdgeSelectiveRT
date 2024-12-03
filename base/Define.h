#pragma once

#define DRAW_WORLD_AXIS 0

#define ASSET 0
#define VIEW 0

#if ASSET == 0
#define ASSET_PATH "models/sponza/Sponza.gltf"
#endif

#if ASSET == 1
#define ASSET_PATH "models/bistro_exterior/BistroExterior.gltf"
#endif

#define CUBEMAP_TEXTURE_PATH "cubeMapTextures/blueSky.ktx"