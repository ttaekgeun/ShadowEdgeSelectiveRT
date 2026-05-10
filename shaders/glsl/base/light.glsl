/*
 * Sogang University
 *
 * Graphics Lab, 2024
 *
 */
#define NUM_OF_LIGHTS 1;
#define SHADOW_RAY_ON true;

//const int POINT_LIGHT = 0;
//const int SPOT_LIGHT = 1;
//const int DIRECTIONAL_LIGHT = 2;

struct LightInfo {
	vec3 color;
	float spotCutoff;
	vec3 spotDirection;
};

//struct SpotLight {
//	double innerConeAngle;
//	double outerConeAngle;
//};

LightInfo lightInfo = LightInfo(vec3(1.0f, 1.0f, 1.0f), 180.0f, vec3(0.0f, 0.0f, 0.0f));

//struct Light {
//	vec3 color;
//	//int type;
//	//SpotLight spot;
//	vec4 position;
//	vec4 rotation;
//};

// Attenuation
struct AttenuationFactor {
	float k0;
	float k1;
	float k2;
};
AttenuationFactor attenuationFactor = AttenuationFactor(1.0f, 0.0001f, 0.000001f);