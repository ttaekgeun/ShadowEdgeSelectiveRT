/*
 * Sogang Univ, Graphics Lab, 2024
 * 
 * Vulkan Full Raytracing
 * 
 */

#version 460
#extension GL_EXT_ray_tracing : enable

struct RayPayload {
	vec3 color;
	float distance;
	vec3 normal;
	uint effectFlag;
};

layout(location = 0) rayPayloadInEXT RayPayload rayPayload;
layout(binding = 5, set = 0) uniform samplerCube samplerColor;

void main()
{
	vec3 worldRayDir = normalize(gl_WorldRayDirectionEXT);
    vec3 envColor = textureLod(samplerColor, worldRayDir, 0.0).rgb;
	rayPayload.color = envColor;
	rayPayload.distance = -1.0f;
	rayPayload.normal = vec3(0.0f);
	rayPayload.effectFlag = 0x0;
}