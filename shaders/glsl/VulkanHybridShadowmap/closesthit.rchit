/*
 * Sogang University
 *
 * Graphics Lab, 2024
 *
 * This closest hit shader is used for reflection or transmission.
 */

#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "geometrytypes.glsl"
#include "bufferreferences.glsl"
#include "../base/light.glsl"
#include "../base/pbr.glsl"

struct RayPayload {
	vec3 color;
	float distance;
	vec3 normal;
	uint objectID;
	vec3 albedo;
	float metallic;
};

layout(location = 0) rayPayloadInEXT RayPayload rayPayload;
layout(location = 2) rayPayloadEXT bool shadowed;
hitAttributeEXT vec2 attribs;

layout(binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 2) buffer GeometryNodes { GeometryNode nodes[]; } geometryNodes;
layout(binding = 3) uniform uniformBuffer
{
	mat4 viewInverse;
	mat4 projInverse;
	mat4 lightSpace;
	uint frame;
	vec4 lightPos[1];
} ubo;
layout(binding = 10) uniform sampler2D shadowMap;
layout(binding = 11) uniform sampler2D lightMap;
layout(binding = 12) uniform sampler2D textures[];

#include "geometryfunctions.glsl"

const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 );

#define amb 0.1

float textureProj(vec4 shadowCoord, vec2 off)
{
    float shadow = 1.0;

    if (shadowCoord.z > -1.0f && shadowCoord.z < 1.0f) 
    {
        float dist = texture(shadowMap, shadowCoord.st + off).r;
		if (shadowCoord.w > 0.0f && dist < shadowCoord.z)
        {
            shadow = amb;
        }
    }
    return shadow;
}

float filterPCF(vec4 sc)
{
    ivec2 texDim = textureSize(shadowMap, 0);
    float scale = 1.5f;
    float dx = scale * 1.0f / float(texDim.x);
    float dy = scale * 1.0f / float(texDim.y);

    float shadowFactor = 0.0f;
    int count = 0;
    int range = 2;  // PCF 필터링 범위를 증가시켜 부드러운 그림자 처리

    for (int x = -range; x <= range; x++)
    {
        for (int y = -range; y <= range; y++)
        {
            shadowFactor += textureProj(sc, vec2(dx * x, dy * y));
            count++;
        }
    }
    return shadowFactor / count;
}

// Spot light unused
/*vec3 CalculateSpotLight(vec3 pos, int lightIdx, LightInfo lightInfo, vec4 lightPos)
{   // These code below is the old version which doesn't read light informations from glTF file.
    vec3 lightToPixel = normalize(pos - lightPos.xyz);
    float spotFactor = dot(lightToPixel, lightInfo.spotDirection);

    if (spotFactor > cos(lightInfo.spotCutoff * DEG_TO_RAD)) {
        vec3 spotResultColor = pow(spotFactor, 2.0f) * vec3(1.0f);
        return spotResultColor;
    }
    else {
        return vec3(0.0f, 0.0f, 0.0f);
    }
}*/

float LinearizeDepth(float depth)
{
  float n = 1.0f;
  float f = 500.0f;
  float z = depth;
  return (2.0 * n) / (f + n - z * (f - n));	
}

void main()
{
	GeometryNode geometryNode = geometryNodes.nodes[gl_GeometryIndexEXT];

	Triangle tri = unpackTriangle(gl_PrimitiveID, 112);

	vec3 N = tri.normal;
	if (geometryNode.textureIndexNormal > -1) {
		N = CalculateNormal(textures[nonuniformEXT(geometryNode.textureIndexNormal)], N, tri.uv, tri.tangent);
	}
	rayPayload.distance = gl_RayTmaxEXT;
	rayPayload.normal = N;
	rayPayload.objectID = gl_GeometryIndexEXT;

	vec3 pos = tri.pos.xyz;
	vec3 V = normalize((ubo.viewInverse * vec4(0.0f, 0.0f, 0.0f, 1.0f)).xyz - pos);
	float ior = geometryNode.ior;
	vec3 F0 = vec3(0.4);
	float metallic = texture(textures[nonuniformEXT(geometryNode.textureIndexMetallicRoughness)], tri.uv).b;
	float roughness = texture(textures[nonuniformEXT(geometryNode.textureIndexMetallicRoughness)], tri.uv).g;
	vec3 albedo = pow(texture(textures[geometryNode.textureIndexBaseColor], tri.uv).rgb, vec3(2.2));//rayPayload.color;
	rayPayload.albedo = albedo;
	rayPayload.metallic = metallic;

	vec3 Lo = vec3(0.0f);

	int numOfLights = NUM_OF_LIGHTS;

	vec4 ShadowCoord;
	float shadow;
	for(int i = 0; i < numOfLights; i++) {

		ShadowCoord = ( biasMat * ubo.lightSpace ) * vec4(pos, 1.0);
		//shadow = textureProj(ShadowCoord / ShadowCoord.w, vec2(0.0));	// No PCF version.
		shadow = filterPCF(ShadowCoord / ShadowCoord.w);
		shadowed = (shadow != 1.0f);

		// Light attenuation unused
		if (!shadowed) {
			vec3 L = normalize(ubo.lightPos[i].xyz - pos);

			vec3 H = normalize(V + L);

			vec3 radiance = lightInfo[i].color;

			float NDF = DistributionGGX(N, H, roughness);
			float G   = GeometrySmith(N, V, L, roughness);      
			vec3 F    = FresnelSchlick(max(dot(H, V), 0.0), F0);
           
			vec3 numerator    = NDF * G * F; 
			float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; 
			vec3 specular = numerator / denominator;
        
			vec3 kS = F;

			vec3 kD = vec3(1.0) - kS;

			kD *= 1.0 - metallic;	  

			float NdotL = max(dot(N, L), 0.0);        

			Lo += (kD * albedo / PI + specular) * radiance * NdotL;
		}
    }
	
	vec3 ambient = vec3(0.05) * albedo;

	vec3 pbrColor = ambient + Lo;// + emissive;

	// for shadow map debugging
	//float depth = texture(shadowMap, inUV).r;
	//pbrColor = vec3(1.0-LinearizeDepth(depth));
	
	rayPayload.color = pbrColor;
}
