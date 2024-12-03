/*
 * Sogang Univ, Graphics Lab, 2024
 * 
 * Vulkan Full Raytracing
 * 
 */

#version 460

#extension GL_EXT_ray_query : enable // ihm
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
	uint effectFlag;
};

layout(location = 0) rayPayloadInEXT RayPayload rayPayload;
layout(location = 2) rayPayloadEXT bool shadowed;
hitAttributeEXT vec2 attribs;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 2, set = 0) uniform uniformBuffer
{
	mat4 viewInverse;
	mat4 projInverse;
	vec4 lightPos[1];
} ubo;

layout(binding = 4, set = 0) buffer GeometryNodes { GeometryNode nodes[]; } geometryNodes;
layout(binding = 6, set = 0) uniform sampler2D textures[];

#include "geometryfunctions.glsl"

vec3 CalculateSpotLight(Triangle tri, int lightIdx, LightInfo lightInfo, vec4 lightPos)
{
    // new updating version
//    float lightAngleScale = 1.0f / max(0.001f, cos(float(lightInfo.spot.innerConeAngle)) - cos(float(lightInfo.spot.outerConeAngle)));
//    float lightAngleOffset = -cos(float(lightInfo.spot.outerConeAngle)) * lightAngleScale;
//
//    float spotFactor = dot(vec3(-0.3f, -1.0f, 0.0f), normalize(tri.pos - lightInfo.position.xyz));
//    if (spotFactor > 50.0f * DEG_TO_RAD) {
//        float angularAttenuation = clamp(spotFactor * lightAngleScale + lightAngleOffset, 0.0f, 1.0f); 
//        angularAttenuation *= angularAttenuation;
//        return angularAttenuation * vec3(1.0f);
//    }
//    else {
//        return vec3(0.0f, 0.0f, 0.0f);
//    }

    // These code below is the old version which doesn't read light informations from glTF file.
    vec3 lightToPixel = normalize(tri.pos - lightPos.xyz);
    float spotFactor = dot(lightToPixel, lightInfo.spotDirection);

    if (spotFactor > cos(lightInfo.spotCutoff * DEG_TO_RAD)) {
        vec3 spotResultColor = pow(spotFactor, 2.0f) * vec3(1.0f);
        return spotResultColor;
	}
	else {
		return vec3(0.0f, 0.0f, 0.0f);
	}
}

void main()
{
	GeometryNode geometryNode = geometryNodes.nodes[nonuniformEXT(gl_GeometryIndexEXT)];

	//	Triangle tri = unpackTriangle(gl_PrimitiveID, 112);
	Triangle tri = unpackTriangle_2(gl_PrimitiveID, 112, geometryNode.vertexBufferDeviceAddress,
														 geometryNode.indexBufferDeviceAddress);
	//vec3 N = tri.normal;

	#define DO_NORMAL_MAPPING
	#ifdef DO_NORMAL_MAPPING
	if (geometryNode.textureIndexNormal > -1) {
		tri.normal = CalculateNormal(textures[nonuniformEXT(geometryNode.textureIndexNormal)], tri.normal, tri.uv, tri.tangent);
	}
	#endif

	//vec3 albedo = tri.color.rgb;
	if (nonuniformEXT(geometryNode.textureIndexBaseColor) > -1) {
		tri.color.rgb = pow(texture(textures[nonuniformEXT(geometryNode.textureIndexBaseColor)], tri.uv).rgb, vec3(2.2));
	}

//	vec3 aoMetallicRoughness = texture(textures[nonuniformEXT(geometryNode.textureIndexMetallicRoughness)], tri.uv).rgb;
	vec3 ao_roughness_metallic = texture(textures[nonuniformEXT(geometryNode.textureIndexMetallicRoughness)], tri.uv).rgb;
//	float metallic = aoMetallicRoughness.b;
//	float roughness = aoMetallicRoughness.g;
//	float ao = aoMetallicRoughness.r;

	// vec3 pbr_color = vec3(0.0f);
	rayPayload.color = vec3(0.0f);

	// vec3 ambient = vec3(0.05f) * tri.color.rgb;

	// vec3 pbrColor = vec3(0.05f) * tri.color.rgb + emissive + Lo;

	// vec3 emissive = vec3(0.0f);
//	if (textureIndexEmissive > -1) // emission
//		rayPayload.color += texture(textures[nonuniformEXT(textureIndexEmissive)], tri.uv).rgb;

	vec3 V = normalize(vec3(0.0f, 0.0f, 0.0f) - tri.pos);
//	float Kr = geometryNode.reflectance;
//	float Kt = geometryNode.refractance;

//	if (Kr>0.0f || Kt>0.0f){
//		tri.color.rgb = vec3(0.0f);
//	}

	rayPayload.effectFlag = 0x0;
	if (geometryNode.reflectance > 0.0f) { // Kr
		rayPayload.effectFlag = 0x1;
		tri.color.rgb = vec3(0.0f);
	}
	else if (geometryNode.refractance > 0.0f) { // Kt
		rayPayload.effectFlag = 0x10;
		tri.color.rgb = vec3(0.0f);
	}

	rayPayload.color += vec3(0.05f) * tri.color.rgb; // ambient (reflection?)

	float ior = geometryNode.ior;
	vec3 F0 = vec3(pow((ior - 1) / (ior + 1), 2));
	F0 = mix(F0, tri.color.rgb, ao_roughness_metallic.b);

	//vec3 Lo = vec3(0.0f);

	int numOfLights = NUM_OF_LIGHTS;
	bool shadowRayOn = SHADOW_RAY_ON;
	for(int i = 0; i < numOfLights; i++) {

		// Shadow casting
//		float tmin = 0.1f;
#define SHADOW_RAY_TMIN 0.1f
#define SHADOW_RAY_ORIGIN_MOVEMENT_EPSILON 0.1f
//		float epsilon = 0.1f;
		vec3 rayDirection = normalize(ubo.lightPos[i].xyz - tri.pos);	// current hit spot to light

		vec3 origin = tri.pos + rayDirection * SHADOW_RAY_ORIGIN_MOVEMENT_EPSILON;

		float tmax = length(ubo.lightPos[i].xyz - origin);

//#define USE_RAY_QUERY
		shadowed = false;
		if (shadowRayOn) {
#ifndef USE_RAY_QUERY
			shadowed = true;  
			traceRayEXT(topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 1, 0, 1, origin, SHADOW_RAY_TMIN, rayDirection, tmax, 2);
#else
			rayQueryEXT rayQuery;
			rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 
				0xFF, origin, SHADOW_RAY_TMIN, rayDirection, tmax);

				// Traverse the acceleration structure and store information about the first intersection (if any)
				while (rayQueryProceedEXT(rayQuery)) {
					const uint geometry_id =  rayQueryGetIntersectionGeometryIndexEXT(rayQuery, false);
					const uint primitive_id = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false);
					const vec2 barycentrics = rayQueryGetIntersectionBarycentricsEXT(rayQuery, false);

					GeometryNode geometryNode = geometryNodes.nodes[nonuniformEXT(geometry_id)];

					const uint triIndex = primitive_id * 3;
					Indices    indices = Indices(geometryNode.indexBufferDeviceAddress);
					Vertices   vertices = Vertices(geometryNode.vertexBufferDeviceAddress);
					vec2 vertices_uv[3];
					vec2 uv;
					for (uint i = 0; i < 3; i++) {
						const uint offset = indices.i[triIndex + i] * 7;
						vec4 d0 = vertices.v[offset + 0]; // pos.xyz, n.x
						vec4 d1 = vertices.v[offset + 1]; // n.yz, uv.xy
						vec4 d2 = vertices.v[offset + 2];
						vec4 d3 = vertices.v[offset + 5];

						vertices_uv[i] = d1.zw;
					}
					uv = vertices_uv[0] * (1.0f - barycentrics.x - barycentrics.y) + vertices_uv[1] * barycentrics.x + vertices_uv[2] * barycentrics.y;
					vec4 color = texture(textures[nonuniformEXT(geometryNode.textureIndexBaseColor)], uv);

					// If the intersection has hit a triangle, the fragment is shadowed
					if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCandidateIntersectionAABBEXT) {  
						if (color.a != 0.0f) {
							shadowed = true;
							rayQueryConfirmIntersectionEXT(rayQuery);
							break;
						}
					}
					else if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCandidateIntersectionTriangleEXT) {
						if (color.a != 0.0f) {
							shadowed = true;
							rayQueryConfirmIntersectionEXT(rayQuery);
							break;
						}
					}
				}
#endif
		}
		if (!shadowed) {
			// Spot light unused
			/*vec3 spot = vec3(1.0f);
			if (lightInfo[i].spotCutoff != 180.0f) {
				spot = CalculateSpotLight(pos, i, lightInfo[i], ubo.lightPos[i]);
			}*/

			vec3 L = normalize(ubo.lightPos[i].xyz - tri.pos);

			vec3 H = normalize(V + L);

			// vec3 radiance = lightInfo[i].color;
			// Light attenuation unused
			/*float dist = length(ubo.lightPos[i].xyz - pos); 
			radiance = ApplyAttenuation(lightInfo[i].color, dist);*/

//			float NDF = DistributionGGX(tri.normal, H, ao_roughness_metallic.g);
//			float G   = GeometrySmith(tri.normal, V, L, ao_roughness_metallic.g);      
			vec3 F    = FresnelSchlick(max(dot(H, V), 0.0f), F0);
           
	//		vec3 numerator    = NDF * G * F; 
	//		float denominator = 4.0 * max(dot(tri.normal, V), 0.0f) * max(dot(tri.normal, L), 0.0f) + 0.0001f; 
			vec3 specular = ( DistributionGGX(tri.normal, H, ao_roughness_metallic.g)
									* GeometrySmith(tri.normal, V, L, ao_roughness_metallic.g) * F)
									/ (4.0 * max(dot(tri.normal, V), 0.0f) * max(dot(tri.normal, L), 0.0f) + 0.0001f);
        
			// vec3 kS = F;

			vec3 kD = vec3(1.0f) - F;	// kS = F
	
			kD *= 1.0f - ao_roughness_metallic.b;	  

			// float NdotL = max(dot(tri.normal, L), 0.0f);        

			rayPayload.color += (kD * tri.color.rgb / PI + specular) * lightInfo[i].color * max(dot(tri.normal, L), 0.0f); // direct lighting
		}
    }   
 
//	rayPayload.color = pbrColor;
	rayPayload.distance = gl_RayTmaxEXT;
	rayPayload.normal = tri.normal;
}
