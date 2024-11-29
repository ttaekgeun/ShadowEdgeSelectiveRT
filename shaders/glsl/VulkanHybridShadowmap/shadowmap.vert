/*
 * Sogang University
 *
 * Graphics Lab, 2024
 *
 */

#version 460

layout (location = 0) in vec4 inPos;

layout (binding = 0) uniform UBO 
{
	mat4 depthMVP;
} ubo;

layout(location = 0) out vec4 outPos;

out gl_PerVertex 
{
    vec4 gl_Position;   
};

 
void main()
{
	outPos =  ubo.depthMVP * inPos;
	gl_Position = outPos;
}