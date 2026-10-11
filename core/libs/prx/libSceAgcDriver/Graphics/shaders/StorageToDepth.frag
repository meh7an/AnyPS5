#version 450

// A storage image of 32-bit texels into the depth plane of the pass's attachment, bit for bit
// (DepthSurface.cpp, DepthTransferPasses); the stencil plane is left as it was.
layout(set = 0, binding = 0, r32ui) uniform readonly uimage2D texels;

void main() {
    gl_FragDepth = uintBitsToFloat(imageLoad(texels, ivec2(gl_FragCoord.xy)).r);
}
