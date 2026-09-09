#version 450
layout(location = 0) in float camera_depth;
layout(location = 0) out float output_depth;
void main() { output_depth = camera_depth; }
