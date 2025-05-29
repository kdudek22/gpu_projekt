#include <math_constants.h>


struct Matrix3x3 {
    float m[3][3];
};


__device__ Matrix3x3 getRotationMatrixYXZ(float pitchRad, float yawRad, float rollRad) {
    float cx = cosf(pitchRad), sx = sinf(pitchRad); // X = pitch
    float cy = cosf(yawRad),  sy = sinf(yawRad);    // Y = yaw
    float cz = cosf(rollRad), sz = sinf(rollRad);   // Z = roll

    Matrix3x3 R;

    R.m[0][0] = cy * cz + sy * sx * sz;
    R.m[0][1] = cz * sy * sx - cy * sz;
    R.m[0][2] = cx * sy;

    R.m[1][0] = cx * sz;
    R.m[1][1] = cx * cz;
    R.m[1][2] = -sx;

    R.m[2][0] = cy * sx * sz - cz * sy;
    R.m[2][1] = sy * sz + cy * cz * sx;
    R.m[2][2] = cx * cy;

    return R;
}


__device__ int getVoxelSpaceCoordinatesToIndex(int x, int y, int z, int voxelSpaceDimX, int voxelSpaceDimY, int voxelSpaceDimZ){
    // Example index calculation:
    //            x                                       y                      z
    //int index = 100 * voxelSpaceDimY * voxelSpaceDimZ + 150 * voxelSpaceDimZ + 100;
    return x * voxelSpaceDimY * voxelSpaceDimZ + y * voxelSpaceDimZ + z;
}


extern "C" __global__ void process_image(int* img, int width, int height, int * cameraData, int *voxelSpace, int voxelSpaceDimX, int voxelSpaceDimY, int voxelSpaceDimZ, int voxelSpaceUnit) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int image_index = blockIdx.z;

    int cameraPosX = cameraData[image_index * 7];
    int cameraPosY = cameraData[1 + image_index * 7];
    int cameraPosZ = cameraData[2 + image_index * 7];

    float cameraRotRadX = (float)cameraData[3 + image_index * 7] * (CUDART_PI_F / 180.0f);
    float cameraRotRadY = (float)cameraData[4 + image_index * 7] * (CUDART_PI_F / 180.0f);
    float cameraRotRadZ = (float)cameraData[5 + image_index * 7] * (CUDART_PI_F / 180.0f);

    int cameraFov = cameraData[6 + image_index * 7];

    // ***** ACTUAL OPERATIONS START HERE *****
    int imageIndex = width * height * image_index + y * width + x;

    // If the value of the pixel is smaller than the threshold, do nothing :)
    if(img[imageIndex] < 10 || x >= width || y >= height){
        return;
    }

    // %%%%% LOGIC %%%%%
    float aspectRatio = (float) width / height;
    float fovRad = ((float) cameraFov) * (CUDART_PI_F / 180.0f);
    float tanHalfFov = tanf(fovRad / 2.0f);

    float u = (x + 0.5) / width;
    float v = (y + 0.5) / height;

    float cameraSpaceX = (2 * u - 1) * aspectRatio * tanHalfFov;
    float cameraSpaceY = (1 - 2 * v) * tanHalfFov;
    float cameraSpaceZ = 1.0f;

    Matrix3x3 rotMatrix = getRotationMatrixYXZ(cameraRotRadX, cameraRotRadY, cameraRotRadZ);

    float rayDirectionX = cameraSpaceX * rotMatrix.m[0][0] + cameraSpaceY * rotMatrix.m[0][1] + cameraSpaceZ * rotMatrix.m[0][2];
    float rayDirectionY = cameraSpaceX * rotMatrix.m[1][0] + cameraSpaceY * rotMatrix.m[1][1] + cameraSpaceZ * rotMatrix.m[1][2];
    float rayDirectionZ = cameraSpaceX * rotMatrix.m[2][0] + cameraSpaceY * rotMatrix.m[2][1] + cameraSpaceZ * rotMatrix.m[2][2];

    float length = rsqrtf(rayDirectionX * rayDirectionX + rayDirectionY * rayDirectionY + rayDirectionZ * rayDirectionZ);

    float normalizedDX = rayDirectionX * length;
    float normalizedDY = rayDirectionY * length;
    float normalizedDZ = rayDirectionZ * length;

    float ox = (float) cameraPosX;
    float oy = (float) cameraPosY;
    float oz = (float) cameraPosZ;

    float t = 0.0f;
    float tMax = 1000.0f;

    float step = 1.0f;

    while (t < tMax){
        float px = ox + normalizedDX * t;
        float py = oy + normalizedDY * t;
        float pz = oz + normalizedDZ * t;

        int ix = (int) (px / voxelSpaceUnit);
        int iy = (int) (py / voxelSpaceUnit);
        int iz = (int) (pz / voxelSpaceUnit);

        if( ix >= 0 && ix < voxelSpaceDimX && iy >= 0 && iy < voxelSpaceDimY && iz >= 0 && iz < voxelSpaceDimZ){
            atomicAdd(&voxelSpace[getVoxelSpaceCoordinatesToIndex(ix, iy, iz, voxelSpaceDimX, voxelSpaceDimY, voxelSpaceDimZ)], 1);
        }
        t += step;
    }
}