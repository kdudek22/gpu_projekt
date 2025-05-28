#include <math_constants.h>

struct Mat3x3 {
    float m[9];
};


__device__ int getVoxelSpaceCoordinatesToIndex(int x, int y, int z, int voxelSpaceDimX, int voxelSpaceDimY, int voxelSpaceDimZ){
    // Example index calculation:
    //            x                                       y                      z
    //int index = 100 * voxelSpaceDimY * voxelSpaceDimZ + 150 * voxelSpaceDimZ + 100;
    return x * voxelSpaceDimY * voxelSpaceDimZ + y * voxelSpaceDimZ + z;
}

__device__ Mat3x3 eulerToRotationMatrix(float rx, float ry, float rz) {
    // Convert individual axis angles to sin/cos
    float sx = sinf(rx), cx = cosf(rx);
    float sy = sinf(ry), cy = cosf(ry);
    float sz = sinf(rz), cz = cosf(rz);

    Mat3x3 R;

    // Z-X-Y rotation (Unity default)
    R.m[0] = cy * cz + sy * sx * sz;
    R.m[1] = cz * sy * sx - cy * sz;
    R.m[2] = cx * sy;

    R.m[3] = cx * sz;
    R.m[4] = cx * cz;
    R.m[5] = -sx;

    R.m[6] = cy * sx * sz - cz * sy;
    R.m[7] = cy * cz * sx + sy * sz;
    R.m[8] = cy * cx;

    return R;
}

extern "C" __global__ void process_image(unsigned char* img, int width, int height, int * cameraData, int *voxelSpace, int voxelSpaceDimX, int voxelSpaceDimY, int voxelSpaceDimZ, int voxelSpaceUnit) {
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
    if(img[imageIndex] == 0 || x >= width || y >= height){
        return;
    }

//     Mat3x3 rotationMatrix = eulerToRotationMatrix(cameraRotX, cameraRotY, cameraRotZ);
//     voxelSpace[getVoxelSpaceCoordinatesToIndex(50, 100, 150, voxelSpaceDimX, voxelSpaceDimY, voxelSpaceDimZ)] = 150;

    // Step 1: Compute camera space direction from pixel
    float aspectRatio = (float)width / (float)height;
    float fovRad = ((float)cameraFov) * (CUDART_PI_F / 180.0f);

    float px = (2.0f * ((x + 0.5f) / (float)width) - 1.0f) * tanf(fovRad / 2.0f) * aspectRatio;
    float py = (1.0f - 2.0f * ((y + 0.5f) / (float)height)) * tanf(fovRad / 2.0f);
    float pz = 1.0f; // looking into -Z in camera space

    // Step 2: Rotate direction into world space
    Mat3x3 R = eulerToRotationMatrix(cameraRotRadX, cameraRotRadY, cameraRotRadZ);
    float dx = R.m[0]*px + R.m[1]*py + R.m[2]*pz;
    float dy = R.m[3]*px + R.m[4]*py + R.m[5]*pz;
    float dz = R.m[6]*px + R.m[7]*py + R.m[8]*pz;


//     printf("%f, %f, %f,%f, %f, %f,%f, %f, %f\n", R.m[0], R.m[1], R.m[2], R.m[3], R.m[4], R.m[5], R.m[6], R.m[7], R.m[8] );

    // Normalize the direction
    float invLen = rsqrtf(dx*dx + dy*dy + dz*dz);
    dx *= invLen; dy *= invLen; dz *= invLen;

//     printf("%f, %f, %f\n", dx, dy, dz);

    // Step 3: Ray origin
    float ox = (float)cameraPosX;
    float oy = (float)cameraPosY;
    float oz = (float)cameraPosZ;

    printf("%f, %f, %f\n", dx, dy, dz);

    // Step 4: DDA traversal
    float t = 0.0f;
    const float tMax = 1000.0f;
    const float step = 1.0f; // Step size in world units
    while (t < tMax) {
        float px = ox + dx * t;
        float py = oy + dy * t;
        float pz = oz + dz * t;

        int ix = (int)(px / voxelSpaceUnit);
        int iy = (int)(py / voxelSpaceUnit);
        int iz = (int)(pz / voxelSpaceUnit);

        if (ix >= 0 && ix < voxelSpaceDimX &&
            iy >= 0 && iy < voxelSpaceDimY &&
            iz >= 0 && iz < voxelSpaceDimZ) {

//             printf("%d, %d, %d\n", ix, iy, iz);
            int idx = getVoxelSpaceCoordinatesToIndex(ix, iy, iz, voxelSpaceDimX, voxelSpaceDimY, voxelSpaceDimZ);
            atomicAdd(&voxelSpace[idx], 1); // Safe for parallel updates
        }

        t += step;
    }

}