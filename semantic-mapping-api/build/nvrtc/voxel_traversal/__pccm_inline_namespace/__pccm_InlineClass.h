#pragma once
#include <cumm/common/TensorViewNVRTCKernel.h>
#include <cumm/common/TensorViewArrayLinalg.h>
#include <cumm/common/TensorViewNVRTC.h>
#include <cumm/common/GemmBasic.h>
namespace __pccm_inline_namespace {
using TensorViewNVRTCKernel = cumm::common::TensorViewNVRTCKernel;
using TensorViewArrayLinalg = cumm::common::TensorViewArrayLinalg;
using TensorViewNVRTC = cumm::common::TensorViewNVRTC;
using GemmBasic = cumm::common::GemmBasic;
__global__ void __pccm_inline_function(float* ray_start, int64_t ray_start_stride_0_, float* ray_end, int64_t ray_end_stride_0_, int64_t point_cloud_range_0_, int64_t point_cloud_range_1_, int64_t point_cloud_range_2_, float voxel_size_0_, float voxel_size_1_, float voxel_size_2_, int64_t spatial_shape_0_, int64_t spatial_shape_1_, int64_t spatial_shape_2_, int64_t voxel_stride_0_, int64_t voxel_stride_1_, int64_t voxel_stride_2_, int32_t* voxel_occ_count, int64_t* points_labels_inrange, int32_t* voxel_free_count, int64_t _cumm_pccm_inline_size)   {
  
  for (auto i : tv::KernelLoopX<int>(_cumm_pccm_inline_size)){
    auto ray_start_p = ray_start + i*ray_start_stride_0_;
    auto ray_end_p = ray_end + i*ray_end_stride_0_;
    // int idx = $idx_tensor[0];
    // int count = 0;
    // if(i==idx){
    //     $debug_tensor[count*3+0] = current_voxel[0];
    //     $debug_tensor[count*3+1] = current_voxel[1];
    //     $debug_tensor[count*3+2] = current_voxel[2];
    //     count ++;
    // }
    // Bring the ray_start_p and ray_end_p in voxel coordinates
    float new_ray_start[3];
    float new_ray_end[3];
    float voxel_size_[3];
    new_ray_start[0] = ray_start_p[0] - point_cloud_range_0_;
    new_ray_start[1] = ray_start_p[1] - point_cloud_range_1_;
    new_ray_start[2] = ray_start_p[2] - point_cloud_range_2_;
    new_ray_end[0] = ray_end_p[0] - point_cloud_range_0_;
    new_ray_end[1] = ray_end_p[1] - point_cloud_range_1_;
    new_ray_end[2] = ray_end_p[2] - point_cloud_range_2_;
    voxel_size_[0] = voxel_size_0_;
    voxel_size_[1] = voxel_size_1_;
    voxel_size_[2] = voxel_size_2_;
    // Declare some variables that we will need
    float ray[3]; // keeep the ray
    int step[3];
    float tDelta[3];
    int current_voxel[3];
    int last_voxel[3];
    int target_voxel[3];
    float _EPS = 1e-9;
    for(int k=0; k<3; k++) {
        // Compute the ray
        ray[k] = new_ray_end[k] - new_ray_start[k];
        // Get the step along each axis based on whether we want to move
        // left or right
        step[k] = (ray[k] >=0) ? 1:-1;
        // Compute how much we need to move in t for the ray to move bin_size
        // in the world coordinates
        tDelta[k] = (ray[k] !=0) ? (step[k] * voxel_size_[k]) / ray[k]: 1000000000.0;
        // Move the start and end points just a bit so that they are never
        // on the boundary
        new_ray_start[k] = new_ray_start[k] + step[k]*voxel_size_[k]*_EPS;
        new_ray_end[k] = new_ray_end[k] - step[k]*voxel_size_[k]*_EPS;
        // Compute the first and the last voxels for the voxel traversal
        current_voxel[k] = (int) floor(new_ray_start[k] / voxel_size_[k]);
        last_voxel[k] = (int) floor(new_ray_end[k] / voxel_size_[k]);
        target_voxel[k] = (int) floor(new_ray_start[k] / voxel_size_[k]); // ray start as point, ray end as origin
    }
    // Make sure that the starting voxel is inside the voxel grid
    // if (
    //     ((current_voxel[0] >= 0 && current_voxel[0] < $grid_x) &&
    //     (current_voxel[1] >= 0 && current_voxel[1] < $grid_y) &&
    //     (current_voxel[2] >= 0 && current_voxel[2] < $grid_z)) == 0
    // ) {
    //     return;
    // }
    // Compute the values of t (u + t*v) where the ray crosses the next
    // boundaries
    float tMax[3];
    float current_coordinate;
    for (int k=0; k<3; k++) {
        if (ray[k] !=0 ) {
            // tMax contains the next voxels boundary in every axis
            current_coordinate = current_voxel[k]*voxel_size_[k];
            if (step[k] < 0 && current_coordinate < new_ray_start[k]) {
                tMax[k] = current_coordinate;
            }
            else {
                tMax[k] = current_coordinate + step[k]*voxel_size_[k];
            }
            // Now it contains the boundaries in t units
            tMax[k] = (tMax[k] - new_ray_start[k]) / ray[k];
        }
        else {
            tMax[k] = 1000000000.0;
        }
    }
    // record point, +1
    if (
        ((target_voxel[0] >= 0 && target_voxel[0] < spatial_shape_0_) &&
        (target_voxel[1] >= 0 && target_voxel[1] < spatial_shape_1_) &&
        (target_voxel[2] >= 0 && target_voxel[2] < spatial_shape_2_))
    ) {
        auto targetIdx = target_voxel[0] * voxel_stride_0_ + target_voxel[1] * voxel_stride_1_ + target_voxel[2] * voxel_stride_2_;
        auto old = atomicAdd(voxel_occ_count + targetIdx, 1);
    }
    // Start the traversal
    // while (voxel_equal(current_voxel, last_voxel) == 0 && ii < $max_voxels) {
    // while((current_voxel[0] == last_voxel[0] && current_voxel[1] == last_voxel[1] && current_voxel[2] == last_voxel[2])==0){
    while(step[0]*(current_voxel[0] - last_voxel[0]) < 1 && step[1]*(current_voxel[1] - last_voxel[1]) < 1 && step[2]*(current_voxel[2] - last_voxel[2]) < 1){ // due to traversal bias, ray may not exactly hit end voxel which cause traversal not stop
        // if tMaxX < tMaxY
        if (tMax[0] < tMax[1]) {
            if (tMax[0] < tMax[2]) {
                // We move on the X axis
                current_voxel[0] = current_voxel[0] + step[0];
                if (current_voxel[0] < 0 || current_voxel[0] >= spatial_shape_0_)
                    break;
                tMax[0] = tMax[0] + tDelta[0];
            }
            else {
                // We move on the Z axis
                current_voxel[2] = current_voxel[2] + step[2];
                if (current_voxel[2] < 0 || current_voxel[2] >= spatial_shape_2_)
                    break;
                tMax[2] = tMax[2] + tDelta[2];
            }
        }
        else {
            // if tMaxY < tMaxZ
            if (tMax[1] < tMax[2]) {
                // We move of the Y axis
                current_voxel[1] = current_voxel[1] + step[1];
                if (current_voxel[1] < 0 || current_voxel[1] >= spatial_shape_1_)
                    break;
                tMax[1] = tMax[1] + tDelta[1];
            }
            else {
                // We move on the Z axis
                current_voxel[2] = current_voxel[2] + step[2];
                if (current_voxel[2] < 0 || current_voxel[2] >= spatial_shape_2_)
                    break;
                tMax[2] = tMax[2] + tDelta[2];
            }
        }
        // set the traversed voxels
        auto currentIdx = current_voxel[0] * voxel_stride_0_ + current_voxel[1] * voxel_stride_1_ + current_voxel[2] * voxel_stride_2_;
        auto distance2start = abs(current_voxel[0] - target_voxel[0]) * voxel_size_[0] + abs(current_voxel[1] - target_voxel[1]) * voxel_size_[1] + abs(current_voxel[2] - target_voxel[2]) * voxel_size_[2];
        auto distance2end = abs(current_voxel[0] - last_voxel[0]) * voxel_size_[0] + abs(current_voxel[1] - last_voxel[1]) * voxel_size_[1] + abs(current_voxel[2] - last_voxel[2]) * voxel_size_[2];
        auto distance2start_height = abs(current_voxel[2] - target_voxel[2]) * voxel_size_[2];
        if(distance2start>1.0 && distance2end>1.0){
            if(points_labels_inrange[i]==72 || points_labels_inrange[i]<=48){
                if(distance2start_height < 1.0){
                    continue;
                }
            }
            auto old = atomicAdd(voxel_free_count + currentIdx, 1);
        }
    }
  }
}
struct __pccm_InlineClass {
};
} // namespace __pccm_inline_namespace