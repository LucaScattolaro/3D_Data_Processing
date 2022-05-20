#include "Registration.h"

using namespace std;

Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
  // TO COMPLETE

  //--Constructor given .ply files
  open3d::io::ReadPointCloud(cloud_source_filename, source_);
  open3d::io::ReadPointCloud(cloud_target_filename, target_);
}

Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
  // TO COMPLETE

  //--Constructor given PointCloud variables
  source_ = cloud_source;
  target_ = cloud_target;
}

void Registration::draw_registration_result()
{
  // visualize target and source with two different colors
  //  TO COMPLETE 1/5

  open3d::geometry::PointCloud source_temp = source_;
  open3d::geometry::PointCloud target_temp = target_;
  source_temp.Transform(transformation_);


  source_temp.PaintUniformColor({1, 0.706, 0});
  target_temp.PaintUniformColor({0, 0.651, 0.929});

  auto source_pointer = std::make_shared<open3d::geometry::PointCloud>(source_temp);
  auto target_pointer = std::make_shared<open3d::geometry::PointCloud>(target_temp);

  open3d::visualization::DrawGeometries({source_pointer, target_pointer});
}

void Registration::preprocess(open3d::geometry::PointCloud pcd, double voxel_size, std::shared_ptr<open3d::geometry::PointCloud> &pcd_down_ptr, std::shared_ptr<open3d::pipelines::registration::Feature> &pcd_fpfh)
{
  // downsample, estimate normals and compute FPFH features

  // TO COMPLETE
  // downsample
  pcd_down_ptr = pcd.VoxelDownSample(voxel_size);

  // Estimate point cloud normals
  double radius_normal = voxel_size * 2;
  pcd_down_ptr->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(radius_normal, 30));

  // Compute FPFH features
  double radius_feature = voxel_size * 5;
  pcd_fpfh = open3d::pipelines::registration::ComputeFPFHFeature(*pcd_down_ptr, open3d::geometry::KDTreeSearchParamHybrid(radius_feature, 100));

  return;
}

open3d::pipelines::registration::RegistrationResult Registration::execute_global_registration(double voxel_size)
{
  // remember to apply the transformation_ matrix to source_cloud
  // create two point cloud to contain the downsampled point cloud and two structure to contain the features
  // call the Registration::preprocess function on target and transformed source
  // execute global transformation and update the transformation matrix
  // TO COMPLETE

  // open3d::pipelines::registration::RegistrationResult result;

  // //--Applying the transformation_ matrix to source_cloud
  // source_.Transform(transformation_);

  // //--Create two point cloud to contain the downsampled point cloud and two structure to contain the features
  // std::shared_ptr<open3d::geometry::PointCloud> source_down, target_down;
  // std::shared_ptr<open3d::pipelines::registration::Feature> source_fpfh, target_fpfh;

  // //--preprocess target and transformed source
  // preprocess(source_, voxel_size, source_down, source_fpfh);
  // preprocess(target_, voxel_size, target_down, target_fpfh);

  // double distance_threshold = voxel_size * 1.5;
  // vector<reference_wrapper<const open3d::pipelines::registration::CorrespondenceChecker>> correspondenceCheckers;

  // auto correspondenceCheckerEdgeLength=open3d::pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(0.9);
  // auto correspondenceCheckerBasedOnDistance=open3d::pipelines::registration::CorrespondenceCheckerBasedOnDistance(distance_threshold);

  // correspondenceCheckers.push_back(correspondenceCheckerEdgeLength);
  // correspondenceCheckers.push_back(correspondenceCheckerBasedOnDistance);

  // cout << ":: RANSAC registration on downsampled point clouds." << endl;
  // cout << "   Since the downsampling voxel size is: " << voxel_size << endl;
  // cout << "   we use a liberal distance threshold " << distance_threshold << endl;
  // result = open3d::pipelines::registration::RegistrationRANSACBasedOnFeatureMatching(
  //     *source_down, *target_down, *source_fpfh, *target_fpfh, true, distance_threshold,
  //     open3d::pipelines::registration::TransformationEstimationPointToPoint(false),
  //     3,correspondenceCheckers,
  //     open3d::pipelines::registration::RANSACConvergenceCriteria(100000, 0.999)
  //     );

  // //--update the transformation matrix
  // set_transformation(result.transformation_);

  // source_ = *source_down;
  // target_ = *target_down;
  // return result;

  source_.Transform(transformation_);

  // Initialize point cloud and FPFH feature pointers
  std::shared_ptr<open3d::geometry::PointCloud> source_pcd_down_ptr, target_pcd_down_ptr;
  std::shared_ptr<open3d::pipelines::registration::Feature> source_pcd_fpfh, target_pcd_fpfh;

  // Pre-processing source and target point clouds
  preprocess(source_, voxel_size, source_pcd_down_ptr, source_pcd_fpfh);
  preprocess(target_, voxel_size, target_pcd_down_ptr, target_pcd_fpfh);

  open3d::pipelines::registration::RegistrationResult result;

  double distance_threshold = 0.5 * voxel_size;

  result = open3d::pipelines::registration::FastGlobalRegistrationBasedOnFeatureMatching(
      *source_pcd_down_ptr,
      *target_pcd_down_ptr,
      *source_pcd_fpfh,
      *target_pcd_fpfh,
      open3d::pipelines::registration::FastGlobalRegistrationOption(1.4, false, true, distance_threshold));

  // Update transformation
  set_transformation(result.transformation_);

  // Update source and target point clouds
  // source_ = *source_pcd_down_ptr;
  // target_ = *target_pcd_down_ptr;

  // Return result
  return result;
}

open3d::pipelines::registration::RegistrationResult Registration::execute_icp_registration(double threshold, double relative_fitness, double relative_rmse, int max_iteration)
{
  open3d::pipelines::registration::RegistrationResult result;

  result = open3d::pipelines::registration::RegistrationICP(
      source_,
      target_,
      threshold,
      transformation_,
      open3d::pipelines::registration::TransformationEstimationPointToPoint(),
      open3d::pipelines::registration::ICPConvergenceCriteria(relative_fitness, relative_rmse, max_iteration));
  return result;
}

void Registration::set_transformation(Eigen::Matrix4d init_transformation)
{
  transformation_ = init_transformation;
}

Eigen::Matrix4d Registration::get_transformation()
{
  return transformation_;
}

void Registration::write_tranformation_matrix(std::string filename)
{
  std::ofstream outfile(filename);
  if (outfile.is_open())
  {
    outfile << transformation_;
    outfile.close();
  }
}

void Registration::save_merged_cloud(std::string filename)
{
  // clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  source_clone.Transform(transformation_);
  open3d::geometry::PointCloud merged = target_clone + source_clone;
  open3d::io::WritePointCloud(filename, merged);
}
