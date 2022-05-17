#include "Registration.h"

using namespace std;

Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
  // TO COMPLETE
  open3d::io::ReadPointCloud(cloud_source_filename, source_);
  open3d::io::ReadPointCloud(cloud_target_filename, target_);
}

Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
  // TO COMPLETE
  source_ = cloud_source;
  target_ = cloud_target;
}

void Registration::draw_registration_result()
{
  // visualize target and source with two different colors
  //  TO COMPLETE 1/5
  auto source_pointer = std::make_shared<open3d::geometry::PointCloud>(source_);
  auto target_pointer = std::make_shared<open3d::geometry::PointCloud>(target_);
  open3d::visualization::DrawGeometries({source_pointer, target_pointer});
}

void Registration::preprocess(open3d::geometry::PointCloud pcd, double voxel_size, std::shared_ptr<open3d::geometry::PointCloud> &pcd_down_ptr, std::shared_ptr<open3d::pipelines::registration::Feature> &pcd_fpfh)
{
  // downsample, estimate normals and compute FPFH features

  // TO COMPLETE
  // downsample
  double radius_normal = voxel_size * 2;
  pcd_down_ptr = pcd.VoxelDownSample(voxel_size);

  // Estimate point cloud normals
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
  open3d::pipelines::registration::RegistrationResult result;

  std::shared_ptr<open3d::geometry::PointCloud> source_down, target_down;
  std::shared_ptr<open3d::pipelines::registration::Feature> source_fpfh, target_fpfh;

  preprocess(source_, voxel_size, source_down, source_fpfh);
  preprocess(target_, voxel_size, target_down, target_fpfh);

  double distance_threshold = voxel_size * 1.5;

  auto vect;//{open3d::pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(0.9),open3d::pipelines::registration::CorrespondenceCheckerBasedOnDistance(distance_threshold)};

  cout << ":: RANSAC registration on downsampled point clouds." << endl;
  cout << "   Since the downsampling voxel size is: " << voxel_size << endl;
  cout << "   we use a liberal distance threshold " << distance_threshold << endl;
  result = open3d::pipelines::registration::RegistrationRANSACBasedOnFeatureMatching(
      *source_down, *target_down, *source_fpfh, *target_fpfh, true, distance_threshold,
      open3d::pipelines::registration::TransformationEstimationPointToPoint(false),
      3,vect,
      open3d::pipelines::registration::RANSACConvergenceCriteria(100000, 0.999)
      );

  set_transformation(result.transformation_);

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
