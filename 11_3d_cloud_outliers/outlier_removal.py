import open3d as o3d
import argparse

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def radius_outlier_removal(point_cloud, min_number_points, radius):
    return point_cloud.remove_radius_outlier(nb_points=min_number_points, radius=radius)

ap = argparse.ArgumentParser()
ap.add_argument("--ipc", required=True, help="Path to where the point cloud is stored")
ap.add_argument("--points", required=True, type=int, help="Minimum number of points for outlier removal")
ap.add_argument("--radius", required=True, type=float, help="Radius for outlier removal")
ap.add_argument("--opc", required=True, help="Resulting point cloud")
args = vars(ap.parse_args())

ipc = args["ipc"]
points = args["points"]
radius = args["radius"]
opc = args["opc"]

pcd = o3d.io.read_point_cloud(ipc)
o3d.visualization.draw_geometries([pcd])

voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
o3d.visualization.draw_geometries([voxel_down_pcd])

cl, ind = radius_outlier_removal(voxel_down_pcd, points, radius)
display_inlier_outlier(voxel_down_pcd, ind)

o3d.io.write_point_cloud(opc, cl)

