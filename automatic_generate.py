# Sarah Chacko

import pyvista as pv
from pyvista import examples
import vtk
from pyvista import demos
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import shutil
import pickle
import pyvistaqt as pvqt

head = pv.read("meshes/Manny_closed_cleaned_decimated.ply")

#p = pv.Plotter(off_screen = True)
p = pv.Plotter(notebook = False)
p.window_size = [240, 240]
p.store_image = True  # permit image caching after plotter is closed
p.show(auto_close=False)
p.remove_bounding_box()

# Display the head, sphere, and add point labels

p.add_mesh(head, show_scalar_bar=False, color='gray')

# p.add_point_labels([head.center], ['Center',],
#                          point_color='yellow', point_size=20)
#
# p.add_point_labels(trans.points[0], ['point',], point_color='yellow', point_size=20)

parent_dir = pathlib.Path(__file__).parent.resolve()
current_dir = "TestDir"

path = os.path.join(parent_dir, current_dir)

if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)


def angle_between_two_vectors(vector_1, vector_2):
    # calculate the dot product
    dot_product = np.dot(vector_1, vector_2)

    # calculate the magnitude of each vector
    vector_1_magnitude = np.linalg.norm(vector_1)
    vector_2_magnitude = np.linalg.norm(vector_2)

    # calculate the angle
    angle_rad = np.arccos(dot_product / (vector_1_magnitude * vector_2_magnitude))

    angle = np.degrees(angle_rad)

    return angle

pic_num = 0
depth_dict = {}

# get pictures at each point and save in pickle file
def take_images(mesh_sphere, radius):
    global pic_num, depth_dict

    points = mesh_sphere.points

    point_cloud = pv.PolyData(points)
    #p.add_mesh(point_cloud, color='maroon', point_size=10.0, render_points_as_spheres=True)

    # change the third value to modify how many images are taken
    for i in range(0, len(points), 10):
        p.camera.position = points[i]
        p.camera.focal_point = head.center

        direction_vector = np.subtract(points[i], head.center)

        # calculate angles for the title
        # calculate lateral angle
        lat_v1 = (direction_vector[0], direction_vector[1], 0)
        lat_v2 = (0, -1, 0)

        lat = angle_between_two_vectors(lat_v1, lat_v2)

        # calculate up-down angle
        ud_v1 = (0, direction_vector[1], direction_vector[2])
        ud_v2 = (0, -1, 0)

        ud = angle_between_two_vectors(ud_v1, ud_v2)

        if direction_vector[0] < 0:
            lat = -lat
        if direction_vector[2] < 0:
            ud = -ud

        try:
            title = str(int(lat)) + " " + str(int(ud)) + " rad_" + str(radius) + ".svg"
        except:
            title = str(pic_num) + ".svg"
            print("not possible")
        current_path = os.path.join(path, title)

        print(current_path)
        p.save_graphic(current_path, title=title)

        lst = [p.camera_position, p.get_image_depth(fill_value=-999)]

        depth_dict[pic_num] = lst

        pic_num += 1
        print(pic_num)


# make spheres

sphere_500 = pv.Sphere(radius=500, start_theta=180, end_theta=360)
trans_500 = sphere_500.translate(head.center, inplace=False)

sphere_600 = pv.Sphere(radius=600, start_theta=180, end_theta=360)
trans_600 = sphere_600.translate(head.center, inplace=False)

take_images(trans_500, 500)
#take_images(trans_600, 600)
#p.show()

filename = "training_data/testing_dict.pkl"
with open(filename, 'wb') as f:
    pickled_dict_small = pickle.dump(depth_dict, f)
