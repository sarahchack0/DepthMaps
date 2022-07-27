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

head = pv.read("meshes/Manny_closed_cleaned_decimated.ply")

p = pv.Plotter()
p.remove_bounding_box()
p.subplot(0, 0)
cylinder = pv.Cylinder(radius = .01)
sphere = pv.Sphere(radius = 5, end_theta = 180)

p.add_mesh(cylinder, color = "tan")
#p.add_mesh(sphere, color="tan")

p.add_mesh(head, show_scalar_bar = False, color='gray')

points = sphere.points
point_cloud = pv.PolyData(points)

#print(points)

def compute_vectors(mesh):
    origin = mesh.center
    vectors = mesh.points - origin
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    return vectors


vectors = compute_vectors(point_cloud)
print(vectors[0:5, :])
#p.camera.position = vectors[0]
point_cloud['vectors'] = vectors

arrows = point_cloud.glyph(
    orient='vectors',
    scale=False,
    factor=0.15,
)

# Display the arrows
p.add_mesh(point_cloud, color='maroon', point_size=10.0, render_points_as_spheres=True)
p.add_mesh(arrows, color='lightblue')
# plotter.add_point_labels([point_cloud.center,], ['Center',],
#                          point_color='yellow', point_size=20)


parent_dir = pathlib.Path(__file__).parent.resolve()
current_dir = "TestDir"

path = os.path.join(parent_dir, current_dir)

if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)



for i in range(0, len(vectors), 10):
    p.camera.position = vectors[i]
    print([p.camera.direction])
    p.camera.focal_point = (0.0, 0.0, 0.0)

    title = str(i) + ".svg"

    current_path = os.path.join(path, title)

    print(current_path)
    p.save_graphic(current_path, title=title)


print(head.center)
p.show()

