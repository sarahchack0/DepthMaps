import pyvista as pv
from pyvista import examples
import pickle
import matplotlib.pyplot as plt

#dataset = examples.download_bunny_coarse()
mesh = pv.read("meshes/Manny_closed_cleaned_decimated.ply")
p = pv.Plotter()
p.store_image = True
p.add_mesh(mesh, label='Clipped')
print(p.camera_position)
p.show()

p.camera_position = [(-15.039953480345513, 382.84110573062367, 130.7079362741775),
 (3.3055686950683594, -197.8695831298828, -144.57818984985352),
 (-0.045723710312163546, -0.42908762969161696, 0.9021048433308179)]

# p.show()

plt.imshow(p.get_image_depth(fill_value=-999))
plt.text(0, 0, p.camera_position)
print(p.camera_position)
plt.show()

# [(3.3055686950683594, -197.8695831298828, 498.34008923163697),
#  (3.3055686950683594, -197.8695831298828, -144.57818984985352),
#  (0.0, 1.0, 0.0)]
