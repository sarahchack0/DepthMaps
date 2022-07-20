# Sarah Chacko

import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import examples
import pickle

#dataset = examples.download_bunny_coarse()
mesh = pv.read("meshes/Manny_closed_cleaned_decimated.ply")
p = pv.Plotter()
p.add_mesh(mesh, label='Clipped', show_scalar_bar = False)
print(p.window_size)
p.remove_bounding_box()

#     pass
class PlotterControls:
    def __init__(self, p):
        self.p = p
        self.p.add_key_event("q", self.quit)
        self.p.add_key_event("p", self.toggle_play)
        self.paused = False
        self.stop = False

        self.depth_dict_small = {}
        self.depth_dict_large = {}
        self.pic_num = 0

    # Whenever 'q' is pressed, end program
    def quit(self):
        print("Stopping")
        self.stop = True


    # Whenever 'p' is pressed, location is saved if it is already not in dictionary
    def toggle_play(self):
        self.pic_num += 1
        print(self.pic_num)
        print(self.p.camera_position)
        self.paused = not self.paused
        self.p.window_size = [1024, 768]
        lst = [self.p.camera_position, self.p.get_image_depth(fill_value=-999)]
        # save as dictionary entry, ex. {1: [cam position, depth map]}
        self.depth_dict_large[self.pic_num] = lst

        self.p.window_size = [240, 240]
        lst = [self.p.camera_position, self.p.get_image_depth(fill_value=-999)]
        # save as dictionary entry, ex. {1: [cam position, depth map]}
        self.depth_dict_small[self.pic_num] = lst

        self.p.window_size = [1024, 768]


pc = PlotterControls(p)

p.show(auto_close=False, interactive_update=True)
while True:
    p.update()  # Non-blocking call to render updated environment, this also catches the key events
    if pc.stop:
        p.close()
        break
    elif pc.paused:
        p.show(auto_close=False, interactive_update=False)  # Blocking call, this also catches the key events

dict_large = pc.depth_dict_large
dict_small = pc.depth_dict_small

filename_large = "training_data/dict1_large.pkl"
filename_small = "training_data/dict1_small.pkl"
# save dictionary as pickled file to access later
with open(filename_large, 'wb') as f:
    pickled_dict_large = pickle.dump(dict_large, f)

with open(filename_small, 'wb') as f:
    pickled_dict_small = pickle.dump(dict_small, f)






