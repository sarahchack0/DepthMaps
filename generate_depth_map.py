# Sarah Chacko

import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import examples
import pickle
import os
import pathlib
import shutil

parent_dir = pathlib.Path(__file__).parent.resolve()
current_dir = "ImagesDir"

path = os.path.join(parent_dir, current_dir)

if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)

#dataset = examples.download_bunny_coarse()
mesh = pv.read("meshes/Manny_closed_cleaned_decimated.ply")
p = pv.Plotter()
p.window_size = [240, 240]
p.remove_bounding_box()

p.add_mesh(mesh, show_scalar_bar = False, color='gray')

print(p.window_size)

#     pass
class PlotterControls:
    def __init__(self, p):
        self.p = p
        self.p.add_key_event("q", self.quit)
        #self.p.add_key_event("p", self.toggle_play)
        self.p.track_click_position(callback=self.toggle_play, side='left')
        self.paused = False
        self.stop = False

        self.depth_dict_small = {}
        self.pic_num = 0

    # Whenever 'q' is pressed, end program
    def quit(self):
        print("Stopping")
        self.stop = True


    # Whenever 'p' is pressed, location is saved if it is already not in dictionary
    def toggle_play(self, a):
        self.pic_num += 1
        print(self.pic_num)
        print(self.p.camera_position)

        # self.paused = not self.paused

        self.p.window_size = [240, 240]
        lst = [self.p.camera_position, self.p.get_image_depth(fill_value=-999)]
        # save as dictionary entry, ex. {1: [cam position, depth map]}
        self.depth_dict_small[self.pic_num] = lst

        title = str(self.pic_num) + ".svg"

        current_path = os.path.join(path, title)


        print(current_path)
        self.p.save_graphic(current_path, title = title)


pc = PlotterControls(p)

p.show(auto_close=False, interactive_update=True)
while True:
    p.update()  # Non-blocking call to render updated environment, this also catches the key events
    if pc.stop:
        p.close()
        break
    elif pc.paused:
        p.show(auto_close=False, interactive_update=False)  # Blocking call, this also catches the key events


dict_small = pc.depth_dict_small

filename_small = "training_data/testing_dict.pkl"


with open(filename_small, 'wb') as f:
    pickled_dict_small = pickle.dump(dict_small, f)






