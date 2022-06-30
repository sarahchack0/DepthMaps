# Sarah Chacko

import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import examples
import pickle

#dataset = examples.download_bunny_coarse()
mesh = pv.read("Manny_closed_cleaned.ply")
p = pv.Plotter()
p.add_mesh(mesh, label='Clipped')

#     pass
class PlotterControls:
    def __init__(self, p, dict_filename):
        self.p = p
        self.p.add_key_event("q", self.quit)
        self.p.add_key_event("p", self.toggle_play)
        self.paused = False
        self.stop = False

        self.depth_dict = {}
        self.filename = dict_filename
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

        lst = [self.p.camera_position, self.p.get_image_depth(fill_value=-999)]
        # save as dictionary entry, ex. {1: [cam position, depth map]}
        self.depth_dict[self.pic_num] = lst


pc = PlotterControls(p, "depth_dict.tsv")

p.show(auto_close=False, interactive_update=True)
while True:
    p.update()  # Non-blocking call to render updated environment, this also catches the key events
    if pc.stop:
        p.close()
        break
    elif pc.paused:
        p.show(auto_close=False, interactive_update=False)  # Blocking call, this also catches the key events

dict = pc.depth_dict

filename = "ten_maps_example.pkl"
# save dictionary as pickled file to access later
with open(filename, 'wb') as f:
    pickled_dict = pickle.dump(dict, f)





