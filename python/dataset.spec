# sample config file
[image1]
files = path/of/image1.tif_or_h5
        path/of/image2.tif_or_h5
offset = 0,0,0
fov = 5,109,109
preprocess = {'type':'standardize','mode':'2D'}
             {'type':'mirror_border','fov':(5,109,109)}
transform =

[label1]
files = path/of/label1.tif_or_h5
offset = 0,0,0
transform = affinitize
masks = path/of/mask1.tif_or_h5

[dataset1]
input1 = image1
input2 = image2
output1 = label1
output2 = label2

[general]
border_mode = mirror
augment = {'type': 'warp', 'mode': 0.5}
          {'type': 'jitter'}
          {'type': 'flip'}