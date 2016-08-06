[dataset]
input = image
label = label
label_mask = label_mask

[image]
file = /Users/kisuklee/Workbench/seung-lab/znn-release/dataset/zfish/r1.kyle.img.h5
preprocess = {'type':'standardize','mode':'2D'}
	{'type':'mirror_border','fov':(19,209,209)}
fov = (19, 209, 209)
offset = (-9, -104, -104)
transform = {'type':'crop','offset':(1,1,1)}

[label]
file = /Users/kisuklee/Workbench/seung-lab/znn-release/dataset/zfish/r1.kyle.lbl.h5
transform = {'type':'affinitize'}
	{'type':'crop','offset':(1,1,1)}
fov = (11, 101, 101)
offset = (0, 0, 0)

[label_mask]
transform = {'is_mask': True, 'type': 'affinitize'}
	{'type':'crop','offset':(1,1,1)}
fov = (11, 101, 101)
offset = (0, 0, 0)
shape = (z,y,x)
filler = {'type':'one'}

