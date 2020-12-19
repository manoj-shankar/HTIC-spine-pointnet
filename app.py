from marching_cubes import MarchingCubes,DiscreteMarchingCubes
import tempfile
from PointCloud import spine_ds
from Pointnet import spineClassifier,vertClassifier
mask_file = 'D:/HTIC documents/xVertSeg.v1/xVertSeg.v1/Data1/masks/3.nii'
temp_file = tempfile.TemporaryDirectory()
print(temp_file.name)
if mask_file.endswith('.nii'):
    MarchingCubes(mask_file,temp_file.name)
    DiscreteMarchingCubes(mask_file,temp_file.name)
print('==========Model==========')
spine_stl = temp_file.name + '/fullspine.stl'
spine_data = spine_ds(spine_stl)
# print(spine_data)
vert_name = ['L1','L2','L3','L4','L5']
vert_stl =[]
for i in range (0,5):
    path = temp_file.name + '/' +str (vert_name[i]) +'.stl'
    vert_stl.append(spine_ds(path))
spine_result = spineClassifier(spine_data)
print('The result of the classification is :'+ spine_result)

for i,data in enumerate(vert_stl):
    vert_result = vertClassifier(data)
    print('vertebra name :' +vert_name[i]+' is predicted to be :' +vert_result)


