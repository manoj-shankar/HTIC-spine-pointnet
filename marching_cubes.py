import vtk
index = [200,210,220,230,240]
name = ['L1','L2','L3','L4','L5']
def MarchingCubes(path,temp_file):
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(path)
        boneExtractor = vtk.vtkMarchingCubes()
        boneExtractor.SetInputConnection(reader.GetOutputPort())
        boneExtractor.SetValue(0,0.1)
        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputConnection(boneExtractor.GetOutputPort())
        smoothFilter.SetNumberOfIterations(15)
        smoothFilter.SetRelaxationFactor(1)
        #smoothFilter.FeatureEdgeSmoothingOff()
        smoothFilter.FeatureEdgeSmoothingOn()
        smoothFilter.BoundarySmoothingOn()
        #smoothFilter.BoundarySmoothingOff()
        smoothFilter.Update()
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(smoothFilter.GetOutputPort())
        writer.SetFileTypeToBinary()
        mask_file = str(temp_file) + '/fullspine.stl'
        writer.SetFileName(mask_file)
        writer.Write()
        print('Full spine created')

def DiscreteMarchingCubes(path,temp_file):
    nifti_reader = vtk.vtkNIFTIImageReader()
    nifti_reader.SetFileName(path)
    nifti_reader.Update()
    image = nifti_reader.GetOutput()
    for i in range (len(index)):
            discreteCubes = vtk.vtkDiscreteMarchingCubes()
            discreteCubes.SetInputData(image)
            discreteCubes.GenerateValues(200,10,index[i]);
            discreteCubes.Update()                   
            smoothFilter = vtk.vtkSmoothPolyDataFilter()
            smoothFilter.SetInputConnection(discreteCubes.GetOutputPort())
            smoothFilter.SetNumberOfIterations(50)
            smoothFilter.SetRelaxationFactor(1)
#     smoothFilter.FeatureEdgeSmoothingOff()
            smoothFilter.FeatureEdgeSmoothingOn()
            smoothFilter.BoundarySmoothingOn()
            smoothFilter.Update()
            stl_name =str(temp_file)+'/'+name[i] + '.stl'
            writer = vtk.vtkSTLWriter()
            writer.SetInputConnection(smoothFilter.GetOutputPort())
            writer.SetFileTypeToBinary()
            writer.SetFileName(stl_name)
            writer.Write()
    print('vertebrae created')
