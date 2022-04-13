## Shortcuts pytorch
| Command | Description |
| --- | --- |
| `keypoints = outputs["instances"].pred_keypoints.to("cpu").detach().numpy()` | pytorch tensor to numpy |
| `t = torch.tensor([1, 2, 3], [4, 5, 6])` | pytorch create a tensor (defualt type will be float32)|
| `t = torch.tensor([1, 2, 3], [4, 5, 6], dtype=torch.float64)` | pytorch create a tensor (specify type)|
| `t.shape` | pytorch tensor sahpe |
| `t.ndim` | pytorch tensor n dimention |
| `t.dtype` | pytorch tensor data types inside (e.g torch.float32) |
| `t.dot(t2)` | pytorch dot product |

## Shortcuts numpy
```
assumeing 
array_a = np.array([1,2,3,4,5])
```
| Command | Description |
| --- | --- |
| `array_b = np.where(array_a >= 3, -1, 1)` | array_b = [ 1  1 -1 -1 -1] |
| `array_b = np.where(array_a >= 3, array_a*10, array_a)` | array_b = [ 1  2 30 40 50]|
| `a = np.empty((0,3), int)` | Create a new numpy for appending or stacking |
| `img = np.zeros([100,100,3],dtype=np.uint8) then img.fill(255) # or img[:] = 255` | create an empty black or white image
| `id_x = np.where((points[:,0] > x_min) & (points[:,0] < x_max))` | Get the id of desired part by filtering|
| `id_x =  points[:,0] > x_min) & (points[:,0] < x_max` | Get boolean array to filter data|
| `np.linspace(min_value, max_value, num=int((max-min)/dist),endpoint=True)` | Create data or points between two number|
| `np.min(points,axis =0),np.max(points,axis =0),np.mean(points,axis =0)` | Get Min, Max, Mean of points
| `points3D = np.vstack((f.x, f.y, f.z)).transpose()` | From las files to numpy in n*3 format
| `np.linalg.norm(ppts2d - np.array([x, y]), axis=1)` | Calc distance of a vec elements to a point
| `diff_to_min = ppts2d - np.array([x, y])` | Calc difference of a vec elements to a point (signed)
| `filter_axe = np.all(diff_to_min > 0, axis=1)` | find 2d points bigger than desired values in both x, y


### Read detection text files in yolo and convert it to pascal
`
img_w = image.shape[0]                      
img_H = image.shape[1]                      
files = [line.strip().split() for line in open(os.path.join(self.label_dir, label_file))]                     
bboxes = [[int((float(item[1]) - (float(item[3]) / 2)) * img_w),  # x_min               
           int((float(item[2]) - (float(item[4]) / 2)) * img_H),  # y_min               
           int((float(item[1]) + (float(item[3]) / 2)) * img_w),  # x_max               
           int((float(item[2]) + (float(item[4]) / 2)) * img_H)]  # y_max               
          for item in files]                
`

                      
