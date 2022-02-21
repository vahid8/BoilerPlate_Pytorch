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


