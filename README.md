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
