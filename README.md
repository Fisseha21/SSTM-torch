# [Speckle Dataset Generation](https://github.com/Computational-Ocularscience/KinemaNet)<br/>

## Package Description

## Installation
```
conda create -n sstm_env python=3.10
conda activate sstm_env
pip install torch torchvision
pip install sstm-flow
```
## Usage
### Two-frame optical flow
```python
from sstm import estimate_flow

flow = estimate_flow(
    "checkpoints/sstm_t++-sintel.pth",
    "frame_0013.png",
    "frame_0014.png"
)
```

### Three-frame optical flow
```python
from sstm import estimate_flow

f12, f23 = estimate_flow(
    "checkpoints/sstm_t++-sintel.pth",
    "frame_0013.png",
    "frame_0014.png",
    "frame_0015.png"
)
```

### Using already loaded images
  The API also supports images already loaded with OpenCV or NumPy.

```python
import cv2
from sstm import estimate_flow

img1 = cv2.imread("frame_0013.png")
img2 = cv2.imread("frame_0014.png")

flow = estimate_flow(
    "checkpoints/sstm_t++-sintel.pth",
    img1,
    img2
)
```

### Faster inference (recommended)
If running inference on multiple frame pairs, load the model once.

```python
import cv2
from sstm import load_model, infer_flow_pair

model = load_model("checkpoints/sstm_t++-sintel.pth")

img1 = cv2.imread("frame_0013.png")
img2 = cv2.imread("frame_0014.png")

flow = infer_flow_pair(model, img1, img2)
```

### Saving flow outputs using .flo and standard flow color-wheel visualization 
```python
from sstm import estimate_flow

estimate_flow(
    "checkpoints/sstm_t++-sintel.pth",
    "frame_0013.png",
    "frame_0014.png",
    output_dir="results",
    save_flo=True,
    save_vis_outputs=True
)
```
This will output:
```
results/
├── flow0001.flo
├── flow0001.png
```

## Citation
If you find this work useful please cite:
```
@article{ferede2023sstm,
  title={SSTM: Spatiotemporal recurrent transformers for multi-frame optical flow estimation},
  author={Ferede, Fisseha Admasu and Balasubramanian, Madhusudhanan},
  journal={Neurocomputing},
  volume={558},
  pages={126705},
  year={2023},
  publisher={Elsevier}
}
```
