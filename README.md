# [SSTM-PyPI](https://pypi.org/project/sstm-flow/)<br/>

## Package Description
[SSTM: Spatiotemporal recurrent transformers for multi-frame optical flow estimation](https://github.com/Computational-Ocularscience/KinemaNet), is a deep learning package for dense optical flow estimation from image sequences.  
It supports both **two-frame** and **three-frame** inference.

For a two-frame input, the package predicts a single flow field:

- `f12`: optical flow from frame 1 → frame 2

For a three-frame input, the package predicts two flow fields:

- `f12`: optical flow from frame 1 → frame 2
- `f23`: optical flow from frame 2 → frame 3

The package also supports:

- input image paths
- already loaded OpenCV / NumPy images
- optional saving of flow fields as `.flo`
- optional color-wheel visualization of predicted flow
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
    "checkpoints/ft-sintel.pth",
    "frame_0013.png",
    "frame_0014.png"
)
```

### Three-frame optical flow
```python
from sstm import estimate_flow

f12, f23 = estimate_flow(
    "checkpoints/ft-sintel.pth",
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
    "checkpoints/ft-sintel.pth",
    img1,
    img2
)
```

### Faster inference (recommended)
If running inference on multiple frame pairs, load the model once.

```python
import cv2
from sstm import load_model, infer_flow_pair

model = load_model("checkpoints/ft-sintel.pth")

img1 = cv2.imread("frame_0013.png")
img2 = cv2.imread("frame_0014.png")

flow = infer_flow_pair(model, img1, img2)
```

### Saving flow outputs using .flo and standard flow color-wheel visualization 
```python
from sstm import estimate_flow

estimate_flow(
    "checkpoints/ft-sintel.pth",
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
## Model weight description and task specific recommendation
Under `checkpoints/.` there are three different fine-tuned versions of SSTM
| Model | Description | Recommended Use |
|------|-------------|----------------|
| **ft-speckle-sintel** | Fine-tuned for motion patterns that exhibit **local spatial deformation**. | Biomechanics, [elastography](https://computational-ocularscience.github.io/kinemanet.github.io/), biological tissue motion, speckle tracking |
| **ft-sintel** | General-purpose optical flow model trained on **non-rigid motion with diverse dynamics**. | Natural scenes, animation, and general research applications |
| **ft-kitti** | Fine-tuned for **rigid scene motion** typical in driving environments. | Autonomous driving, vehicle motion, rigid object tracking |

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
