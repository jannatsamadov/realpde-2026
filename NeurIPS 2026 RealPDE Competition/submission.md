Submission
Submit a zip file containing a submission.py file at the root of the archive. Optional model weights can be included in the same archive.

Expected Archive Layout
submission.zip
  submission.py
  model.pth              # optional
  any_supporting_files/  # optional, e.g. vendored pure-Python packages
Submission Size Limit
Your submission archive must stay under 256 MB after extraction, model checkpoint included. Oversized submissions fail immediately without being evaluated.

Aim to meet this budget by design (model width, depth, and architecture) rather than by compressing an oversized checkpoint after the fact. Half-precision (fp16) packing is a lossy workaround, not the intended path, so we discourage relying on it; the safetensors format keeps checkpoints compact without loss, and a complex-safe fp16 packing tool is available in the starting kit only as a last resort.

Evaluation Environment
Submissions run offline in the official evaluation image pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime (Python 3.10, CUDA 12.1). Nothing is installed at evaluation time: the container has no network access, and a requirements.txt in your archive is ignored.

Available libraries include: torch / torchvision / torchaudio 2.2.2, numpy 1.26, pillow, PyYAML, requests, tqdm, sympy, networkx.

Not available (non-exhaustive): scipy, pandas, matplotlib, h5py, einops, scikit-learn, opencv.

If your code needs a pure-Python package that is not in the image, vendor it: copy the package directory into your archive next to submission.py. Compiled/native dependencies cannot be vendored reliably; stick to the image's libraries.

Minimal Interface
The simplest supported interface is a module-level predict function:

def predict(input_array, metadata=None):
    """Return predictions with shape (N, T_out, H, W, C)."""
input_array is a NumPy array with shape (N, T_in, H, W, C). The return value can be a NumPy array or a Torch tensor.

How predict Is Called
predict may be called more than once, each call in a fresh isolated subprocess. It reads your submission archive, writes temp files and caches as usual, and uses torch, CUDA and numpy as usual. Only the hidden evaluation data is unreadable. Module-level state, caches and files written during one call do not carry to the next.

metadata is an empty dictionary on scored calls. If your code reads keys from it, use .get() with a default or drop the dependency. The input horizon is input_array.shape[1], and the output horizon is on the Data page.

torch.utils.data.DataLoader must use num_workers=0.

Class-Based Interface
The ingestion program also supports a SubmissionModel class:

class SubmissionModel:
    def __init__(self, submission_dir=None, device="cpu"):
        ...

    def predict(self, input_array, metadata=None):
        ...
If your class implements load_checkpoint(path, device), the ingestion program will call it when model.pth exists.

Output Shape
Predictions must have shape (N, T_out, H, W, C), where C = 3. The scorer evaluates only measured real-world channels u and v; the pressure channel p may be returned as zeros.

Optional interval outputs are supported by returning a dictionary:

{
    "prediction": pred,
    "lower": lower_bound,
    "upper": upper_bound,
}
All arrays should use the same shape. If bounds are omitted, the scorer builds a default interval from the predictions.

Bounds are all-or-nothing across the whole run. predict may be called more than once, and every call must make the same choice: if some calls return lower / upper and others do not, all of them are discarded and every window is scored on the default band instead.

Practical Notes
Do not access external network resources during evaluation.
Keep inference deterministic and bounded in runtime.
Use relative paths inside your submission archive.
Heavy training should be done before submission; Codabench evaluation is for inference and scoring.