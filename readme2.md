# Video creation based on PRNet + Synthesizing Obama
## Terms
Common terminology for this doc and the rest of the project:

<dl>
<dt>Position map</dt>
<dd>
A <code>256x256x3</code> array whose <code>[v, u]</code> index stores the position of the vertex with uv-space coordinates <code>(u, v)</code> as an <code>(x, y, z)</code> triple
</dd>

<dt>Texture</dt>
<dd>
A <code>256x256x3</code> array whose <code>[v, u]</code> index stores the color of the vertex with uv-space coordinates <code>(u, v)</code> as a <code>(b, g, r)</code> triple
</dd>

<dt>Canonical face/position-map/texture</dt>
<dd>
Canonical face is any face that is used to align other faces, so that the center, size and orientation of the other faces are similar to the canonical face. Usually, there is one canonical face evaluated for every video chunk.<br> 
This face itself can have any size and orientation, but usually tends to be aligned so that the <code>x, y, z</code> values lie between ±2.5, with mean 0; and the face is looking directly in the front.<br>
</dd>

<dt>Frontalized position map</dt>
<dd>
A position map that has been aligned with a specific canonical position map
</dd>
</dl>

## Directory structure
Each video/video-chunk should have a corresponding directory, usually located in the `videos` folder, which should have the following files/subdirectories:
* **input**: _[Directory]_ Contains each video frame as an image (named `00001.jpg`, `00002.jpg` etc.)
* **posmap**: _[Directory] [Optional, not used after data extraction]_ Contains position map detected by PRNet for each video frame (named `00001.exr`, `00002.exr` etc.)
* **canonical**: _[Directory]_ Contains data for the canonical position-map and texture. Has the following files:
  * _`frontalized.exr`_: The canonical position map
  * _`texture.webp`_: The canonical texture
  * _`keypoints.npy`_: A 68`x`3 array of keypoint positions corresponding to `frontalized.exr`
  * _`normals.exr`_: A `256x256x3` array whose `[v, u]` index stores the area-vector of the quadrilateral formed by all vertices adjacent to the vertex with uv-space coordinates `(u, v)`
  * _`selected.txt`_: _[Optional, not used for anything]_ A list of all the frames that were used to create this canonical face.  

* **output**: _[Directory]_ Contains a directory for each face-containing video frame (named `00001`, `00002` etc.) containing information of the face. Each such directory has the following files within it:
  * _`frontalized.exr`_: The frontalized position map of the face
  * _`texture.webp`_: The texture of the face
  * _`keypoints.npy`_: A 68`x`3 array of frontalized keypoint positions corresponding to `frontalized.exr`
  * _`params.npy`_: A vector (usually 20-dimensional) containing PCA parameters for mouth shape and inner-mouth colors
  * _`lighting.npy`_: A 4-dimensional vector representing the lighting in the frame
  * _`texWithoutL.exr`_: The texture with the lighting separated
  * _`transform.txt`_: A 4x3 matrix which, when when multiplied with a homogenized frontalized position, gives the actual non-frontalized position. (i.e. The affine transform required to take the positions in `frontalized.exr` to the actual positions in the video)  

* **pca.npz**: Contains the matrices and coefficients required to go from actual frontalized positions and colors of inner mouth to the PCA parameters, and vice-versa.
* **albedo.npy**: Contains the matrix required to go from actual texture to the lighting parameters, and vice-versa.
* **video.mp4**: _[Optional, not used for anything]_ The actual video.

## Instructions for Extracting data
For instructions on how to run a particular python script within this repo, use `python <script> -h`
1. Create base directory and the `input` directory and `video.mp4` within it
1. Extract images using `ffmpeg -i <path/to/base>/video.mp4 -q:v 2 <path/to/base>/input/%05d.jpg`
1. Run `pipeline/demo.py` to get posmap folder
1. Run `pipeline/make_canonical.py` to get the canonical face
1. Run `pipeline/get_output.py` to get the `output` folder
1. Run TMFR to get better results for the keypoints
1. Run `pipeline/convert_tmfr_output.py` to get a `new_output` folder using TMFR results. Replace the output folder with this folder
1. Run `pipeline/get_pca_params.py` to get `pca.npz` in the base directory, and `params.npy` files in each directory in `output`
1. Run `pipeline/save_light_albedo.py` to get `albedo.npy` in base, and `lighting.npy` and `texWithoutL.exr` in each directory in `output`

Alternatively, just use the **`extract-data`** repository (which contains this one and TMFR as submodules).
