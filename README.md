# Tracking Low-Level Cloud Systems with Topology

The repository includes the implementation for the paper "Tracking Low-Level Cloud Systems with Topology". 

## Tools for preparing the data from raw COD field and for visualization (Optional):
You may want to install ParaView == 5.13.3. Other versions of ParaView may have different APIs for toolkits.
Then, follow https://topology-tool-kit.github.io/installation.html to install the plugin of Topology Toolkit (TTK).
This toolkit helps compute the merge tree from the scientific scalar fields.
ParaView is a visualization tool for such data and comes with the Visualization Toolkit (VTK) for processing such data.
To use TTK- or VTK-related scripts, you should make sure that "pvpython" is installed (which comes in bundle with ParaView) and added to $PATH.

We do not recommend mixing pvpython with anaconda environments. The scripts are written with the assumption that two environments are isolated.

The pre-processed COD field and merge tree data is available at https://doi.org/10.5281/zenodo.16924680. For post-processing and visualization, TTK is not necessary.

## Installation
All implementations are done in Python and Jupyter Notebook. It is highly recommended that you use Anaconda for environment control.

```
conda create -n pfgw_cloud python==3.10 networkx scipy jupyter matplotlib pandas
conda activate pfgw_cloud
pip install netcdf4 pot
```

## Pipeline
##### Running Merge-Tree-Based Tracking
Run "MarineCloud/tracking-binary.ipynb" for the Marine Cloud dataset, or "LandCloud/tracking-binary-juelich.ipynb" for the Land Cloud dataset

##### Running Post-processing Script to Compute Cloud Trajectories
Run "MarineCloud/postprocess.py", or "LandCloud/postprocess.py".

##### Generate files used for visualization of trajectories
Run "MarineCloud/plotting.ipynb", or "LandCloud/plotting-juelich.py"

##### ParaView/VTK/TTK-related
We also provide pvpython scripts in "./pvscripts", including
* generate.py (with generate.config): generate merge trees using TTK modules
* mat2vtx.py (with mat2vtx.config): convert files storing 2D scalar matrices (typically .npy or .txt) to a 2D VTK file for the scalar field
  - generating scalar field files for TTK to generate merge trees
  - converting the output images of cloud systems with trajectory IDs to VTI files for visualization
* highlightAP.py: convert the output from "plotting.ipynb" to vtu files, representing the centroid of cloud systems
* tracking.py: connect centroids across timesteps to trajectories

## Demo videos
* Marine Cloud: https://youtu.be/4K6omuC6qM4
* Land Cloud (Morning): https://youtu.be/-mtFO5tGmpI
* Land Cloud (Midday): https://youtu.be/JZRAfMNfpfU