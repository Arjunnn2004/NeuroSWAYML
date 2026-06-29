"""
NeuroSWAYML — Domain Dataset Loaders
Each loader ingests a specific public dataset and returns
(X, y) arrays whose feature dimension matches
DatasetLoader.FEATURE_NAMES (30 features).

Available loaders
-----------------
ElderlyLoader       PhysioNet LTMM — 71 elderly participants, trunk
                    accelerometer, fall-risk labels.
                    https://physionet.org/content/ltmm/1.0.0/

IntoxicationLoader  PhysioNet HBEDB — 163 subjects, stabilography under
                    eyes-open/closed / foam conditions *plus* an optional
                    generic CSV folder (Kaiserslautern IMU data or any
                    time,acc_x,acc_y,acc_z,label file).
                    HBEDB: https://physionet.org/content/hbedb/1.0.0/
                    Kaiserslautern: contact TU-KL IUUI lab or IEEE DataPort.

CongenitalLoader    GaitRec v1 — 2 084 subjects (healthy + hip / knee /
                    ankle / back / neurological pathologies), GRF CSV.
                    DOI: 10.6084/m9.figshare.13598962.v1
                    https://figshare.com/articles/dataset/GaitRec_.../13598962
"""

from .elderly_loader      import ElderlyLoader
from .intoxication_loader import IntoxicationLoader
from .congenital_loader   import CongenitalLoader
from .urfd_loader         import URFDLoader

__all__ = ["ElderlyLoader", "IntoxicationLoader", "CongenitalLoader", "URFDLoader"]
