# terrain_fallback.py
from pathlib import Path
import rasterio
import requests, boto3, tempfile
from botocore import UNSIGNED
from botocore.client import Config
from rasterio.session import AWSSession

COPERNICUS_BUCKET = "copernicus-dem-30m"
NASADEM_BUCKET    = "nasadem"

def _open_aws_geotiff(bucket:str, key:str):
    s3 = boto3.Session().client("s3", config=Config(signature_version=UNSIGNED))
    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    # tiny HEAD request to see if the object exists
    if s3.head_object(Bucket=bucket, Key=key):
        return rasterio.open(f"/vsicurl/{url}")
    raise FileNotFoundError(key)

def robust_load_dem(bbox, cache_dir:Path):
    n,s,w,e = bbox                      # (latN, latS, lonW, lonE)
    # 1 ▸ try Copernicus DEM on AWS
    try:
        key = f"{int(n):02d}/Copernicus_DSM_GLO-30_{int(n):02d}_{int(e):03d}_DEM.tif"
        return _open_aws_geotiff(COPERNICUS_BUCKET, key)
    except Exception:
        pass
    # 2 ▸ fall back to NASADEM
    try:
        key = f"NASADEM_HGT_{int(n):02d}{int(e):03d}.TIF"
        return _open_aws_geotiff(NASADEM_BUCKET, key)
    except Exception:
        pass
    # 3 ▸ fall back to your existing OpenTopography helper
    from terrain import _download_dem   # re-use your current routine
    return rasterio.open(_download_dem(n, s, w, e, "ASTER30m"))
