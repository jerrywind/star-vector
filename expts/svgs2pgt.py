import pandas as pd
import pyarrow.parquet as pq
import os

inDir = "svgs"
outName = "nano.parquet"
tupleList = []

count = 0

for rn, dns, fns in  os.walk(inDir):
    for fn in fns:
        if fn.endswith(".svg"):
            with  open(os.path.join(rn, fn), "rb") as f:
                svgData = f.read().replace(b"\r\n", b"").decode("utf-8")
            if "data:image/png;base64" in svgData:
                print(f"Skipping {fn}")
                continue
            tupleList.append((svgData, fn))
            count += 1

df = pd.DataFrame(tupleList, columns=['Svg', 'Filename'])
df.to_parquet(outName, engine='pyarrow')