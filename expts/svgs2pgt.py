import pandas as pd
import pyarrow.parquet as pq
import os


def convert_svgs_to_parquet(inSvgDir, outFileName):
    count = 0
    tupleList = []
    for rn, dns, fns in  os.walk(inSvgDir):
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
    assert count == len(df.Svg)
    df.to_parquet(outFileName, engine='pyarrow')
    print(f"Collected {count} svgs into {outFileName}")

if __name__ == "__main__":
    trainDir = "train_nano_svgs"
    testDir = "test_nano_svgs"
    out_train = "train.nano.parquet"
    out_test = "test.nano.parquet"
    convert_svgs_to_parquet(trainDir, out_train)
    convert_svgs_to_parquet(testDir, out_test)