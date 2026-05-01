import pathlib

f = pathlib.Path(r'C:\Users\Dell\.cache\torch\hub\facebookresearch_dinov2_main\dinov2\layers\attention.py')
txt = f.read_text()
txt = txt.replace('float | None', 'Optional[float]')
if 'from typing import Optional,' not in txt:
    txt = txt.replace('from typing import', 'from typing import Optional,')
f.write_text(txt)
print('attention.py fixed!')

f2 = pathlib.Path(r'C:\Users\Dell\.cache\torch\hub\facebookresearch_dinov2_main\dinov2\layers\block.py')
txt2 = f2.read_text()
txt2 = txt2.replace('float | None', 'Optional[float]')
if 'from typing import Optional,' not in txt2:
    txt2 = txt2.replace('from typing import', 'from typing import Optional,')
f2.write_text(txt2)
print('block.py fixed!')

print('All fixes done! Now run: python train_segmentation.py')
