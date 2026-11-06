"""Export the selected original study schematic without changing its content.

The retained schematic is exported without interpolation or content edits.
"""
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    import sys
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        raise SystemExit(0)
    folder=ROOT/'figures'/'Fig02_design';folder.mkdir(parents=True,exist_ok=True)
    im=Image.open(ROOT/'assets'/'study_design.png').convert('RGB')
    im.save(folder/'Fig02_design.png',dpi=(300,300))
    im.save(folder/'Fig02_design.pdf',resolution=250)
    sub=folder/'subfigures';sub.mkdir(exist_ok=True)
    (sub/'README.md').write_text('# Subfigures\n\nSingle integrated schematic; no independent quantitative subpanels.\n',encoding='utf8')
    print('Fig02_design exported at native resolution',im.size)
