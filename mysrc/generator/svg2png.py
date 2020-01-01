import argparse
import io
import os
import glob

import numpy as np
from PIL import Image
import cairosvg
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="/workspace/CRAFT-pytorch/data/icons"
    )
    parser.add_argument(
        "--db", type=str, default="sqlite:////workspace/CRAFT-pytorch/crawl_noun.db"
    )
    parser.add_argument("--n_samples", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    opt = get_parameters()
    os.makedirs(opt.root, exist_ok=True)

    engine = create_engine(opt.db, echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    md = MetaData(bind=engine, reflect=True)
    for r in session.query(md.tables["icon"]).filter().limit(opt.n_samples).all():
        uid, svg, tags = r[0], r[2], r[4]
        try:
            png = cairosvg.svg2png(bytestring=svg)
        except:
            continue
        im = Image.open(io.BytesIO(png))
        im.save(os.path.join(opt.root, "{}_{}.png".format(uid, tags)))

    # for f in glob.glob(opt.root + "/*.svg"):
    #     im = Image.open(io.BytesIO(cairosvg.svg2png(url=f)))
    #     im = im.crop(im.getbbox())
    #     im.save(f.replace(".svg", ".png"))

