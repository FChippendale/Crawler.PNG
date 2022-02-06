import argparse
import os
import errno
from src_py.crawler import Crawler
from src_py.utils import open_image, save_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', 
                        '--input_path', 
                        type=str, 
                        required=True, 
                        help='path to input image')

    parser.add_argument('-o', 
                        '--output_path', 
                        type=str, 
                        help='path to save output image separately')
                        
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(
            errno.ENOENT, 
            os.strerror(errno.ENOENT), 
            args.input_path
        )

    # if output path exists, check if it's valid

    world = open_image(args.input_path)
    crawler = Crawler(2, 2, world)

    while not crawler.isdead:
        crawler()

    if args.output_path is not None:
        save_image(world, args.output_path)
    else:
        save_image(world, args.input_path)
    