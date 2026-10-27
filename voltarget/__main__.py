import sys

from .report import main

main(sys.argv[1] if len(sys.argv) > 1 else "data/spy_us_d.csv")
