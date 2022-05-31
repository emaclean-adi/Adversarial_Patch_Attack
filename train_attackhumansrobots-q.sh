#!/bin/sh
python Attackcatsdogs.py --model ai85cdnet --dataset cats_vs_dogs --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/robotshumans/humansrobots1_qat-q.pth.tar -8 --device MAX78000 "$@"
