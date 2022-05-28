#!/bin/sh
python Attackcatsdogs.py --epochs 250 --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress policies/schedule-catsdogs.yaml --model ai85cdnet --dataset cats_vs_dogs --confusion --param-hist --embedding  --exp-load-weights-from ../ai8x-synthesis/trained/ai85-catsdogs-qat8.pth.tar --device MAX78000 "$@"
