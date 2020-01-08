
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ifile', help='Input file', required=True)
parser.add_argument('--ofile', help='Output randomly sampled file', required=True)
parser.add_argument('--nlines', help='number of lines in the output file (approx.)', required=False)
parser.add_argument('--plines', help='percentage of lines from the input in the output sampled file', required=False)
args = parser.parse_args()

f=open(args.ifile)
ilines=[line.decode("utf8").rstrip() for line in f]
f.close()

if args.nlines is None and args.plines is None:
  plines=0.05 #DEFAULT VALUE
elif args.plines is None:
  plines=float(float(args.nlines)/float(len(ilines)))
else:
  plines=float(args.plines)

wf=open(args.ofile, "w")

for line in ilines:
  p = random.random()
  if p<plines:
    wf.write(line.encode("utf8")+"\n")

wf.close()
