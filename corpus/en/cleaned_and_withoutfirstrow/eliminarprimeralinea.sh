for file in `ls *.txt`
do
  less $file | tail -n +2 > $file.tmp
  mv $file.tmp $file
done
