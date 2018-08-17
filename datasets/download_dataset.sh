DATASET=$1

if [[$DATASET != "portrait" && $DATASET != "cat2dog"]]; then
  echo "dataset not available"
  exit
fi

URL=http://vllab.ucmerced.edu/hylee/DRIT/datasets/$DATASET.zip
wget -N $URL -O ../datasets/$DATASET.zip
unzip ../datasets/$DATASET.zip -d ../datasets
rm ../datasets/$DATASET.zip
