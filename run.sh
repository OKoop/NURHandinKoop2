#!/bin/bash
echo "Run handin 2 s1676059 ..."


cd Problem1

echo "Download the file for exercise 1"
if [ ! -e randomnumbers.txt ]; then
  wget https://home.strw.leidenuniv.nl/~nobels/coursedata/randomnumbers.txt
fi

#Runs the python script and thus creates all plots and saves outputs to a txt-file
echo "Run the python scripts for Exercise 1 ..."
python3 Handin21a.py > outputs1a.txt
python3 Handin21bthen.py > outputs1bt.txt
python3 Handin21bnow.py > outputs1bn.txt
python3 Handin21c.py > outputs1c.txt
python3 Handin21d.py > outputs1d.txt
python3 Handin21e.py > outputs1e.txt


cd ..

cd Problem2

echo "Run the python scripts for Exercise 2 ..."
python3 Handin22.py > outputs2.txt


cd ..
cd Problem3

echo "Run the python scripts for Exercise 3 ..."
python3 Handin23.py > outputs3.txt

cd ..
cd Problem4

echo "Create new maps to store the frames for the movie"
if [ ! -d "UF1" ]; then
  echo "Directory UF1 does not exist create it!"
  mkdir UF1
fi
if [ ! -d "UF2" ]; then
  echo "Directory UF2 does not exist create it!"
  mkdir UF2
fi


echo "Run the python scripts for Exercise 4 ..."
python3 Handin24ab.py > outputs4ab.txt
python3 Handin24c.py > outputs4c.txt
python3 Handin24d.py > outputs4d.txt

echo "Check if the c movie exist"
if [ -e moviec.mp4 ]; then
  echo "Remove mp4 file"
  rm moviec.mp4
fi

echo "Check if the d movie exist"
if [ -e movied.mp4 ]; then
  echo "Remove mp4 file"
  rm movied.mp4
fi


echo "Create the movies"
ffmpeg -framerate 30 -pattern_type glob -i "UF1/step*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 30 -threads 0 -f mp4 moviec.mp4

ffmpeg -framerate 30 -pattern_type glob -i "UF2/step*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 30 -threads 0 -f mp4 movied.mp4

cd ..
cd Problem5

echo "Run the python scripts for Exercise 5 ..."
python3 Handin25ab.py > outputs5ab.txt
python3 Handin25c.py > outputs5c.txt
python3 Handin25d.py > outputs5d.txt
python3 Handin25e.py > outputs5e.txt
python3 Handin25fg.py > outputs5fg.txt

cd ..
cd Problem6

echo "Download the file for exercise 6"
if [ ! -e GRBs.txt ]; then
  wget https://home.strw.leidenuniv.nl/~nobels/coursedata/GRBs.txt
fi

echo "Run the python scripts for Exercise 6 ..."
python3 Handin26.py > outputs6.txt

cd ..
cd Problem7

echo "Download the file for exercise 7"
if [ ! -e colliding.hdf5 ]; then
  wget https://home.strw.leidenuniv.nl/~nobels/coursedata/colliding.hdf5
fi

echo "Run the python scripts for Exercise 7 ..."
python3 Handin27.py > outputs7.txt

cd ..

#Generate the pdf by using LaTeX
echo "Generating the pdf ..."
pdflatex HI2.tex
bibtex HI2.aux
pdflatex HI2.tex
pdflatex HI2.tex
