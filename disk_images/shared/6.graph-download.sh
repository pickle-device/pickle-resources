#!/bin/bash

cd $HOME
mkdir graphs
cd $HOME/graphs

# amazon graph
wget https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz
gunzip com-amazon.ungraph.txt.gz
tail -n 925872 com-amazon.ungraph.txt > amazon.el
rm com-amazon.ungraph.txt

# youtube graph
wget https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz
gunzip com-youtube.ungraph.txt.gz
tail -n 2987624 com-youtube.ungraph.txt > youtube.el
rm com-youtube.ungraph.txt

# web_google graph
wget https://snap.stanford.edu/data/web-Google.txt.gz
gunzip web-Google.txt.gz
tail -n 5105039 web-Google.txt > web_google.el
rm web-Google.txt

# web_berkstan graph
wget https://snap.stanford.edu/data/web-BerkStan.txt.gz
gunzip web-BerkStan.txt.gz
tail -n 7600595 web-BerkStan.txt > web_berkstan.el
rm web-BerkStan.txt

# roadNetCA graph
wget https://snap.stanford.edu/data/roadNet-CA.txt.gz
gunzip roadNet-CA.txt.gz
tail -n 5533214 roadNet-CA.txt > roadNetCA.el
rm roadNet-CA.txt

# wiki_talk graph
wget https://snap.stanford.edu/data/wiki-Talk.txt.gz
gunzip wiki-Talk.txt.gz
tail -n 5021410 wiki-Talk.txt > wiki_talk.el
rm wiki-Talk.txt

# higgs graph
wget https://snap.stanford.edu/data/higgs-social_network.edgelist.gz
gunzip higgs-social_network.edgelist.gz
mv higgs-social_network.edgelist higgs.el

# wiki_topcats graph
wget https://snap.stanford.edu/data/wiki-topcats.txt.gz
gunzip wiki-topcats.txt.gz
mv wiki-topcats.txt wiki_topcats.el

# pokec graph
wget https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz
gunzip soc-pokec-relationships.txt.gz
mv soc-pokec-relationships.txt pokec.el

# livejournal graph
wget https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
gunzip soc-LiveJournal1.txt.gz
head -n 68993773 soc-LiveJournal1.txt > livejournal.el
rm soc-LiveJournal1.txt
