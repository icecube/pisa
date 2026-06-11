# Script for generating the root-level README.md from the jupyter notebook pisa_examples/README.ipynb
git rm README_files/*
jupyter nbconvert --to markdown --output-dir . pisa_examples/README.ipynb
git add -f README_files/*.png
git add README.md
git add pisa_examples/README.ipynb
