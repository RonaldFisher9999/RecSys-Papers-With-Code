#!/bin/bash

pip install -r requirements.txt

pip install torch_geometric==2.4.0 torch-sparse==0.6.17 torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
