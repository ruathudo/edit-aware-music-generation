To run the training:

`export TF_ENABLE_ONEDNN_OPTS=0`  
`python main.py train --model baseline`  
`python main.py train --model edit_aware --alpha 0.5`  

To run the evaluation:

`python main.py evaluate --model baseline --dataset test`  
`python main.py evaluate --model edit_aware_alpha_0.5 --dataset test`  
