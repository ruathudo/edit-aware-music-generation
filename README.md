To run the training:

`export TF_ENABLE_ONEDNN_OPTS=0`  
`python main.py train --model baseline`  
`python main.py train --model edit_aware --alpha 0.5`  

To run the evaluation:

`python main.py evaluate --model baseline --dataset test`  
`python main.py evaluate --model edit_aware_alpha_0.5 --dataset test`  


To generate edit scripts for all datasets
`python main.py sample --dataset train --max-edits 5`
`python main.py sample --dataset val --max-edits 5` 
`python main.py sample --dataset test --max-edits 5`


Test 1-4 with min-improvement 1e-4
Test 5-6 with min-improvement 1e-3