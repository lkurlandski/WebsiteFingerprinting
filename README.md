# WebsiteFingerprinting
Use deep learning to perform website fingerprinting.

## Setup
```mkdir outputs data```

```mv full.zip ./data/full.zip```

```unzip data/full.zip ./data/full```

```conda env create -f environment.yml```

## Usage

```conda activate WF_Assigment```

Please the code and comments contained within the __main__ clause in main.py carefully. Adjust as desired.

```python main.py```

## Output

### Structure

```

-- "outputs"

   -- {OPTIMIZER}
   
      -- {LEARNING_RATE}
      
         -- {"RNN|LSTM|GRU"}
         
            -- {HIDDEN_UNITS}
            
               -- {NUM_LAYERS}
               
                  -- {BIDIRECTIONAL}
                  
                     -- "models"
                     
                        -- "{EPOCH}.pt"
                        
                     -- reports
                     
                        -- "{EPOCH}.pt"
                        
                     -- report.json
                     
         -- "CNN"
         
            -- {FILTER_SIZE}
            
               -- {ACTIVATION_FUNCTION}
               
                  -- history.json
                  
                  -- report.json

```

### Files

#### RNN/LSTM/GRU

file: outputs/{OPTIMIZER}/{RNN|LSTM|GRU}/{HIDDEN_UNITS}/{NUM_LAYERS}/{BIDIRECTIONAL}/models/0.pt

example: output_file_examples/1.pt

description: saved model at a particular epoch

-------------------------------------------------------------------------------------------------

file: outputs/{OPTIMIZER}/{RNN|LSTM|GRU}/{HIDDEN_UNITS}/{NUM_LAYERS}/{BIDIRECTIONAL}/reports/0.pt

example: output_file_examples/2.json

description: model evaluated on the validation set at a particular epoch

-------------------------------------------------------------------------------------------------

file: outputs/{OPTIMIZER}/{RNN|LSTM|GRU}/{HIDDEN_UNITS}/{NUM_LAYERS}/{BIDIRECTIONAL}/report.json

example: output_file_examples/3.json

description: best model (PATIENCE epochs before early stopping) evaluated on the test set

#### CNN

file: outputs/{OPTIMIZER}/CNN/{FILTER_SIZE}/{ACTIVATION_FUNCTION}/history.json

example: output_file_examples/4.json

description: model evaluated on the validation set at every epoch

-------------------------------------------------------------------------------------------------

file: outputs/{OPTIMIZER}/CNN/{FILTER_SIZE}/{ACTIVATION_FUNCTION}/report.json

example: output_file_examples/5.json

description: best model (PATIENCE epochs before early stopping) evaluated on the test set

