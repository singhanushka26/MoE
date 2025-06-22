mixture of experts implemetation with real and fake dataset <br />
All images stored in folders: /train/real/, /train/fake/ <br />
Dataset is sampled to 500 images for each real and fake <br />
Real: 1 and Fake:0 <br />
Images are resized to (224, 224) <br />
Normalized to [0,1] 

Defining models used -
1. Gating funtion : A small neural net that outputs weights over experts for each input. These weights decide how much each expert contributes
2. Expert : A simple dense layer in which each expert processes the same input, independently
3. ForensicMoEBlock : Combines all expert outputs using the gating weights. Uses jnp.sum(gating * expert_outputs) to compute final output
4. VMoEBinaryClassifier : The full model: 2 MoE blocks → Dense → ReLU → Dense(2). Outputs logits for classification (real vs fake) 

During validation: <br />
Gets predictions + softmax, Computes classification report and AUC, Logs gating weights for each image, Averages expert weights across all images <br />
Runs for several epochs: <br />
Trains on batches, Evaluates after each epoch <br />

Results : <br />
Epoch 30/30 <br />
Avg Training Loss: 0.4336 <br />

                    precision    recall  f1-score   support 
           0           0.51      0.99      0.68     10000 
           1           0.89      0.05      0.09     10000 
           
    accuracy                               0.52     20000 
    macro avg          0.70      0.52      0.38     20000 
    weighted avg       0.70      0.52      0.38     20000 

AUC Score: 0.7213479249999999
