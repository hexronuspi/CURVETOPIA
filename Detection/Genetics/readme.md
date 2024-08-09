## Neural Network for 2D Shape Detection Using a GANN Approach

This approach combines a Genetic Algorithm (GA) with Neural Networks (NN) to detect 2D shapes in images:

- **Genetic Algorithm**: Heuristic optimization to explore and identify optimal neural network configurations.
- **Neural Network**: Precision training via gradient descent to fine-tune and improve shape detection accuracy.

The GANN approach integrates broad search capabilities with fine-tuned optimization for robust shape recognition.

Run:

1. Copy the code in code.py
2. In lines 290, 291 and 331 re-write the dataset train and test path 
3. Run the code 
 - (It takes around 1 hour to run on `Dataset/Synthetic`, ie: for around 14000 image on cpu)
 Note : The model was run on cpu most of the time.

4. Keep the image in validate path

Results: 
The Results are in results.txt

Limitation: 
The model was not able to detect all the shapes in the single image.