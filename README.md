## Testing Instructions

There are two main ways to test the functionality of the neural network:

---
### **1. Running Unit Tests (test.py)**

- Execute all of the unit tests by running the following command in the terminal:
  ```bash
  python -m unittest test.py
  ```

- These tests check the functionality of each class in the project:
  - **Neuron class:** verifies weight and bias initialization, and both forward and backward passes.  
  - **Layer class:** tests forward and backward propagation across multiple neurons.  
  - **NeuralNetwork class:** tests the forward pass, backward pass and full training loop.  

- If the tests run successfully, this will output:

  ```
  ...
  -------------------
  Ran X tests in Y.s
  OK
  ```

  indicating that all the tests have passed successfully.

---

### **2. Running the Main Script (main.py)**

- You can run the network manually using the following command in the terminal:

  ```bash
  python main.py
  ```

- The output should include:
  - A **network summary**, showing the number of layers and the number of neurons in each layer.  
  - A **loss report**, indicating the average loss per epoch. The loss should decrease and stabilise over time if a real dataset is used. However the data entered is random, with no correlation, so loss may not converge. 
  - **Final model predictions** printed after training completes.  

Successful results confirm that the forward pass, backpropagation, and weight updates are functioning as intended.