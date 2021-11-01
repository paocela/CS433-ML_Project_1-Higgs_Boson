# CS433 Machine Learning - Project1
### **Authors**: Olle Ottander, Paolo Celada and Gustav Karlbom
#### *Department of Computer Science, EPFL Lausanne, Switzerland*
## Content
A critical part in the ATLAS experiment is being able to distinguish between a *tau tau decay of a Higgs boson* versus *background*, using data detected after a head-on collision between 2 protons.<br> The projected consisted of applying machine learning methods to a set of original and already classified decay signatures in order to predict unseen ones. After an initial cleaning and preprocessing step, multiple ML methods have been tested on training data and the relative test error measured locally using cross-validation. The best model's prediction were then submitted to an online platform which calculated both accuracy and F1 score.

## Project Structure
Files created and used throughtout the project implementation are the following, with the corresponding purpose:
- **/project1.ipynb**: Jupyter Notebook file center of our whole implementation. It contains loading of the training and test data, data cleaning and preprocessing, together with the use of all machine learning methods taken into consideration and K-fold cross validation. Lastly, predictions are computed
- **/run.py**: script able to produce our best prediction, stored in a .csv file. The predictions are the ones submitted to the online platform. It's possible to run it in the correct folder with 
```sh
  python run.py
  ```
- **/implementaions.py**: contains used machine learning methods implementation
- **/helpers.py**: contains all functions used for data cleaning and preprocessing, together with feature transformation
- **/data**: folder containing training data (test data must be added manually in this folder after downloading them from )
- **/proj1_helpers.py**: contains functions to load training data, handle them and store prediction to an output file
- **/report.pdf**: report of the project
