# Pan2019-AuthorshipAttribution-eecj

We used the data from the PAN Shared Task 2019 to do authorship attribution with 20 different problems.  
Five problems belonged to one of four languages (English, French, Spanish, Italian) respectively.  
The aim was to imitate the approach of the best performing team and try to reproduce their results.  
In the end, we used two different approaches:

1. Simple SVM
2. Ensemble SVM Setup (training per feature type) with soft voting

The features used for the task were:

1. Character n-grams
2. Word n-grams
3. Distortion n-grams
4. punctuation n-grams

## Results

We trained task-wise and on all texts combined.

The different tasks yielded very different results, which also applies to the original submissions to the competition. In our setup, the overall accuracy of the best approach (Soft Voting) resulted in an accuracy of 57% and an F1 score of 45%.
We did not manage to reach the quality of the results of the team whose approach we wanted to reproduce.  
This can have different reasons. Differences in the setup of the SVM classifiers (we used Grid Search to find the best classifier configutation) can lead to that.
Another possibility is that the feature setup was not done in the exact same way, even though we tried remain as close to the original description as possible.

Overall, the results we obtained were 13% (Accuracy) worse than those of the original best performing team.  
A look at the performances of the other teams shows that our results are a bit below the average of all results.
This shows that, even though we did not reach the desired results, our classifier setup was reasonable for most tasks.

Our F1 score was considerably worse. It made us conclude that some of the minority classes (some authors not represented as much in the test set), obtained very bad predictions (especially in some of the tasks that got F1 scores below 20%). That would certainly be a point to improve.

## Always keep in mind!

1. All commands below must be run from the **root directory** of the project.
2. The first time you work with the repository, run the ```make``` command (see *setup* section).
3. Everytime you work in this project on the commandline, run the ```source env/bin/activate``` command (see virtual environment section).
4. Everytime a **new dependency** is added to requirements.txt, run ```make dist-clean``` and then run ```make``` (see *setup* and *clean and re-install* sections).

## Workflow to Implement New Features

If you want to work on a feature, follow these steps:

1. Go the the **Issues** section and assign yourself the issue you want to work on.
2. Go to your local repository, type ```git checkout master``` and then type: ```git pull```
3. Create a new feature-branch using: ```git checkout -b your-feature-branch``` (use your desired branch name)
4. Implement the feature on your local machine
5. Add and commit your changes!
6. Push the feature branch to github using: ```git push -u origin your-feature-branch```
7. Create a Pull Request (on Github) that demands to merge your feature branch into master
8. Assign the merge request to one of the other two developers

9. The other person will check your implementation and comment on it. If things should be improved, you can make more commits on your local repository and then push them using ```git push``` (because the branch is already on github).
10. Once all problems have been resolved, the other developer can merge the branch into master (there is a button for that in the Pull Request).

**Important**: If you encounter **merge conflicts** and don't know how to handle them, tell the others.

## Setup

For the setup of this repository simply type:

    make

This will

- set up a virtual environment for this repository,
- install all necessary project dependencies.

### What if it does not work?

If anything does not work, try installing with ```Python 3.8```.  
Also make sure that you have the library ```virtualenv``` installed on your system. If not, install it with: ```pip install virtualenv```

## Clean and Re-install

To reset the repository to its inital state, type:

    make dist-clean

This will remove the virtual environment and all dependencies.  
With the `make` command you can re-install them.

To remove temporary files like .pyc or .pyo files, type:

    make clean

## Virtual Environment

Activate the virtual environment by typing:

    source env/bin/activate

In the beginning of your terminal prompt, you should now see ```(env)```.

Make sure to always work in the virtual environment because otherwise you will not have the correct dependencies at your disposition.  
If you encounter any **problems working in your IDE**, make sure that it also uses the python interpreter from the virtual environment.

## Dependencies

- scikit-learn (0.24.0)

## Data

The data for this project comes from the Pan Shared Task 2019 (only the Authorship attribution Task).

Link to the Shared Task: https://pan.webis.de/clef19/pan19-web/authorship-attribution.html  
The data was downloaded from there.

## Running

To regenerate the results for the base SVM classifier, run:

    python3 scripts/train.py

To regenerate the results for the soft voting approach, run:

    python3 scripts/train_soft_voting.py

Note that the running of the scripts is going to take some time (not hours but some minutes).
