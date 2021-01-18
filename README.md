# Pan2019-AuthorshipAttribution-eecj

*Description goes here.*

## Always keep in mind!

1. All commands below must be run from the **root directory** of the project.
2. The first time you work with the repository, run the ```make``` command (see *setup* section).
3. Everytime you work in this project on the commandline, run the ```source env/bin/activate``` command (see virtual environment section).
4. Everytime a **new dependency** is added to requirements.txt, run ```make dist-clean``` and then run ```make``` (see *setup* and *clean and re-install* sections).

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

*An explanation of the data could go here.*

## Running

*Explanation of how to run the code in this repository.*
