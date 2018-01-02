## Methods used in Recommender Systems

The project implements and compares the various methods used in the building of recommender systems on the basis of the errors using Root Mean Square Error, Precision on Top and Spearman Correlation.

This repo uses python3.5. The dependencies can be installed using pip3.

### Installation:

To run the application, we need to install the following modules numpy, scipy and matplotlib. 
Run the following command in your terminal.
```
$ sudo pip3 install -r requirements.txt
```

If you face any issues, you can also install them separately.

#### Usage:

For using the application, go to the root folder of the project and type the following in the terminal:
```
$ pip3 run.py
```

You can comment and uncomment the lines accordingly to run the various algorithms separately.

### Issues:
- Collaborative Filtering takes a bit of time to run of the large dataset.
- There are a few corner cases where the code does not perform well.
