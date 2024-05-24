# policycompression_timememorycosts
 Code for the manuscript "Time and memory costs jointly determine a speed-accuracy trade-off and set-size effects". 
- **Bold** words are .m filenames.
- *Italicized* words are .mat filenames or directory folder names.

All code assumes that the current directory of Matlab is the main folder. Hence, do the following before running any code:
```
# The variable main_folder contains the path to the main folder on your computer
cd(main_folder)
addpath(genpath(main_folder))
```
The code runs on Matlab version R2023a. 

## Online experiments
- Experiment 1: https://gershmanlab.com/experiments/shuze/bandits/experiment/index_exp1.html
- Experiment 2: https://gershmanlab.com/experiments/shuze/bandits/experiment/index_exp2.html
- Experiment 3: https://gershmanlab.com/experiments/shuze/bandits/experiment/index_exp3.html
- Experiment 3 variant with 5 set-size conditions (only appears in Figure S5): https://gershmanlab.com/experiments/shuze/bandits/experiment/index_exp3_old2.html

## Main folder
- **plot_manuscript_figures.m** creates all figures for the manuscript, based on the datafiles and the saved LBA model fits.

## *data* subfolder
- *iti_bandit_data_exp1.mat* to *iti_bandit_data_exp3_old2.mat* are the raw datafiles for each experiment. We excluded all participants who, in any test block, had an average RT over 5 seconds. These datafiles are used by **plot_manuscript_figures.m** above.
- Other *.mat* files with the suffix *_train_blocks_* contain only training block performance. They include both included and excluded participants. 
- Other *.mat* files with the suffix *_test_blocks_* contain only test block performance.
- Among the above, *.mat* files with the suffix *_noexclusion* do not apply the exclusion criterion. Hence they include both included and excluded participants. 

## *figures* subfolder
- It contains all figures generated by **plot_manuscript_figures.m** above.

## *utils* subfolder
- It contains helper functions for the Blahut-Arimoto algorithm, mutual information and conditional entropy estimation, within-participant errorbar visualization, and color palettes.

## *lba_models* subfolder
- **lba_fits_exp1.m** and **lba_fits_exp3.m** perform LBA fits for Experiment 1 and 3. They parse datafiles into LBA-compatible format and fit parameters.
### *data* subsubfolder
- It contains parsed datafiles for Experiment 1 and 3 for subsequent LBA fits. 
### *fits* subsubfolder
- It contains LBA fitted parameters for each experiment. They are used by **plot_manuscript_figures.m** above.
### *utils* subsubfolder
- It contains LBA code adapted from https://github.com/smfleming/LBA. Some of the code was modified for our model fits.

