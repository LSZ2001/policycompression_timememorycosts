# policycompression_timememorycosts
 Code for the manuscript "Time and memory costs jointly determine a speed-accuracy trade-off and set-size effects". 
- **Bold** words are .m filenames.
- *Italicized* words are .mat filenames or directory folder names.

All code assumes that the current directory of Matlab is the main folder. The code runs on Matlab version R2023a. 

## Main folder
- **plot_manuscript_figures.m** creates all figures for the manuscript, based on the datafiles and the saved LBA model fits.

## *data* subfolder
- *iti_bandit_data_exp1.mat* to *iti_bandit_data_exp3.mat* are the raw datafiles for each experiment. We remove all participants who, in any test block, has an average RT over 5 seconds. These datafiles are used by **plot_manuscript_figures.m** above.
- Other *.mat* files with the suffix *_train_blocks_* contain only training block performance.
- Other *.mat* files with the suffix *_test_blocks_* contain only test block performance.
  - Among them, *.mat* files with the suffix *_noexclusion* do not apply the exclusion criterion. Hence they include all participants.
