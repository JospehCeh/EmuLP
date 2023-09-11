# EmuLP
---
Rudimentary photo-z estimator in python for specific needs related to FORS2 studies.

---
This software uses a $\chi^2$ function to estimate the redshift from photometry.
1. Configure the run in a JSON file
2. Load the input data sample
Both of these steps can be facilitated thanks to a notebook, such as `Check_and_prepare_inputs.ipynb` - example of COSMOS2020 data subset limited to HSc photometry.
3. Run software from parent directory with the command `python -m EmuLP {config_file_location}/{config_file_name}.json`. _E.g._ `python -m EmuLP EmuLP/COSMOS2020-with-FORS2-HSC_only-jax-CC-TogglePriorTrue.json`.

---
This software makes use of JAX and therefore is best used on GPUs.
In addition to point estimates of photo-z, its main goal is to provide a multi-dimensional array of $\chi^2$ values to be interpreted and analysed as probability distribution functions, mainly to study the influence of various parameters on the photometric redshift estimation performances.

---
Other estimation methods, such as Gaussian Processes, may be available in the future.


