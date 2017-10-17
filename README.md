# bbFMR

**bbFMR has been renamed to pybbfmr and is maintained with 
major updates at https://gitlab.com/wmi/pybbfmr/.**

This repository contains a python package to load, process and
model 2D (field-frequency) broadband ferromagnetic resonance 
(hence the name bbFMR) measurements.

## Usage
The basic concept is to load the measurement data stored in the
binary NI-TDMS file format using one of the Measurement classes.
(The base Measurement class can also be fed directly with 2D 
data.) Several processing operations are included that can be 
applied to the data by using Measurement().add_operation().
The data can be plotted (using matplotlib) by using 
Measurement().plot().

Several lmfit models for fitting the data to bbFMR models
such as the Polder susceptibility and various damping and 
dispersion models.

Finally, a graphical user interface to visualize and process 
the 2D data using guidata and guiqwt is included.

See the data [1] of [2] for a usage example and for a detailed
description of the physics behind the modeling and the 
"derivative divide" processing method in particular. 

## Requirements
The following packages are required. (The versions in brackets are the tested versions. Other versions will probably work just as well):

  + Python (3.5.2)
  + lmfit (0.9.5)
  + matplotlib (1.5.3)
  + npTDMS (0.8.2)
  + numpy (1.11.2)
  + scipy (0.18.1)
  + guiqwt (3.0.3)
  + guidata (1.7.6)

## Contribute
Please use the issue tracker to report problems and suggest changes 
and new features. Get in touch if you want to know more about the 
package.

## References

  1. H. Maier-Flaig, “Analysis of broadband CPW in the frequency domain - dataset and reference implemenation of derivative divide,” (2017), [https://osf.io/u27sf/...](https://osf.io/u27sf/?view_only=bc9d8bd783324875960eab1e0286e77a)
  2. Hannes Maier-Flaig, Sebastian T. B. Goennenwein, Ryo Ohshima, Masashi Shiraishi, Rudolf Gross, Hans Huebl: “Analysis of broadband ferromagnetic resonance in the frequency domain”, 2017; [arXiv:1705.05694](http://arxiv.org/abs/1705.05694).
  
